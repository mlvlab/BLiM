#    Copyright 2024
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod
import re
import torch
import torch.nn as nn
import random
from typing import List, Optional, Tuple, Union, Dict

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import Qwen2Config

from .vision_tower_builder import build_vision_tower
from .mm_projector_builder import build_vision_projector

from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
from .conversation import conv_templates, SeparatorStyle
from .mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, get_anyres_image_grid_shape, load_video
from .modeling_qwen2_flash import Qwen2Model_Flash, Qwen2ForCausalLM_Flash


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
            if "nopad" in getattr(config, "mm_patch_merge_type", "") and getattr(self.config, "mm_newline_position", "nothing") != "nothing":
                self.frame_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()



        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
            if "nopad" in getattr(self.config, "mm_patch_merge_type", "") and getattr(self.config, "mm_newline_position", "nothing") != "nothing":
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.frame_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            if self.config.mm_projector_type =='lxh_qformer':
                incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"), strict=False)
            else:
                incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()


    def encode_video_image(self, images_list, video_idx_in_batch, return_video_feature=False, video_feature=False, tvg=False):
        # video encoder编码后按图像的connector处理
        bs = len(images_list)
        if not video_feature:
            concat_images = []
            concat_videos = []
            for idx, image in enumerate(images_list):
                if idx in video_idx_in_batch:
                    concat_videos.append(image)
                else:
                    concat_images.append(image)
            # print(concat_videos[0].shape)
            has_image = len(concat_images) > 0
            has_video = len(concat_videos) > 0

            mm_local_num_frames = getattr(self.config, "mm_local_num_frames", -1)
            assert mm_local_num_frames != -1
            if has_image:
                image_split_sizes = [image.shape[0] for image in concat_images]
                concat_images = torch.cat([image.unsqueeze(1) for image in concat_images], dim=0)
                # print("input vit image.shape:", concat_images.shape)
                images_features = self.get_model().get_vision_tower()(concat_images) # B_i, N, D
                images_features = torch.split(images_features, image_split_sizes)

            if has_video:
                video_split_sizes = [video.shape[0] // mm_local_num_frames for video in concat_videos]
                concat_videos = torch.cat([video.reshape(video.shape[0] // mm_local_num_frames, mm_local_num_frames, video.shape[1], video.shape[2], video.shape[3]) for video in concat_videos], dim=0)
                # print("input vit video.shape:", concat_videos.shape)
                videos_features = self.get_model().get_vision_tower()(concat_videos) # B_v, N, D
                videos_features = [v.reshape(-1, v.shape[-2] // mm_local_num_frames, v.shape[-1]) for v in torch.split(videos_features, video_split_sizes)]
        else:
            videos_features = [v.squeeze(0) for v in images_list]
            has_video, has_image = True, False

        all_videos_or_images_features = []
        img_idx = 0
        vid_idx = 0

        for idx in range(bs):

            if idx in video_idx_in_batch:
                feat = self.get_model().mm_projector(videos_features[vid_idx], compress=True, local_num_frames=getattr(self.config, "mm_local_num_frames", -1), return_video_feature=return_video_feature,
                                                     video_feature=video_feature, tvg=tvg)
                vid_idx += 1
            else:
                feat = self.get_model().mm_projector(images_features[img_idx], compress=False)
                img_idx += 1
            # print("video_idx_in_batch:", video_idx_in_batch)
            all_videos_or_images_features.append(feat)

        if has_video:
            assert vid_idx == len(videos_features), f"vid: {vid_idx} != {len(videos_features)}"
        if has_image:
            assert img_idx == len(images_features), f"img: {img_idx} != {len(images_features)}"

        return all_videos_or_images_features



    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, video_feature=False, tvg=False, cpn=False):
        assert type(modalities) is list, modalities

        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))


            vision_encode_type = getattr(self.config, "vision_encode_type", "image")
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            frame_aspect_ratio = getattr(self.config, "frame_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "nothing")


            if vision_encode_type == "video_image": # video backbone, process video with compress
                image_features = self.encode_video_image(images_list, video_idx_in_batch=video_idx_in_batch, video_feature=video_feature, tvg=tvg)
            else:
                raise NotImplementedError(vision_encode_type)

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):

                    if image_idx in video_idx_in_batch:  # video operations

                        if "anyres" in frame_aspect_ratio:
                            raise NotImplementedError
                        else:
                            frame_feature = image_feature

                        if "pad" in mm_patch_merge_type:
                            if mm_newline_position == 'one_token':
                                frame_feature = frame_feature.flatten(0, 1)
                                if "unpad" in mm_patch_merge_type:
                                    frame_feature = torch.cat((frame_feature, self.model.image_newline[None].to(frame_feature.device)), dim=0)
                                else:
                                    frame_feature = torch.cat((frame_feature, self.model.frame_newline[None].to(frame_feature.device)), dim=0)
                            elif mm_newline_position == 'nothing':
                                frame_feature = frame_feature.mean(1) if tvg else frame_feature.flatten(0, 1)
                            else:
                                raise NotImplementedError("add pad please!!")
                        else:
                            frame_feature = frame_feature.flatten(0, 1)

                        # print(f"final video frame_feature.shape: {frame_feature.shape}")
                        image_feature = frame_feature

                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        origin_size = image_feature.shape

                        height = width = self.get_model().mm_projector.num_image_patches_per_side
                        assert height * width == base_image_feature.shape[0], f"height:{height}, width: {width}, base_image_feature: {base_image_feature.shape}"

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if "anyres" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size, max_resolutions=None)
                            except Exception as e:
                                print(f"Error: {e}")
                                raise e
                                # num_patch_width, num_patch_height = 2, 2

                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError(image_aspect_ratio)
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            raise NotImplementedError
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            raise NotImplementedError
                        elif "unpad" in mm_patch_merge_type:
                            raise NotImplementedError
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            try:
                                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            except Exception as e:
                                raise ValueError(f"{num_patch_width} {num_patch_height} now: base_image_feature: {base_image_feature.shape}, {image_feature.shape}, image_sizes[image_idx]: {image_sizes[image_idx]}, origin_size: {origin_size}, {image_sizes[image_idx]}, {self.config.image_grid_pinpoints}, {vision_tower_image_size}")
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    # print(f"image/video_feature.shape: {image_feature.shape}")
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # raise NotImplementedError(f"images.shape={images.shape},  modalities={modalities}")
            image_features = self.encode_image(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # print(f"Total images len(image_features: {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        mm_llm_compress = getattr(self.config, "mm_llm_compress", False)

        if mm_llm_compress:
            self.model.llm_compress_type = getattr(self.config, "llm_compress_type", "attention")
            self.model.llm_compress_layer_list = getattr(self.config, "llm_compress_layer_list", [8, 16, 24])
            self.model.llm_image_token_ratio_list = getattr(self.config, "llm_image_token_ratio_list", [1.0, 0.5, 0.25, 0.125])
            first_image_token_position = []
            text_prompt_lens = []
        else:
            self.model.llm_compress_type = "attention"
            self.model.llm_compress_layer_list = []
            self.model.llm_image_token_ratio_list = []
            first_image_token_position = []
            text_prompt_lens = []

        # rank_print("Inserting Images embedding")
        cpn_attention_mask = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if mm_llm_compress:
                ####### copy from pdrop, only support single image/video NOTE ##################
                # record image position for further dropping
                image_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                assert len(image_index) == 1, f"Only support singe/video: {image_index}"
                if image_index == []:
                    first_image_token_position.append(-1)
                else:
                    first_image_token_position.append(image_index[0])


                # record input instruction length in inference mode
                if not self.training:
                    if image_index == []:
                        assert num_images == 0, num_images
                    else:
                        assert num_images == 1, f"num_images={num_images}"
                    text_prompt_lens.append(cur_input_ids.shape[0] - num_images)   # consider image place holder

                ###############################################

            # print(f"num_images={num_images}")
            if num_images == 0:
                try:
                    cur_image_features = image_features[cur_image_idx]
                except Exception as e:
                    print(f"Error cat: {e}")
                    cur_image_features = image_features[cur_image_idx - 1]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            # 151644,   8948,    198,   2610,    525,    264,  10950,  17847, 13, 151645,    198, 151644,    872,    198,  31115,    264,   2766,   2661, 279, 17256, 624
            cur_cpn_attention_mask = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                if tvg and i == 0:
                    _mask = torch.zeros_like(cur_labels_noim[i], dtype=attention_mask.dtype, device=attention_mask.device)
                    _mask[:self.tvg_prefix_length] = 1
                    cur_cpn_attention_mask.append(_mask)
                else:
                    cur_cpn_attention_mask.append(torch.ones_like(cur_labels_noim[i], dtype=attention_mask.dtype, device=attention_mask.device))

                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        print(f"cur_image_idx={cur_image_idx} is not ok")
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    if tvg:
                        cur_cpn_attention_mask.append(torch.ones((cur_image_features.shape[0],), dtype=attention_mask.dtype, device=attention_mask.device))
                    else:
                        cur_cpn_attention_mask.append(torch.zeros((cur_image_features.shape[0],), dtype=attention_mask.dtype, device=attention_mask.device))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]


            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_cpn_attention_mask = torch.cat(cur_cpn_attention_mask)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            cpn_attention_mask.append(cur_cpn_attention_mask)

        if mm_llm_compress:
            self.model.first_image_token_position = first_image_token_position
            self.model.text_prompt_lens = text_prompt_lens
            self.model.num_image_token_lens = [image_feature.shape[0] for image_feature in image_features]

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        _cpn_attention_mask = [x[:tokenizer_model_max_length] for x, modality in zip(cpn_attention_mask, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        cpn_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        # print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels, cur_cpn_attention_mask) in enumerate(zip(new_input_embeds, new_labels, _cpn_attention_mask)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    cpn_attention_mask[i, -cur_len:] = cur_cpn_attention_mask
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    cpn_attention_mask[i, :cur_len] = cur_cpn_attention_mask

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
            cpn_attention_mask = cpn_attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # print("Finish preparing")
        if cpn:
            return None, position_ids, (attention_mask, cpn_attention_mask), past_key_values, new_input_embeds, new_labels
        else:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False



class VideoChatFlashQwenConfig(Qwen2Config):
    model_type = "videochat_flash_qwen"


class VideoChatFlashQwenModel(LlavaMetaModel, Qwen2Model_Flash):
    config_class = VideoChatFlashQwenConfig

    def __init__(self, config: VideoChatFlashQwenConfig):
        super(VideoChatFlashQwenModel, self).__init__(config)


class VideoChatFlashQwenForCausalLM(LlavaMetaForCausalLM, Qwen2ForCausalLM_Flash):
    config_class = VideoChatFlashQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM_Flash.__init__(self, config)
        config.model_type = "videochat_flash_qwen"
        # config.rope_scaling = None

        self.model = VideoChatFlashQwenModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.visual_head = nn.Linear(config.hidden_size, config.mm_hidden_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_video_vocab(self, video_vocab):
        self.video_vocab = video_vocab
        
    def set_tvg_prefix_length(self, tvg_prefix_length):
        self.tvg_prefix_length = tvg_prefix_length

    def get_model(self):
        return self.model

    def forward_visual(self, visual_token_embeds):
        return self.visual_head(visual_token_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        # print("inputs_embeds.shape:", inputs_embeds.shape)
        if dpo_forward:
            raise NotImplementedError
        else:
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                                      use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,)
            return outputs


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            self.model.image_token_posi = [-1]
            self.model.prompt_len = None
            self.model.image_tokens = [0]
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    @torch.no_grad()
    def chat(self,
        video_path,
        tokenizer,
        user_prompt,
        chat_history=None,
        return_history=True,
        max_num_frames=512,
        media_dict=None,
        generation_config={}):

        frames, time_msg  = load_video(video_path, max_num_frames=max_num_frames, media_dict=media_dict)

        image_sizes = [frames[0].shape[:2]]

        frames = [self.get_vision_tower().image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()]

        conv = conv_templates["qwen_2"].copy()

        if chat_history is None or len(chat_history) == 0:
            user_prompt = f'{DEFAULT_IMAGE_TOKEN}\n{time_msg.strip()} {user_prompt}'
        else:
            assert DEFAULT_IMAGE_TOKEN in chat_history[0]['content'], chat_history
            for msg in chat_history:
                conv.append_message(msg['role'], msg['content'])

        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        if tokenizer.pad_token_id is None:
            if "qwen" in tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.generate(
                inputs=input_ids,
                images=frames,
                attention_mask=attention_masks,
                modalities=["video"],
                image_sizes=image_sizes,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                **generation_config
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]

        outputs = outputs.strip()

        # print(f"\033[91m== Question: \033[0m\n{prompt}\n")
        # print(f"\033[91m== Response: \033[0m\n{outputs}\n")

        if chat_history is None:
            chat_history = []

        chat_history.append({"role":conv.roles[0], "content":user_prompt})
        chat_history.append({"role":conv.roles[1], "content":outputs})
        if return_history:
            return outputs, chat_history
        else:
            return outputs



    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("videochat_flash_qwen", VideoChatFlashQwenConfig)
AutoModelForCausalLM.register(VideoChatFlashQwenConfig, VideoChatFlashQwenForCausalLM)