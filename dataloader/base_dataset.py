import copy
import torch
from torch.utils.data import Dataset
import glob
from videochat_flash.conversation import conv_templates, SeparatorStyle, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer=None, image_processor=None, split=None):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.features = glob.glob(f'./data/{args.dataset}/features/*.pth')
        self.split = split
        self.tvg_prefix_length = self.get_tvg_prefix_length("Generate a video given the caption.")
        
    def get_tvg_prefix_length(self, init_prompt):
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], f"{init_prompt}")
        tvg_prefix_length = len(self.tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")) - 2
        return tvg_prefix_length

    def load_video_feature(self, vid):
        if f"./data/{self.args.dataset}/features/{vid}.pth" not in self.features:
            video = torch.zeros(4, 64, 1024)
        else:
            video = torch.load(f"./data/{self.args.dataset}/features/{vid}.pth", weights_only=True)
        return video

    def get_video_vocab(self):
        vids = sorted(set([data["vid"] for data in self.data]))
        video_vocab = [self.load_video_feature(vid).mean(1) for vid in tqdm(vids)]
        video_vocab = torch.stack(video_vocab, dim=0)
        return vids, video_vocab

    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == "pt":
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f"Unsupported tensor type: {return_tensors}")
        return input_ids

    def get_vtg_id(self, item):
        if self.args.dataset in ["DiDeMo", "ActivityNet"]:
            init_prompt = "Describe this video in detail."
        elif self.args.dataset in ["LSMDC"]:
            init_prompt = "Describe this video in one sentence."
        elif self.args.dataset in ["MSRVTT"]:
            init_prompt = "Describe this video briefly."

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{init_prompt}")
        conv.append_message(conv.roles[1], None)
        prompt_ids = self.tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{init_prompt}")
        conv.append_message(conv.roles[1], item["text"])
        input_ids = self.tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        assert (prompt_ids != input_ids[:len(prompt_ids)]).sum() == 0

        labels = copy.deepcopy(input_ids)
        labels[:len(prompt_ids)] = IGNORE_INDEX

        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_masks

    def get_tvg_id(self, item):
        init_prompt = "Generate a video given the caption."

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], f"{init_prompt}\nCaption: {item['text']}")
        conv.append_message(conv.roles[1], None)
        prompt_ids = self.tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], f"{init_prompt}\nCaption: {item['text']}")
        conv.append_message(conv.roles[1], DEFAULT_IMAGE_TOKEN)
        input_ids = self.tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        assert (prompt_ids != input_ids[:len(prompt_ids)]).sum() == 0

        labels = copy.deepcopy(input_ids)
        labels[:len(prompt_ids)] = IGNORE_INDEX

        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_masks

    def __getitem__(self, idx):
        item = self.data[idx]
        video = self.load_video_feature(item["vid"])

        vtg_ids, vtg_labels, vtg_masks = self.get_vtg_id(item)
        tvg_ids, tvg_labels, tvg_masks = self.get_tvg_id(item)

        return {"vid": item["vid"], "video": video, "vtg_ids": vtg_ids, "vtg_labels": vtg_labels, "vtg_masks": vtg_masks, "tvg_ids": tvg_ids, "tvg_labels": tvg_labels, "tvg_masks": tvg_masks, "tvg_video_labels": self.vids.index(item["vid"])}

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        bs = len(batch)
        vid = [batch[i]["vid"] for i in range(bs)]
        video = [batch[i]["video"] for i in range(bs)]

        vtg_ids = [batch[i]["vtg_ids"] for i in range(bs)]
        vtg_labels = [batch[i]["vtg_labels"] for i in range(bs)]
        vtg_masks = [batch[i]["vtg_masks"] for i in range(bs)]

        tvg_ids = [batch[i]["tvg_ids"] for i in range(bs)]
        tvg_labels = [batch[i]["tvg_labels"] for i in range(bs)]
        tvg_masks = [batch[i]["tvg_masks"] for i in range(bs)]

        if self.split == 'train':
            max_len = max([len(ids) for ids in vtg_ids])
            vtg_ids_padded = torch.full((bs, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
            vtg_labels_padded = torch.full((bs, max_len), IGNORE_INDEX, dtype=torch.long)
            vtg_masks_padded = torch.full((bs, max_len), 0, dtype=torch.long)
            for i in range(bs):
                cur_len = len(vtg_ids[i])
                vtg_ids_padded[i, -cur_len:] = vtg_ids[i]
                vtg_labels_padded[i, -cur_len:] = vtg_labels[i]
                vtg_masks_padded[i, -cur_len:] = vtg_masks[i]

            max_len = max([len(ids) for ids in tvg_ids])
            tvg_ids_padded = torch.full((bs, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
            tvg_labels_padded = torch.full((bs, max_len), IGNORE_INDEX, dtype=torch.long)
            tvg_masks_padded = torch.full((bs, max_len), 0, dtype=torch.long)
            for i in range(bs):
                cur_len = len(tvg_ids[i])
                tvg_ids_padded[i, -cur_len:] = tvg_ids[i]
                tvg_labels_padded[i, -cur_len:] = tvg_labels[i]
                tvg_masks_padded[i, -cur_len:] = tvg_masks[i]
        else:
            vtg_ids_padded = [batch[i]["vtg_ids"] for i in range(bs)]
            vtg_labels_padded = [batch[i]["vtg_labels"] for i in range(bs)]
            vtg_masks_padded = [batch[i]["vtg_masks"] for i in range(bs)]

            tvg_ids_padded = [batch[i]["tvg_ids"] for i in range(bs)]
            tvg_labels_padded = [batch[i]["tvg_labels"] for i in range(bs)]
            tvg_masks_padded = [batch[i]["tvg_masks"] for i in range(bs)]

        tvg_video_labels = torch.tensor([self.vids.index(batch[i]["vid"]) for i in range(bs)])

        return {"vid": vid, "video": video, "vtg_ids": vtg_ids_padded, "vtg_labels": vtg_labels_padded, "vtg_masks": vtg_masks_padded, "tvg_ids": tvg_ids_padded, "tvg_labels": tvg_labels_padded, "tvg_masks": tvg_masks_padded, "tvg_video_labels": tvg_video_labels}
