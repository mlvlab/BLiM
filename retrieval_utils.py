import datetime
import logging
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
import util.misc as misc
import math
from pathlib import Path
from videochat_flash.conversation import IGNORE_INDEX, IMAGE_TOKEN_ID

logger = logging.getLogger(__name__)

class VTGCriterion(nn.Module):
    def __init__(self):
        super(VTGCriterion, self).__init__()
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.loss_fct(shift_logits, shift_labels).reshape(logits.shape[0], -1)
        loss = loss.sum(1) / loss.bool().sum(1)
        return -loss

class TVGCriterion(nn.Module):
    def __init__(self):
        super(TVGCriterion, self).__init__()
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).reshape(logits.shape[0], -1)
        loss = loss.mean(1)
        return -loss

vtg_criterion = VTGCriterion()
tvg_criterion = TVGCriterion()

def compute_v2t_scores_x(v2t_scores_x, iterator, start, input_ids, attention_masks, labels, video, video_vocab, tvg_video_labels, model, device, args, forward_type=None, cpn=False):
    for i, sims in enumerate(iterator):
        k = min(len(sims), args.topk)
        bs = args.batch_size_eval
        topk_sim, topk_idx = sims.topk(k=k, dim=0)
        topk_idx = topk_idx.cpu()

        encoder_output = video[start + i].to(device, non_blocking=True)
        vtm_scores = []
        if len(topk_idx) % bs != 0:
            left = len(topk_idx) % bs
            left_encoder_output = [encoder_output.to(device) for _ in range(left)]
        encoder_output = [encoder_output.to(device) for _ in range(bs)]

        for j in range(0, len(topk_idx), bs):
            if j + bs > len(topk_idx):
                if forward_type == "vtg":
                    vtg_labels = labels[topk_idx[j:]].to(device)
                    (vtg_inputs, _, (vtg_masks, cpn_vtg_masks), _, vtg_embeds, vtg_labels) = model.module.prepare_inputs_labels_for_multimodal(input_ids[topk_idx[j:]].to(device), None,
                                                                                                                              attention_masks[topk_idx[j:]].to(device), None,
                                                                                                                              vtg_labels, left_encoder_output, ["video" for _ in range(left)],
                                                                                                                              image_sizes=[(448, 448) for _ in range(left)], video_feature=True, cpn=True)
                if forward_type == "tvg":
                    tvg_labels = labels[topk_idx[j:]].to(device)
                    (tvg_inputs, _, (tvg_masks, cpn_tvg_masks), _, tvg_embeds, tvg_labels) = model.module.prepare_inputs_labels_for_multimodal(input_ids[topk_idx[j:]].to(device), None,
                                                                                                                              attention_masks[topk_idx[j:]].to(device), None,
                                                                                                                              tvg_labels, left_encoder_output, ["video" for _ in range(left)],
                                                                                                                              image_sizes=[(448, 448) for _ in range(left)], video_feature=True, tvg=True, cpn=True)
                    repeat_n = left
            else:
                if forward_type == "vtg":
                    vtg_labels = labels[topk_idx[j:j + bs]].to(device)
                    (vtg_inputs, _, (vtg_masks, cpn_vtg_masks), _, vtg_embeds, vtg_labels) = model.module.prepare_inputs_labels_for_multimodal(input_ids[topk_idx[j:j + bs]].to(device), None,
                                                                                                                              attention_masks[topk_idx[j:j + bs]].to(device), None,
                                                                                                                              vtg_labels, encoder_output, ["video" for _ in range(bs)],
                                                                                                                              image_sizes=[(448, 448) for _ in range(bs)], video_feature=True, cpn=True)
                if forward_type == "tvg":
                    tvg_labels = labels[topk_idx[j:j + bs]].to(device)
                    (tvg_inputs, _, (tvg_masks, cpn_tvg_masks), _, tvg_embeds, tvg_labels) = model.module.prepare_inputs_labels_for_multimodal(input_ids[topk_idx[j:j + bs]].to(device), None,
                                                                                                                              attention_masks[topk_idx[j:j + bs]].to(device), None,
                                                                                                                              tvg_labels, encoder_output, ["video" for _ in range(bs)],
                                                                                                                              image_sizes=[(448, 448) for _ in range(bs)], video_feature=True, tvg=True, cpn=True)
                    repeat_n = bs
            if forward_type == "vtg":
                if cpn:
                    vtg_outputs = model(inputs_embeds=vtg_embeds, attention_mask=cpn_vtg_masks)
                else:
                    vtg_outputs = model(inputs_embeds=vtg_embeds, attention_mask=vtg_masks)
                scores = vtg_criterion(vtg_outputs.logits, vtg_labels)
                vtm_scores.append(scores)
            if forward_type == "tvg":
                visual_token_indices = (tvg_labels == IMAGE_TOKEN_ID).nonzero()[:, 1][:, None].repeat(1, args.num_clips) + (torch.arange(args.num_clips) - (args.num_clips + 1)).to(device)
                if cpn:
                    tvg_outputs = model(inputs_embeds=tvg_embeds, attention_mask=cpn_tvg_masks)
                else:
                    tvg_outputs = model(inputs_embeds=tvg_embeds, attention_mask=tvg_masks)
                visual_token_embeds = torch.gather(tvg_outputs.hidden_states, 1, visual_token_indices[..., None].repeat(1, 1, tvg_outputs.hidden_states.shape[-1]))
                visual_token_embeds = model.module.forward_visual(visual_token_embeds)
                tvg_logits = torch.bmm(visual_token_embeds.permute(1, 0, 2), video_vocab.permute(1, 2, 0)).transpose(0, 1) / math.sqrt(video_vocab.shape[-1])
                scores = tvg_criterion(tvg_logits, tvg_video_labels[start + i].repeat(repeat_n, args.num_clips).to(device))
                vtm_scores.append(scores)
        vtm_scores = torch.cat(vtm_scores, dim=0)
        v2t_scores_x[start + i, topk_idx] = vtm_scores.to(v2t_scores_x.dtype)
    return v2t_scores_x

def compute_t2v_scores_x(t2v_scores_x, iterator, start, input_ids, attention_masks, labels, video, video_vocab, tvg_video_labels, model, device, args, forward_type=None, cpn=False):
    for i, sims in enumerate(iterator):
        k = min(len(sims), args.topk)
        bs = args.batch_size_eval
        topk_sim, topk_idx = sims.topk(k=k, dim=0)
        topk_idx = topk_idx.cpu()

        vtm_scores = []
        for j in range(0, len(topk_idx), bs):
            encoder_output = [video[k].to(device) for k in topk_idx[j:j + bs]]
            repeat_n = len(encoder_output)
            if forward_type == "vtg":
                vtg_labels = labels[start + i].repeat(repeat_n, 1).to(device)
                (vtg_inputs, _, (vtg_masks, cpn_vtg_masks), _, vtg_embeds, vtg_labels) = model.module.prepare_inputs_labels_for_multimodal(input_ids[start + i].repeat(repeat_n, 1).to(device), None,
                                                                                                                          attention_masks[start + i].repeat(repeat_n, 1).to(device), None,
                                                                                                                          vtg_labels, encoder_output, ["video" for _ in range(repeat_n)],
                                                                                                                          image_sizes=[(448, 448) for _ in range(repeat_n)], video_feature=True, cpn=True)
                if cpn:
                    vtg_outputs = model(inputs_embeds=vtg_embeds, attention_mask=cpn_vtg_masks)
                else:
                    vtg_outputs = model(inputs_embeds=vtg_embeds, attention_mask=vtg_masks)
                scores = vtg_criterion(vtg_outputs.logits, vtg_labels)
            if forward_type == "tvg":
                tvg_labels = labels[start + i].repeat(repeat_n, 1).to(device)
                (tvg_inputs, _, (tvg_masks, cpn_tvg_masks), _, tvg_embeds, tvg_labels) = model.module.prepare_inputs_labels_for_multimodal(input_ids[start + i].repeat(repeat_n, 1).to(device), None,
                                                                                                                          attention_masks[start + i].repeat(repeat_n, 1).to(device), None,
                                                                                                                          tvg_labels, encoder_output, ["video" for _ in range(repeat_n)],
                                                                                                                          image_sizes=[(448, 448) for _ in range(repeat_n)], video_feature=True, tvg=True, cpn=True)
                visual_token_indices = (tvg_labels == IMAGE_TOKEN_ID).nonzero()[:, 1][:, None].repeat(1, args.num_clips) + (torch.arange(args.num_clips) - (args.num_clips + 1)).to(device)
                if cpn:
                    tvg_outputs = model(inputs_embeds=tvg_embeds, attention_mask=cpn_tvg_masks)
                else:
                    tvg_outputs = model(inputs_embeds=tvg_embeds, attention_mask=tvg_masks)
                visual_token_embeds = torch.gather(tvg_outputs.hidden_states, 1, visual_token_indices[..., None].repeat(1, 1, tvg_outputs.hidden_states.shape[-1]))
                visual_token_embeds = model.module.forward_visual(visual_token_embeds)
                tvg_logits = torch.bmm(visual_token_embeds.permute(1, 0, 2), video_vocab.permute(1, 2, 0)).transpose(0, 1) / math.sqrt(video_vocab.shape[-1])
                scores = tvg_criterion(tvg_logits, tvg_video_labels[topk_idx[j : j + bs]][:, None].repeat(1, args.num_clips).to(device))
            vtm_scores.append(scores)
        vtm_scores = torch.cat(vtm_scores, dim=0)
        t2v_scores_x[start + i, topk_idx] = vtm_scores.to(t2v_scores_x.dtype)
    return t2v_scores_x

def padding_ids(input_ids, labels, masks, tokenizer=None):
    num_samples = len(input_ids)
    max_len = max([len(ids) for ids in input_ids])
    input_ids_padded = torch.full((num_samples, max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels_padded = torch.full((num_samples, max_len), IGNORE_INDEX, dtype=torch.long)
    masks_padded = torch.full((num_samples, max_len), 0, dtype=torch.long)

    for i in range(num_samples):
        cur_len = len(input_ids[i])
        input_ids_padded[i, -cur_len:] = input_ids[i]
        labels_padded[i, -cur_len:] = labels[i]
        masks_padded[i, -cur_len:] = masks[i]
    return input_ids_padded, labels_padded, masks_padded

@torch.no_grad()
def evaluation(model, data_loader, device, tokenizer, args):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    print("Extracting features...")
    start_time = time.time()

    print_freq = int(len(data_loader) / 2)

    video, tvg_video_labels = [], []
    vtg_ids, vtg_labels, vtg_masks = [], [], []
    tvg_ids, tvg_labels, tvg_masks = [], [], []
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, 'Extracting:')):
        video += [v for v in data["video"]]

        vtg_ids += data["vtg_ids"]
        vtg_labels += data["vtg_labels"]
        vtg_masks += data["vtg_masks"]

        tvg_ids += data["tvg_ids"]
        tvg_labels += data["tvg_labels"]
        tvg_masks += data["tvg_masks"]

        tvg_video_labels.append(data["tvg_video_labels"])

    vtg_ids, vtg_labels, vtg_masks = padding_ids(vtg_ids, vtg_labels, vtg_masks, tokenizer)
    tvg_ids, tvg_labels, tvg_masks = padding_ids(tvg_ids, tvg_labels, tvg_masks, tokenizer)
    tvg_video_labels = torch.cat(tvg_video_labels, dim=0)

    if args.resume == '' and args.eval:
        scores = torch.load(f"./scores/{args.dataset.lower()}_zeroshot.pth", weights_only=True)
    else:
        scores = torch.load(f"./scores/{args.dataset.lower()}.pth", weights_only=True)
    v2t_iv2_scores, t2v_iv2_scores = scores["v2t"], scores["t2v"]
    num_texts, num_videos = t2v_iv2_scores.shape

    print("Rerank InternVideo2 results with BLiM...")
    num_tasks = misc.get_world_size()
    rank = misc.get_rank()
    video_vocab = data_loader.dataset.video_vocab.cuda()
    model.module.set_tvg_prefix_length(data_loader.dataset.tvg_prefix_length)

    # compute video2text #
    step = num_videos // num_tasks + 1
    start = rank * step
    end = min(num_videos, start + step)

    print(f"V2T shape: {v2t_iv2_scores[start:end].shape}")
    iterator = metric_logger.log_every(v2t_iv2_scores[start:end], 100, 'Calculating V2T (Candidate Likelihood; VTG):')
    v2t_candidate_likelihood = torch.full((num_videos, num_texts), -100.0).to(device, torch.float, non_blocking=True)
    v2t_candidate_likelihood = compute_v2t_scores_x(v2t_candidate_likelihood, iterator, start, vtg_ids, vtg_masks, vtg_labels, video, video_vocab, tvg_video_labels, model, device, args, forward_type="vtg")

    if args.cpn:
        iterator = metric_logger.log_every(v2t_iv2_scores[start:end], 100, 'Calculating V2T (Candidate Likelihood; VTG) CPN:')
        v2t_candidate_prior = torch.full((num_videos, num_texts), -100.0).to(device, torch.float, non_blocking=True)
        v2t_candidate_prior = compute_v2t_scores_x(v2t_candidate_prior, iterator, start, vtg_ids, vtg_masks, vtg_labels, video, video_vocab, tvg_video_labels, model, device, args, forward_type="vtg", cpn=True)

    if args.resume != "" or not args.eval:
        iterator = metric_logger.log_every(v2t_iv2_scores[start:end], 100, 'Calculating V2T (Query Likelihood; TVG):')
        v2t_query_likelihood = torch.full((num_videos, num_texts), -100.0).to(device, torch.float, non_blocking=True)
        v2t_query_likelihood = compute_v2t_scores_x(v2t_query_likelihood, iterator, start, tvg_ids, tvg_masks, tvg_labels, video, video_vocab, tvg_video_labels, model, device, args, forward_type="tvg")

    # compute text2video #
    step = num_texts // num_tasks + 1
    start = rank * step
    end = min(num_texts, start + step)

    print(f"T2V shape: {t2v_iv2_scores[start:end].shape}")
    iterator = metric_logger.log_every(t2v_iv2_scores[start:end], 100, 'Calculating T2V (Query Likelihood; VTG):')
    t2v_query_likelihood = torch.full((num_texts, num_videos), -100.0).to(device, torch.float, non_blocking=True)
    t2v_query_likelihood = compute_t2v_scores_x(t2v_query_likelihood, iterator, start, vtg_ids, vtg_masks, vtg_labels, video, video_vocab, tvg_video_labels, model, device, args, forward_type="vtg")

    if args.resume != "" or not args.eval:
        iterator = metric_logger.log_every(t2v_iv2_scores[start:end], 100, 'Calculating T2V (Candidate Likelihood; TVG):')
        t2v_candidate_likelihood = torch.full((num_texts, num_videos), -100.0).to(device, torch.float, non_blocking=True)
        t2v_candidate_likelihood = compute_t2v_scores_x(t2v_candidate_likelihood, iterator, start, tvg_ids, tvg_masks, tvg_labels, video, video_vocab, tvg_video_labels, model, device, args, forward_type="tvg")

        if args.cpn:
            iterator = metric_logger.log_every(t2v_iv2_scores[start:end], 100, 'Calculating T2V (Candidate Likelihood; TVG) CPN:')
            t2v_candidate_prior = torch.full((num_texts, num_videos), -100.0).to(device, torch.float, non_blocking=True)
            t2v_candidate_prior = compute_t2v_scores_x(t2v_candidate_prior, iterator, start, tvg_ids, tvg_masks, tvg_labels, video, video_vocab, tvg_video_labels, model, device, args, forward_type="tvg", cpn=True)

    if args.distributed:
        dist.barrier()
        dist.all_reduce(v2t_candidate_likelihood, op=dist.ReduceOp.SUM)
        dist.all_reduce(t2v_query_likelihood, op=dist.ReduceOp.SUM)
        if args.resume != "" or not args.eval:
            dist.all_reduce(t2v_candidate_likelihood, op=dist.ReduceOp.SUM)
            dist.all_reduce(v2t_query_likelihood, op=dist.ReduceOp.SUM)
        if args.cpn:
            dist.all_reduce(v2t_candidate_prior, op=dist.ReduceOp.SUM)
            if args.resume != "" or not args.eval:
                dist.all_reduce(t2v_candidate_prior, op=dist.ReduceOp.SUM)
    
    t2v_dict, v2t_dict = {}, {}
    if args.resume != "" or not args.eval:
        t2v_dict["candidate_likelihood"] = t2v_candidate_likelihood.cpu().numpy()
    t2v_dict["query_likelihood"] = t2v_query_likelihood.cpu().numpy()
    t2v_dict["internvideo2"] = t2v_iv2_scores.cpu().numpy()
    v2t_dict["candidate_likelihood"] = v2t_candidate_likelihood.cpu().numpy()
    if args.resume != "" or not args.eval:
        v2t_dict["query_likelihood"] = v2t_query_likelihood.cpu().numpy()
    v2t_dict["internvideo2"] = v2t_iv2_scores.cpu().numpy()
    if args.cpn:
        if args.resume != "" or not args.eval:
            t2v_dict["candidate_prior"] = t2v_candidate_prior.cpu().numpy()
        v2t_dict["candidate_prior"] = v2t_candidate_prior.cpu().numpy()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Evaluation time {total_time_str}")
    return t2v_dict, v2t_dict