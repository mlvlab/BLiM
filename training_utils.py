import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import util.misc as misc
import util.metrics as metrics
import util.lr_sched as lr_sched
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from retrieval_utils import evaluation
from videochat_flash.conversation import IMAGE_TOKEN_ID
    
class VTGCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels, reduction="mean"):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens

        shift_logits = shift_logits.view(-1, logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction=reduction)

        if reduction == "none":
            loss = loss.reshape(logits.shape[0], -1)
            loss = loss.sum(1) / loss.bool().sum(1)
        return loss

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    dist.barrier()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) / 4)
    accum_iter = args.accum_iter

    optimizer.zero_grad()
    video_vocab = data_loader.dataset.video_vocab.to(device)
    model.module.set_video_vocab(video_vocab)
    model.module.set_tvg_prefix_length(data_loader.dataset.tvg_prefix_length)
    vtg_criterion = VTGCriterion()
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            bs = len(data["video"])
            video = [v.to(device) for v in data["video"]]
            modalities = ["video" for _ in range(bs)]
            image_sizes = [(448, 448) for _ in range(bs)]

            vtg_inputs, vtg_masks, vtg_labels = data["vtg_ids"].to(device), data["vtg_masks"].to(device), data["vtg_labels"].to(device)
            tvg_inputs, tvg_masks, tvg_labels = data["tvg_ids"].to(device), data["tvg_masks"].to(device), data["tvg_labels"].to(device)

            (vtg_inputs, _, (vtg_masks, cpn_vtg_masks), _, vtg_embeds, vtg_labels) = model.module.prepare_inputs_labels_for_multimodal(vtg_inputs, None, vtg_masks, None, vtg_labels, video, modalities, image_sizes=image_sizes, video_feature=True, cpn=True)

            vtg_outputs = model(inputs_embeds=vtg_embeds, attention_mask=vtg_masks)
            vtg_loss = vtg_criterion(vtg_outputs.logits, vtg_labels)

            (tvg_inputs, _, (tvg_masks, cpn_tvg_masks), _, tvg_embeds, tvg_labels) = model.module.prepare_inputs_labels_for_multimodal(tvg_inputs, None, tvg_masks, None, tvg_labels, video, modalities, image_sizes=image_sizes, video_feature=True, tvg=True, cpn=True)
            visual_token_indices = (tvg_labels == IMAGE_TOKEN_ID).nonzero()[:, 1][:, None].repeat(1, args.num_clips) + (torch.arange(args.num_clips) - (args.num_clips + 1)).to(device)
            tvg_video_labels = data["tvg_video_labels"][:, None].repeat(1, args.num_clips).to(device)

            tvg_outputs = model(inputs_embeds=tvg_embeds, attention_mask=tvg_masks)
            visual_token_embeds = torch.gather(tvg_outputs.hidden_states, 1, visual_token_indices[..., None].repeat(1, 1, tvg_outputs.hidden_states.shape[-1]))
            visual_token_embeds = model.module.forward_visual(visual_token_embeds)
            tvg_logits = torch.bmm(visual_token_embeds.permute(1, 0, 2), video_vocab.permute(1, 2, 0)).transpose(0, 1) / math.sqrt(video_vocab.shape[-1])
            tvg_loss = F.cross_entropy(tvg_logits.reshape(-1, tvg_logits.shape[-1]), tvg_video_labels.reshape(-1))

            loss = vtg_loss + tvg_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()


        metric_logger.update(loss=loss.item())
        metric_logger.update(vtg_loss=vtg_loss.item())
        metric_logger.update(tvg_loss=tvg_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def calculate_score(t2v_1, v2t_1, t2v_2, v2t_2, t2v_ids, v2t_ids):
    best_v2t, v2t_c = 0, 0
    best_t2v, t2v_c = 0, 0
    for c in tqdm(np.linspace(0, 1, 11)):
        res = get_recall(c * t2v_1 + (1 - c) * t2v_2, c * v2t_1 + (1 - c) * v2t_2, t2v_ids, v2t_ids)

        if best_v2t < res["v2t_r1"]:
            best_v2t = res["v2t_r1"]
            v2t_c = round(float(c), 1)
        if best_t2v < res["t2v_r1"]:
            best_t2v = res["t2v_r1"]
            t2v_c = round(float(c), 1)

    v2t = v2t_c * v2t_1 + (1 - v2t_c) * v2t_2
    t2v = t2v_c * t2v_1 + (1 - t2v_c) * t2v_2
    return t2v, v2t, t2v_c, v2t_c

def calculate_cpn_score(t2v, v2t, t2v_prior, v2t_prior, t2v_ids, v2t_ids):
    best_v2t, v2t_c = 0, 0
    best_t2v, t2v_c = 0, 0
    for c in tqdm(np.linspace(0, 1, 11)):
        res = get_recall(t2v - c * t2v_prior, v2t - c * v2t_prior, t2v_ids, v2t_ids)

        if best_v2t < res["v2t_r1"]:
            best_v2t = res["v2t_r1"]
            v2t_c = round(float(c), 1)
        if best_t2v < res["t2v_r1"]:
            best_t2v = res["t2v_r1"]
            t2v_c = round(float(c), 1)

    v2t = v2t - v2t_c * v2t_prior
    t2v = t2v - t2v_c * t2v_prior
    return t2v, v2t, t2v_c, v2t_c

def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, tokenizer=None, args=None):
    prefix = 'test'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        t2v_dict, v2t_dict = evaluation(model, data_loader, device, tokenizer, args)
    
    if misc.is_main_process():
        v2t_ids = {i: i for i in range(len(data_loader.dataset))}
        t2v_ids = {i: i for i in range(len(data_loader.dataset))}

        results = dict()
        names = ["internvideo2", "candidate_likelihood", "query_likelihood", "cpn_candidate_likelihood", "blim"]
        for name in names:
            if name == "cpn_candidate_likelihood":
                if args.cpn:
                    cpn_t2v = t2v_dict["candidate_likelihood"] - args.alpha[0] * t2v_dict["candidate_prior"] if args.resume != "" or not args.eval else np.zeros((len(t2v_ids), len(v2t_ids)))
                    cpn_v2t = v2t_dict["candidate_likelihood"] - args.alpha[1] * v2t_dict["candidate_prior"]
                    results[name] = get_recall(cpn_t2v, cpn_v2t, t2v_ids, v2t_ids)
                else:
                    cpn_t2v = t2v_dict["candidate_likelihood"] if args.resume != "" or not args.eval else np.zeros((len(t2v_ids), len(v2t_ids)))
                    cpn_v2t = v2t_dict["candidate_likelihood"]
            elif name == "blim":
                blim_t2v = args.c[0] * t2v_dict["query_likelihood"] + (1 - args.c[0]) * cpn_t2v
                blim_v2t = args.c[1] * v2t_dict["query_likelihood"] + (1 - args.c[1]) * cpn_v2t if args.resume != "" or not args.eval else cpn_v2t
                blim_t2v = args.c[2] * blim_t2v + (1 - args.c[2]) * t2v_dict["internvideo2"]
                blim_v2t = args.c[3] * blim_v2t + (1 - args.c[3]) * v2t_dict["internvideo2"]
                results[name] = get_recall(blim_t2v, blim_v2t, t2v_ids, v2t_ids)
            else:
                results[name] = get_recall(t2v_dict.get(name, np.zeros((len(t2v_ids), len(v2t_ids)))), v2t_dict.get(name, np.zeros((len(v2t_ids), len(t2v_ids)))), t2v_ids, v2t_ids)

        return results
    
    
@torch.no_grad()
def get_recall(t2v, v2t, t2v_ids, v2t_ids):
    if np.count_nonzero(v2t == 0) != 0:
        v2t_r1, v2t_r5, v2t_r10 = 0., 0., 0.
    else:
    
        ranks = np.zeros(v2t.shape[0])
        for index, score in enumerate(v2t):
            inds = np.argsort(score)[::-1]
            gt_txt_ids = v2t_ids[index]
            if isinstance(gt_txt_ids, int):
                ranks[index] = np.where(inds == gt_txt_ids)[0][0]
            else:
                rank = 1e20
                for i in gt_txt_ids:
                    tmp = np.where(inds == i)[0][0]
                    if tmp < rank:
                        rank = tmp
                ranks[index] = rank
        v2t_r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        v2t_r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        v2t_r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    if np.count_nonzero(t2v == 0) != 0:
        t2v_r1, t2v_r5, t2v_r10 = 0., 0., 0.
    else:
        ranks = np.zeros(t2v.shape[0])
        for index, score in enumerate(t2v):
            inds = np.argsort(score)[::-1]
            gt_video_ids = t2v_ids[index]
            if isinstance(gt_video_ids, int):
                ranks[index] = np.where(inds == gt_video_ids)[0][0]
            else:
                rank = 1e20
                for i in gt_video_ids:
                    tmp = np.where(inds == i)[0][0]
                    if tmp < rank:
                        rank = tmp
                ranks[index] = rank
        t2v_r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        t2v_r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        t2v_r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    v2t_mean = (v2t_r1 + v2t_r5 + v2t_r10) / 3
    t2v_mean = (t2v_r1 + t2v_r5 + t2v_r10) / 3
    r_mean = (v2t_mean + t2v_mean) / 2

    eval_result = {"t2v_r1": t2v_r1, "t2v_r5": t2v_r5, "t2v_r10": t2v_r10, "t2v_r_mean": t2v_mean, "v2t_r1": v2t_r1, "v2t_r5": v2t_r5, "v2t_r10": v2t_r10, "v2t_r_mean": v2t_mean, "r_mean": r_mean}
    eval_result = {k: round(v, 2) for k, v in eval_result.items()}
    return eval_result