import os
import argparse
import datetime
import json
import time
import copy
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from training_utils import train_one_epoch, val_one_epoch
from dataloader import load_data

from transformers import AutoTokenizer
from videochat_flash.modeling_videochat_flash import VideoChatFlashQwenForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def get_args_parser():
    parser = argparse.ArgumentParser('BLiM', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--batch_size_eval', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--model_path', default='./pretrained/VideoChat-Flash-Qwen2-7B_res448', type=str, help='model path')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='DiDeMo', type=str, choices=['DiDeMo', 'ActivityNet', 'LSMDC', 'MSRVTT'], help='dataset')
    parser.add_argument('--output_dir', default='./checkpoint', help='path where to save, empty for no saving')
    parser.add_argument('--num_clips', default=4, type=int, help='number of clips in a video')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=8, help='attention bias')
    parser.add_argument('--lora_alpha', type=int, default=32, help='attention bias')
    parser.add_argument('--lora_drop', type=float, default=0.05, help='attention bias')
    
    # Inference parameters
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--topk', type=int, default=10, help='reranking top-k during inference')
    parser.add_argument('--cpn', action='store_true')
    parser.add_argument('--alpha', nargs='+', type=float, default=[0.0, 0.0], help='t2v and v2t cpn strength')
    parser.add_argument('--c', nargs='+', type=float, default=[0.0, 0.0, 0.0, 0.0], help='coefficients for ensemble')
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    global_rank = misc.get_rank()

    # define the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = VideoChatFlashQwenForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).half().to(device)
    image_processor = model.get_vision_tower().image_processor
    
    mlp_peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["0", "2"], lora_dropout=args.lora_drop, bias="none")
    model.model.mm_projector.mlp = get_peft_model(model.model.mm_projector.mlp, mlp_peft_config)
    model.model.mm_projector.tvg_mlp = copy.deepcopy(model.model.mm_projector.mlp)

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_drop, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    for n, p in model.named_parameters():
        if "visual_head" in n:
            p.requires_grad = True
            p.data = p.data.float()

    print('*' * 113)
    total_parameters = sum(p.numel() for p in model.parameters())
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total params: {total_parameters:,}')
    print(f'Trainable params: {n_parameters:,}')
    print('*' * 113)

    if not args.eval:
        data_loader_train = load_data(args, tokenizer=tokenizer, image_processor=image_processor, split='train')
    data_loader_val = load_data(args, tokenizer=tokenizer, image_processor=image_processor, split='test')

    if args.resume:
        state_dict = torch.load(args.resume, "cpu")
        model.load_state_dict(state_dict['model'], strict=False)
        assert sum(p.numel() for p in state_dict['model'].values()) == n_parameters

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    best_r1 = 0.

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    model._set_static_graph()

    if not args.eval:
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed and not args.eval:
            data_loader_train.sampler.set_epoch(epoch)

        if not args.eval:
            train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, args=args)
            if misc.is_main_process():
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name=f'epoch{epoch}')

        results = val_one_epoch(model, data_loader_val, optimizer, device, epoch, loss_scaler, tokenizer=tokenizer, args=args)
        if args.eval:
            if args.output_dir and misc.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write("\n\n" + pd.DataFrame(results).transpose().to_string())
            print("\n" + pd.DataFrame(results).transpose().to_string())
            sys.exit(0)


        if not args.eval and misc.is_main_process():
            cur_r1 = results['blim']['t2v_r1'] + results['blim']['v2t_r1']
            
            if args.output_dir and best_r1 < cur_r1:
                best_r1 = cur_r1
                model_name = 'checkpoint_best'
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name=model_name)

            log_stats = {'epoch': epoch, **{f'train_{k}': v for k, v in train_stats.items()}, **{f'val_{k}': v for k, v in results.items()}}

            if args.output_dir:
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

                    print("\n" + pd.DataFrame(results).transpose().to_string())
                    f.write(pd.DataFrame(results).transpose().to_string() + "\n")
                    if epoch == args.epochs - 1:
                        f.write('\n')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
