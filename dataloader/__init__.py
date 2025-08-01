import torch
from util import misc
from .lsmdc import *
from .msrvtt import *
from .didemo import *
from .activitynet import *

def load_data(args, tokenizer=None, image_processor=None, split='train'):
    dataset = eval(args.dataset)(args=args, tokenizer=tokenizer, image_processor=image_processor, split=split)
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=(split == 'train'))
    if split == 'train':
        data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset.collate_fn, 
                                                  pin_memory=args.pin_mem, drop_last=False)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_eval, num_workers=args.num_workers, collate_fn=dataset.collate_fn, shuffle=False,
                                                  pin_memory=args.pin_mem, drop_last=False)
    return data_loader