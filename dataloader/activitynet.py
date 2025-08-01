import json
from .base_dataset import BaseDataset

class ActivityNet(BaseDataset):
    def __init__(self, args=None, tokenizer=None, image_processor=None, split='train'):
        super().__init__(args, tokenizer=tokenizer, image_processor=image_processor, split=split)
        if split == 'train':
            self.annotations = json.load(open(f'./data/{args.dataset}/anet_ret_train.json'))
        else:
            self.annotations = json.load(open(f'./data/{args.dataset}/anet_ret_val_1.json'))
        
        self.data = []
        for anno in self.annotations:
            vid = anno['video'].split('.')[0]
            if self.split == "test" or (f"./data/{args.dataset}/features/{vid}.pth" in self.features and self.split == "train"):
                self.data.append({"vid": vid, "text": ("".join(anno["caption"]).strip())})
        
        self.vids, self.video_vocab = self.get_video_vocab()
        print(f'num {split} data: {len(self.data)}/{len(self.annotations)}')