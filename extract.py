import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from videochat_flash.modeling_videochat_flash import VideoChatFlashQwenForCausalLM
import glob
import os
from decord import VideoReader

parser = argparse.ArgumentParser(description="Easy video feature extractor")
parser.add_argument('--dataset', default='DiDeMo', type=str, choices=['DiDeMo', 'ActivityNet', 'LSMDC', 'MSRVTT'], help='dataset')
parser.add_argument("--model_path", type=str, default="./pretrained/VideoChat-Flash-Qwen2-7B_res448", help="model path")
parser.add_argument("--num_frames", type=int, default=16, help="max number of frames")
parser.add_argument("--num_chunk", required=True, type=int, help="number of chunks")
parser.add_argument("--chunk_idx", required=True, type=int, help="chunk idx")
parser.add_argument("--batch_size", type=int, default=1, help="batch size for extraction")
parser.add_argument("--save_iter", type=int, default=10, help="save interval")
parser.add_argument("--clear", action='store_true', help="clear the feature folder")
args = parser.parse_args()

if args.clear:
    existing_features = glob.glob(f'./data/{args.dataset}/features/*.pth')
    for feature in existing_features:
        os.remove(feature)
    print('clear the feature folder!!!')

class VideoDataset(Dataset):
    def __init__(self, video_list, args=None, image_processor=None):
        self.video_list = video_list
        self.args = args
        self.image_processor = image_processor
    
    def __len__(self):
        return len(self.video_list)
    
    def load_video(self, video_path, num_frames=16):
        frames, frame_indices, fps, duration = self.read_frames_decord(video_path=video_path, num_frames=num_frames)
        sec = [str(round(f / fps, 1)) for f in frame_indices]
        msg = f"\nThe video lasts for {duration:.2f} seconds, and {len(sec)} frames are uniformly sampled from it. "
        return frames, msg
    
    def read_frames_decord(self, video_path, num_frames):
        video_reader = VideoReader(video_path, num_threads=1)
        
        vlen = len(video_reader)
        fps = video_reader.get_avg_fps()
        duration = vlen / float(fps)
        
        if duration > 30 and self.args.dataset == "DiDeMo":
            vlen = 30 * fps
            duration = 30.
        
        frame_indices = np.linspace(0, vlen - 2, num_frames, dtype=int)
        frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), torch.uint8
        video_reader.seek(0)
        
        frames = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half()
        return frames, frame_indices, float(fps), duration
    
    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        
        if self.args.dataset == 'LSMDC':
            vid = os.path.basename(video_path)[:-4]
        else:
            vid = os.path.basename(video_path).split(".")[0]

        try:
            video, time_msg = self.load_video(video_path, num_frames=self.args.num_frames)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return self.__getitem__(idx + 1)
        return vid, video
        
if args.dataset in ['LSMDC']:
    video_list = glob.glob(f'./data/{args.dataset}/videos/*/*')
else:
    video_list = glob.glob(f'./data/{args.dataset}/videos/*')
video_list.sort()
print(f"Number of videos: {len(video_list)}")

chunk_size = len(video_list) // args.num_chunk
chunk_start = chunk_size * args.chunk_idx
if args.chunk_idx == args.num_chunk - 1:
    chunk_end = len(video_list)
else:
    chunk_end = min(chunk_size * (args.chunk_idx + 1), len(video_list))
video_list = video_list[chunk_start:chunk_end]
print(f"num_chunk: {args.num_chunk}")
print(f"chunk_size: {chunk_end - chunk_start:,}")
print(f"{args.chunk_idx}-th chunk: {chunk_start:,} to {chunk_end:,}")
print(f'Using Batch size: {args.batch_size}')
    
model = VideoChatFlashQwenForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).half().to("cuda")
image_processor = model.get_vision_tower().image_processor

dataset = VideoDataset(video_list, args=args, image_processor=image_processor)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)

with torch.no_grad():
    for i, (vid, video) in enumerate(tqdm(dataloader)):
        bs = len(vid)
        video = [v.to("cuda") for v in video]
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            video_features = model.encode_video_image(video, [j for j in range(bs)], return_video_feature=True)
            video_features = [v.cpu().half() for v in video_features]
            for v, feature in zip(vid, video_features):
                torch.save(feature, f'./data/{args.dataset}/features/{v}.pth')