from PIL import Image
from io import BytesIO
import base64
import math
import ast
import re
import torch
from transformers import StoppingCriteria
from .constants import IMAGE_TOKEN_INDEX
import random
import os
import io
import av
import cv2
import imageio
from decord import VideoReader
import numpy as np



######################## load video ########################

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base


def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time
    )
    return float(video_duration)



def get_frame_indices(num_frames, vlen, sample='middle', fix_start=None, input_fps=1, min_num_frames=1, max_num_frames=-1, local_num_frames=8):

    if min_num_frames > vlen:
        if sample == 'dynamic_fps1':
            min_num_frames = (vlen // local_num_frames) * local_num_frames
        else:
            min_num_frames = vlen


    if sample == 'dynamic_fps1':

        duration = float(vlen) / input_fps
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments

        if max_num_frames > 0:
            num_frames = min(num_frames, max_num_frames)
        sample = "middle" # NOTE

        # logger.info(f"? is OK (img), duation={duration} frames={num_frames}!!!!")

    num_frames = max(min_num_frames, num_frames)

    # print(f"\033[0;31m vlen={vlen}, input_fps={input_fps} num_frames={num_frames} \033[0m")
        
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError(f"Not support sample type: {sample}")
    
    
    return frame_indices


def read_frames_av(video_path, num_frames, sample='rand', client=None, fix_start=None, min_num_frames=1, max_num_frames=-1, clip=None, local_num_frames=8):
    if clip is not None:
        raise NotImplementedError("av don't support clip!!!")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        byteio = io.BytesIO(video_bytes)
        byteio.seek(0)
        reader = av.open(byteio)
    else:
        byteio = None
        reader = av.open(video_path)
    frames = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
    vlen = len(frames)
    duration = get_pyav_video_duration(reader)
    fps = vlen / float(duration)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, min_num_frames=min_num_frames, max_num_frames=max_num_frames, local_num_frames=local_num_frames
    )
    frames = np.stack([frames[idx] for idx in frame_indices])  # (T, H, W, C), torch.uint8
    # frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    if byteio != None:
        byteio.close()
        
    reader.close()

    return frames, frame_indices, float(fps), duration


def read_frames_gif(
        video_path, num_frames, sample='rand', fix_start=None, 
        min_num_frames=1, max_num_frames=-1, client=None, clip=None, local_num_frames=8
    ):
    if clip is not None:
        raise NotImplementedError("Gif don't support clip!!!")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        byteio = io.BytesIO(video_bytes)
        gif = imageio.get_reader(byteio)
    else:
        byteio = None
        gif = imageio.get_reader(video_path)
    vlen = len(gif)
    fps = 1.
    duration = vlen / fps
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        min_num_frames=min_num_frames,
        max_num_frames=max_num_frames, local_num_frames=local_num_frames,
        input_fps=fps 
    )
    frames = []

    min_h = min_w = 100000
    hw_set = set()
    for index, frame in enumerate(gif):
        # for index in frame_idxs:
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = frame.astype(np.uint8)
            # # (H x W x C) to (C x H x W)
            # frame = frame.permute(2, 0, 1)
            frames.append(frame)
            hw_set.add(frame.shape)
            if frame.shape[0] < min_h:
                min_h = frame.shape[0]
            if frame.shape[1] < min_w:
                min_w = frame.shape[1]
    # print(hw_set, min_h, min_w)
    if len(hw_set) > 1:
        frames = [i[:min_h, :min_w] for i in frames]

    frames = np.stack(frames)  # .float() / 255

    if byteio != None:
        byteio.close()

    return frames, frame_indices, float(fps), duration # for tgif



def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, min_num_frames=1,
        max_num_frames=-1, client=None, clip=None, local_num_frames=8
    ):

    if video_path.endswith('.avi'):
        return read_frames_av(video_path=video_path, num_frames=num_frames, sample=sample,
                    fix_start=fix_start, min_num_frames=min_num_frames, max_num_frames=max_num_frames, 
                    client=client, clip=clip, local_num_frames=local_num_frames)
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        if video_bytes is None or len(video_bytes) == 0:
            raise ValueError(f"Can't read byte from {video_path}!")
        byteio = io.BytesIO(video_bytes)
        video_reader = VideoReader(byteio, num_threads=1)
    else:
        byteio = None
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    

    if clip:
        start, end = clip
        start = max(0, start)
        end = min(duration - 0.1, end)
        duration = end - start
        vlen = int(duration * fps) 
        start_index = int(start * fps)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, min_num_frames=min_num_frames, max_num_frames=max_num_frames, local_num_frames=local_num_frames
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]

    # print(fps, frame_indices)
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), torch.uint8
    # https://github.com/dmlc/decord/issues/208
    video_reader.seek(0)

    if byteio != None:
        byteio.close()
    # frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, float(fps), duration



def read_frames_img(
        video_path, num_frames, sample='rand', fix_start=None, min_num_frames=1,
        max_num_frames=-1, client=None, clip=None, local_num_frames=8
    ):
    def extract_frame_number(filename):
        # Extract the numeric part from the filename using regular expressions
        if filename.endswith('.jpg'):
            match = re.search(r'_(\d+).jpg$', filename)
        elif filename.endswith('.jpeg'):
            match = re.search(r'_(\d+).jpeg$', filename)
        elif filename.endswith('.png'):
            match = re.search(r'_(\d+).png$', filename)
        else:
            raise NotImplementedError(f"Wrong filename: {filename}")

        return int(match.group(1)) if match else -1


    def sort_frames(frame_paths):
        # Extract filenames from each path and sort by their numeric part
        return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))

    # img_list=[]

    if "s3://" in video_path:
        img_list = sort_frames(client.list(video_path))
    else:
        img_list = sort_frames(list(os.listdir(video_path)))


    if 'tvqa' in video_path.lower():
        fps = 3.0
    else:
        fps = 1.0 

    if clip is not None:
        start = float(clip[0])
        end = float(clip[1])
        start = max(0, start)
        end = min(len(img_list) / fps, end)
        vlen = (end - start) * fps
    else:
        vlen = len(img_list)
    
    duration = vlen / fps

    if min_num_frames > vlen:
        if sample == 'dynamic_fps1':
            min_num_frames = (vlen // local_num_frames) * local_num_frames
        else:
            min_num_frames = vlen

    if sample == 'dynamic_fps1':
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        num_frames = min(num_frames, max_num_frames) 
        num_frames = max(min_num_frames, num_frames)

    num_frames = int(num_frames)
    if clip is not None:
        def _get_index_by_time(start_sec, end_sec, num_segments=8, fps=1., max_frame=9999):
            start_idx = max(1, round(start_sec * fps))
            end_idx = min(round(end_sec * fps), max_frame)
            seg_size = float(end_idx - start_idx) / (num_segments - 1)
            offsets = np.array([start_idx + int(np.round(seg_size * idx)) for idx in range(num_segments)])
            return offsets

        frame_indices = _get_index_by_time(float(clip[0]), float(clip[1]), num_segments=num_frames, fps=fps, max_frame=len(img_list)-1)
    else:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            min_num_frames=min_num_frames,
            max_num_frames=max_num_frames, local_num_frames=local_num_frames
        )

    imgs = []
    for idx in frame_indices:
        frame_fname = os.path.join(video_path, img_list[idx])
        if "s3://" in video_path:
            img_bytes = client.get(frame_fname)
        else:
            with open(frame_fname, 'rb') as f:
                img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        imgs.append(img)

    frames = np.array(imgs, dtype=np.uint8)


    return frames, frame_indices, fps, duration 



VIDEO_READER_FUNCS = {
    'av': read_frames_av,
    'decord': read_frames_decord,
    'gif': read_frames_gif,
    'img': read_frames_img,
    'frame': read_frames_img
}



def load_video(video_path, max_num_frames=512, media_dict=None): #, media_dict):

    if media_dict is None:
        media_dict = {'video_read_type': 'decord'}

    if type(video_path) != str:
        assert len(video_path) == 1, video_path
        video_path = video_path[0]

    if 'start' in media_dict:
        clip = [media_dict['start'], media_dict['end']]
    else:
        clip = None
    
    client = None

    frames, frame_indices, fps, duration = VIDEO_READER_FUNCS[media_dict['video_read_type']](video_path=video_path, num_frames=max_num_frames, sample='dynamic_fps1', fix_start=None, min_num_frames=64, max_num_frames=max_num_frames, client=client, clip=clip, local_num_frames=8)

    sec = [str(round(f / fps, 1)) for f in frame_indices]

    msg = f"\nThe video lasts for {duration:.2f} seconds, and {len(sec)} frames are uniformly sampled from it. "

    return frames, msg


######################## load video ########################


def resize_and_center_crop(image, shortest_edge_length):
    # Calculate new dimensions and resize
    aspect_ratio = float(image.width) / float(image.height)
    if aspect_ratio > 1:
        new_width = int(shortest_edge_length * aspect_ratio)
        new_height = shortest_edge_length
    else:
        new_width = shortest_edge_length
        new_height = int(shortest_edge_length / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate the position and perform the center crop
    left = (new_width - shortest_edge_length) / 2
    top = (new_height - shortest_edge_length) / 2
    right = (new_width + shortest_edge_length) / 2
    bottom = (new_height + shortest_edge_length) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image


def auto_pad_images(image, grid_params):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert len(grid_params) > 0, "Grid parameters should not be empty"

    # Step 1: Calculate and find the closest aspect ratio
    input_width, input_height = image.size
    input_aspect_ratio = input_width / input_height
    candidate_resolutions = [(w / h, w, h) for w in grid_params for h in grid_params]
    closest_aspect_ratio = min(candidate_resolutions, key=lambda x: abs(input_aspect_ratio - x[0]))

    candidate_resolutions = [(x[1], x[2]) for x in candidate_resolutions if abs(x[0] - closest_aspect_ratio[0]) < 1e-3]

    target_resolution = min(candidate_resolutions, key=lambda res: abs(max(input_width, input_height) / max(res) - 1))

    resize_width, resize_height = target_resolution
    if input_width > input_height:
        resize_height = int(resize_width / input_aspect_ratio)
    else:
        resize_width = int(resize_height * input_aspect_ratio)
    resized_image = image.resize((resize_width, resize_height), Image.ANTIALIAS)

    # Step 5: Pad the resized image if necessary to match the target resolution
    pad_width = target_resolution[0] - resize_width
    pad_height = target_resolution[1] - resize_height
    padded_image = Image.new("RGB", target_resolution, color=(0, 0, 0))
    padded_image.paste(resized_image, (pad_width // 2, pad_height // 2))

    return padded_image


def extract_patches(image, patch_size, overlap_ratio):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert patch_size > 0, "Patch size should be greater than 0"
    assert 0 <= overlap_ratio < 1, "Overlap ratio should be between 0 and 1"

    W, H = image.size
    patches = []

    stride = int(patch_size * (1 - overlap_ratio))

    num_patches_y = (H - patch_size) // stride + 1
    num_patches_x = (W - patch_size) // stride + 1

    y_start = (H - (num_patches_y - 1) * stride - patch_size) // 2
    x_start = (W - (num_patches_x - 1) * stride - patch_size) // 2

    for y in range(y_start, y_start + num_patches_y * stride, stride):
        for x in range(x_start, x_start + num_patches_x * stride, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches


def process_highres_image_crop_split(image, data_args, processor=None):
    crop_resolution = data_args.image_crop_resolution
    split_resolution = data_args.image_split_resolution
    if processor is None:
        processor = data_args.image_processor
    image_crop = resize_and_center_crop(image, crop_resolution)
    image_patches = extract_patches(image_crop, patch_size=split_resolution, overlap_ratio=0)
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def process_highres_image(image, processor, grid_pinpoints):
    grid_params = [int(x) for x in grid_pinpoints.split(",")]
    width_height = max(image.size)
    fit_grid_params = [x for x in grid_params if x >= width_height]
    if len(fit_grid_params) == 0:
        select_size = max(grid_params)
    else:
        select_size = min(fit_grid_params)
    # FIXME: always select the 448
    select_size = max(grid_params)
    image_padded = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))

    # FIXME: this seems to be a bug that it always resizes instead of padding
    image_original_resize = image.resize((processor.size["shortest_edge"], processor.size["shortest_edge"]))
    image_padded = image_padded.resize((select_size, select_size))
    image_patches = extract_patches(image_padded, patch_size=processor.size["shortest_edge"], overlap_ratio=0)
    image_patches = [image_original_resize] + image_patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def select_best_resolution(original_size, possible_resolutions, max_resolutions, patch_size):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        if max_resolutions != None and (width * height != patch_size * patch_size):
            if (width * height+patch_size*patch_size) > max_resolutions: # NOTE 要算一个global
                continue
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    # print(f"original_size={original_size}, possible_resolutions={possible_resolutions}, max_resolutions={max_resolutions}, best_fit={best_fit}")
    assert best_fit is not None, f"Can't find suitable fit in {possible_resolutions} at max:{max_resolutions}"
    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size, max_resolutions=None):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions, max_resolutions=max_resolutions, patch_size=patch_size)

    # print("get width/patch size", width, patch_size, flush=True)

    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    raise NotImplementedError
    # Convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size["height"])

    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]

    # print("image.size", image.size, "len(image_patches):",  len(image_patches), "patch_size:", image_patches[0].shape)
    return torch.stack(image_patches, dim=0)

def process_anyres_image_nopad(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    try:
        patch_size = processor.size[0]
    except Exception as e:
        patch_size = processor.size["shortest_edge"]

    assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"

    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions, max_resolutions=None, patch_size=patch_size) # 目前图像无限制
    # image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image.resize(best_resolution), patch_size)

    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]

    # raise ValueError(f"image.size: {image.size} len(image_patches): {len(image_patches)}, patch_size:, {image_patches[0].shape}, possible_resolutions:, {possible_resolutions}, best: {best_resolution}")
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "highres":
        raise NotImplementedError
        for image in images:
            image = process_highres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif "anyres" in image_aspect_ratio:
        for image in images:
            if "nopad" in image_aspect_ratio:
                image = process_anyres_image_nopad(image, image_processor, model_cfg.image_grid_pinpoints)
            else:
                image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "crop_split":
        raise NotImplementedError
        for image in images:
            image = process_highres_image_crop_split(image, model_cfg, image_processor)
            new_images.append(image)
    elif image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
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


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
