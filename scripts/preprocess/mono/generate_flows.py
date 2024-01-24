# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import os
import tqdm
import argparse
import pathlib
import subprocess
import numpy as np
from os.path import join
from functools import lru_cache
from skimage.transform import resize as imresize

# from glob import glob

try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
finally:
    import cv2

import torch
import torch.nn.functional as F

sys.path.insert(0, "./third_party/RAFT")
sys.path.insert(0, "./third_party/RAFT/core")
from third_party.RAFT.core.raft import RAFT


def resize_flow(flow, size):
    resized_width, resized_height = size
    H, W = flow.shape[:2]
    scale = np.array((resized_width / float(W), resized_height / float(H))).reshape(
        1, 1, -1
    )
    resized = cv2.resize(
        flow, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC
    )
    resized *= scale
    return resized


def get_oob_mask(flow_1_2):
    H, W, _ = flow_1_2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([H, W, 2])
    coord[..., 0] = ww
    coord[..., 1] = hh
    target_range = coord + flow_1_2
    m1 = (target_range[..., 0] < 0) + (target_range[..., 0] > W - 1)
    m2 = (target_range[..., 1] < 0) + (target_range[..., 1] > H - 1)
    return (m1 + m2).float().numpy()


def backward_flow_warp(im2, flow_1_2):
    H, W, _ = im2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([1, H, W, 2])
    coord[0, ..., 0] = ww
    coord[0, ..., 1] = hh
    sample_grids = coord + flow_1_2[None, ...]
    sample_grids[..., 0] /= (W - 1) / 2
    sample_grids[..., 1] /= (H - 1) / 2
    sample_grids -= 1
    im = torch.from_numpy(im2).float().permute(2, 0, 1)[None, ...]
    out = F.grid_sample(im, sample_grids, align_corners=True)
    o = out[0, ...].permute(1, 2, 0).numpy()
    return o


def get_L2_error_map(v1, v2):
    return np.linalg.norm(v1 - v2, axis=-1)


def load_RAFT():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--path", help="dataset for evaluation")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    args = parser.parse_args(
        ["--model", "./third_party/RAFT/models/raft-sintel.pth", "--path", "./"]
    )
    net = torch.nn.DataParallel(RAFT(args).cuda())
    net.load_state_dict(torch.load(args.model))
    return net


@lru_cache(maxsize=200)
def read_frame_data(data_root, frame_id):
    data = np.load(data_root / f"frame_{frame_id:05d}.npz")
    data_dict = {}
    for k in data.keys():
        data_dict[k] = data[k]
    return data_dict


def generate_pair_data(data_root, save_dir, frame_id_1, frame_id_2, save=True):
    im1_data = read_frame_data(data_root, frame_id_1)
    im2_data = read_frame_data(data_root, frame_id_2)

    im1 = im1_data["img_orig"] * 255
    im2 = im2_data["img_orig"] * 255
    im1 = imresize(im1, [288, 512], anti_aliasing=True)
    im2 = imresize(im2, [288, 512], anti_aliasing=True)

    images = [im1, im2]
    images = np.array(images).transpose(0, 3, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).cuda()
    with torch.no_grad():
        flow_low, flow_up = net(
            image1=im[0:1, ...], image2=im[1:2, ...], iters=20, test_mode=True
        )
        flow_1_2 = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()

    H, W, _ = im1_data["img"].shape
    flow_1_2 = resize_flow(flow_1_2, [W, H])

    with torch.no_grad():
        flow_low, flow_up = net(
            image1=im[1:2, ...], image2=im[0:1, ...], iters=20, test_mode=True
        )
        flow_2_1 = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()

    flow_2_1 = resize_flow(flow_2_1, [W, H])

    warp_flow_1_2 = backward_flow_warp(
        flow_1_2, flow_2_1
    )  # using latter to sample former
    err_1 = np.linalg.norm(warp_flow_1_2 + flow_2_1, axis=-1)
    mask_1 = np.where(err_1 > 1, 1, 0)
    oob_mask_1 = get_oob_mask(flow_2_1)
    mask_1 = np.clip(mask_1 + oob_mask_1, a_min=0, a_max=1)
    warp_flow_2_1 = backward_flow_warp(flow_2_1, flow_1_2)
    err_2 = np.linalg.norm(warp_flow_2_1 + flow_1_2, axis=-1)
    mask_2 = np.where(err_2 > 1, 1, 0)
    oob_mask_2 = get_oob_mask(flow_1_2)
    mask_2 = np.clip(mask_2 + oob_mask_2, a_min=0, a_max=1)
    save_dict = {}
    save_dict["flow_1_2"] = flow_1_2.astype(np.float32)
    save_dict["flow_2_1"] = flow_2_1.astype(np.float32)
    save_dict["mask_1"] = mask_1.astype(np.uint8)
    save_dict["mask_2"] = mask_2.astype(np.uint8)
    save_dict["frame_id_1"] = frame_id_1
    save_dict["frame_id_2"] = frame_id_2
    if save:
        np.savez(
            save_dir / f"flowpair_{frame_id_1:05d}_{frame_id_2:05d}.npz",
            **save_dict,
        )
        return 1
    else:
        return save_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--save_dir", default=".")
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_root)
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    net = load_RAFT()

    scene_dir = data_root / "frames_midas"

    scene_save_dir = save_dir / "flow_pairs"
    scene_save_dir.mkdir(parents=True, exist_ok=True)

    l = len(list(scene_dir.glob("frame_*.npz")))

    gaps = [1, 2, 3, 4, 5, 6, 7, 8]

    for g in tqdm.tqdm(gaps, desc="#gaps"):
        print("\ngap: ", g)

        for k in tqdm.tqdm(range(l - g), desc="#frames"):
            generate_pair_data(scene_dir, scene_save_dir, k, k + g)
