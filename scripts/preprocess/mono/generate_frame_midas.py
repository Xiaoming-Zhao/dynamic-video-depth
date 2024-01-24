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
import pathlib
import argparse
import trimesh
import tqdm
import PIL.Image
import numpy as np
from skimage.transform import resize as imresize
from scipy.ndimage import map_coordinates

import torch

# sys.path.insert(0, "")
# from configs import midas_pretrain_path
# from third_party.MiDaS import MidasNet
import scripts.preprocess.mono.colmap_reader as colmap_reader
from third_party.full_midas.midas.model_loader import default_models, load_model


TINY_VAL = 1e-8

FLAG_MIDAS_OUT_DISP = True


def read_poses(cam_f, n_img_fs):
    poses_arr = np.load(cam_f, allow_pickle=True)  # [#frames, 32]

    # NOTE: for DynIBaR, poses are repeated. I.e., the first 12 cameras are repeated for the whole video.
    # Namely, for any i \in [0, 12], i and i + 12 x n has the same camera matrices for any n.

    assert poses_arr.shape[0] == n_img_fs, f"{poses_arr.shape}, {n_img_fs}"

    all_c2w = poses_arr[:, :16].reshape([-1, 4, 4])  # [#frame, 4, 4]
    all_K = poses_arr[:, 16:].reshape([-1, 4, 4])[
        :, :3, :3
    ]  # [#frame, 4, 4] -> [#frame, 3, 3]

    return all_c2w, all_K


@torch.no_grad()
def run_midas(
    device, model, transform, original_image_rgb, target_size, first_execution=True
):
    image = transform({"image": original_image_rgb})["image"]

    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    if first_execution:
        height, width = sample.shape[2:]
        print(
            f"\n[MiDaS] Input resized to {width}x{height} before entering the encoder.\n"
        )

    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size,  # [H, W]
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )  # inverse depth

    # print("\nprediction: ", np.min(np.abs(prediction)), np.max(np.abs(prediction)), "\n")

    # Do we want to have them in range [0, 1]?
    # https://github.com/isl-org/MiDaS/issues/16#issuecomment-593889568

    if FLAG_MIDAS_OUT_DISP:
        return prediction
    else:
        # NOTE: this may be problematic since prediciton may have really small negative value.
        # After adding TINY_VAL, the pred_depth will be quite large.
        pred_depth = 1 / (prediction + TINY_VAL)
        return pred_depth


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--save_dir", default=".")
    parser.add_argument("--midas_ckpt_dir", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    midas_model_type = "dpt_beit_large_512"
    midas_square = False
    midas_height = None
    midas_optimize = False
    midas_ckpt_path = str(pathlib.Path(args.midas_ckpt_dir) / f"{midas_model_type}.pt")
    model, transform, net_w, net_h = load_model(
        device,
        midas_ckpt_path,
        midas_model_type,
        midas_optimize,
        midas_height,
        midas_square,
    )

    midas_tgt_shape = (net_w, net_h)

    data_root = pathlib.Path(args.data_root)
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    scene_save_dir = save_dir / "frames_midas"
    scene_save_dir.mkdir(parents=True, exist_ok=True)

    img_exts = PIL.Image.registered_extensions()
    supported_img_exts = {ex for ex, f in img_exts.items() if f in PIL.Image.OPEN}
    all_img_fs = []
    for tmp_ext in supported_img_exts:
        all_img_fs = all_img_fs + list(data_root.glob(f"rgbs/*{tmp_ext}"))
    all_img_fs = sorted(all_img_fs)

    n_img_fs = len(all_img_fs)

    print("\nUse our own masks\n")
    all_mask_fs = sorted(
        list((data_root / "masks/final").glob("*_final.png")),
        key=lambda x: x.stem.split("_final")[0],
    )
    assert len(all_mask_fs) == n_img_fs, f"{len(all_mask_fs)}, {n_img_fs}"
    for tmp_i, tmp in enumerate(all_mask_fs):
        assert (
            tmp.stem.split("_final")[0] == all_img_fs[tmp_i].stem
        ), f"{tmp}, {all_img_fs[tmp_i]}"

    print([_.stem for _ in all_mask_fs])

    cam_f = data_root / "poses.npy"

    if False:
        all_c2w, all_hwf = read_poses(
            cam_f, n_img_fs
        )  # c2w: [#frame, 4, 4]; hwf: [#frame, 3]
    else:
        all_c2w, all_K = read_poses(
            cam_f, n_img_fs
        )  # c2w: [#frame, 4, 4]; all_K: [#frame, 3, 3]

    print("\nall_c2w: ", all_c2w.shape, all_K.shape, "\n")

    pts3d_f = data_root / "undistorted/sparse/points3D.bin"
    pts3d_dict = colmap_reader.read_points3d_binary(pts3d_f)

    pts3d = [pts3d_dict[_].xyz for _ in pts3d_dict]
    pts3d = np.array(pts3d).astype(np.float32)  # [#pt, 3]

    print("pts3d: ", pts3d.shape, pts3d.dtype)

    h_pt = np.ones([pts3d.shape[0], 4])
    h_pt[:, :3] = pts3d  # [#pt, 4]
    h_pt = h_pt.T  # [4, #pt]

    print("calculating NN_depth")
    full_pred_depths = []
    pts_list = []

    mvs_depths = []
    pred_depths = []
    masks = []

    all_pred_scale_shifts = []
    all_mvs_scale_shifts = []

    for img_i, img_f in enumerate(tqdm.tqdm(all_img_fs, desc="Run midas")):
        img = np.asarray(PIL.Image.open(img_f)).astype(np.float32) / 255
        img_h, img_w, _ = img.shape

        # img_batch = (
        #     torch.from_numpy(img).permute(2, 0, 1)[None, ...].float().cuda()
        # )  # [1, 3, H, W]

        with torch.no_grad():
            ori_size = (img_h, img_w)
            pred_d = run_midas(
                device,
                model,
                transform,
                img,
                ori_size,
                first_execution=(img_i == 0),
            )
            full_pred_depths.append(pred_d)

            # print("\npred_d: ", pred_d.shape, np.min(pred_d), np.max(pred_d), "\n")

        out = np.linalg.inv(all_c2w[img_i, :]) @ h_pt

        cur_K = all_K[img_i, ...]
        im_pt = cur_K @ out[:3, :]

        depth = im_pt[2, :].copy()
        im_pt = im_pt / im_pt[2:, :]  # [3, #pt]

        # True value measn dynamic
        mask = np.asarray(PIL.Image.open(all_mask_fs[img_i])).astype(np.float32)

        masks.append(mask)

        select_idx = np.where(
            (im_pt[0, :] >= 0)
            * (im_pt[0, :] < img_w)
            * (im_pt[1, :] >= 0)
            * (im_pt[1, :] < img_h)
        )[0]

        pts = im_pt[:, select_idx]  # [3, #pt]
        depth = depth[select_idx]

        out = map_coordinates(mask, [pts[1, :], pts[0, :]])
        select_idx = np.where(out < 0.1)[0]  # static areas
        pts = pts[:, select_idx]
        depth = depth[select_idx]
        select_idx = np.where(depth > 1e-3)[0]
        pts = pts[:, select_idx]
        depth = depth[select_idx]

        pred_depth = map_coordinates(pred_d, [pts[1, :], pts[0, :]])
        mvs_depths.append(depth)
        pred_depths.append(pred_depth)
        pts_list.append(pts)

    print("\nimg: ", img.shape, "\n")

    print("calculating scale")

    # Sec. 3 in DynIBaR's supp: align predicted depth wrt SfM.
    all_disp_scales = []
    all_disp_shifts = []
    for x in tqdm.tqdm(range(n_img_fs)):
        nn_depth = pred_depths[x]
        mvs_depth = mvs_depths[x]

        # scales.append(np.median(nn_depth / mvs_depth))

        # nn_disp = np.clip(1 / (nn_depth + TINY_VAL), 0.01, None)
        # mvs_disp = np.clip(1 / (mvs_depth + TINY_VAL), 0.01, None)

        if FLAG_MIDAS_OUT_DISP:
            nn_disp = nn_depth
        else:
            nn_disp = 1 / (nn_depth + TINY_VAL)
        mvs_disp = 1 / (mvs_depth + TINY_VAL)

        # estimate raw value median, then remove the medians
        nn_disp_median = np.median(nn_disp)
        mvs_disp_median = np.median(mvs_disp)

        nn_disp_normalized = nn_disp - nn_disp_median
        mvs_disp_normalized = mvs_disp - mvs_disp_median

        # estimate scales
        disp_scale = np.median(mvs_disp_normalized / (nn_disp_normalized + TINY_VAL))
        all_disp_scales.append(disp_scale)

        # estimate shift
        disp_shift = np.median(mvs_disp - nn_disp * disp_scale)
        all_disp_shifts.append(disp_shift)

        # scale-shift-invariant loss:
        # - https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0
        # - https://github.com/isl-org/MiDaS/issues/2#issuecomment-511753522

        # t_pred = torch.median(pred_depth)
        # s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    final_disp_scale = np.mean(all_disp_scales)
    final_disp_shift = np.mean(all_disp_shifts)

    print("\nfinal_disp_scale/shift: ", final_disp_scale, final_disp_shift, "\n")

    print("saving per frame output")

    for img_i, img_f in enumerate(tqdm.tqdm(all_img_fs, desc="save results")):
        img_orig = np.asarray(PIL.Image.open(img_f)).astype(np.float32) / 255

        H, W, _ = img_orig.shape

        img = img_orig

        T_G_1 = all_c2w[img_i, ...].astype(np.float32)

        # Maybe not a bug. See https://github.com/google/dynamic-video-depth/issues/6
        depth_mvs = full_pred_depths[img_i].astype(np.float32)

        in_1 = all_K[img_i, ...]
        in_1 = in_1.astype(np.float32)

        depth = full_pred_depths[img_i].astype(np.float32)

        resized_mask = masks[img_i]
        resized_mask = np.where(resized_mask > 1e-3, 1, 0)

        np.savez(
            scene_save_dir / f"frame_{img_i:05d}.npz",
            img=img,
            pose_c2w=T_G_1,
            depth_mvs=depth_mvs,
            intrinsics=in_1,
            depth_pred=depth,
            img_orig=img_orig,
            motion_seg=resized_mask,
            disp_scale=final_disp_scale,
            disp_shift=final_disp_shift,
            depth_is_disp=FLAG_MIDAS_OUT_DISP,
        )
