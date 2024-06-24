# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
import hydra
from visdom import Visdom

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from lightglue import LightGlue, SuperPoint, SIFT, ALIKED

import pycolmap

from minipytorch3d.cameras import PerspectiveCameras

from vggsfm.datasets.sequence_loader import SequenceLoader
from vggsfm.two_view_geo.estimate_preliminary import estimate_preliminary_cameras

try:
    import poselib
    from vggsfm.two_view_geo.estimate_preliminary import estimate_preliminary_cameras_poselib
    print("Poselib is available")
except:
    print("Poselib is not installed. Please disable use_poselib")
    
from vggsfm.utils.utils import (
    set_seed_and_print,
    farthest_point_sampling,
    calculate_index_mappings,
    switch_tensor_order,
)
from vggsfm.utils.metric import camera_to_rel_deg, calculate_auc, calculate_auc_np


@hydra.main(config_path="cfgs/", config_name="demo")
def test_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # Print configuration
    print("Model Config:", OmegaConf.to_yaml(cfg))

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Set seed
    seed_all_random_engines(cfg.seed)

    # Model instantiation
    model = instantiate(cfg.MODEL, _recursive_=False, cfg=cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # Prepare test dataset
    test_dataset = SequenceLoader(
        SEQ_DIR=cfg.SEQ_DIR, img_size=1024, normalize_cameras=False, load_gt=cfg.load_gt, cfg=cfg
    )

    if cfg.resume_ckpt:
        # Reload model
        checkpoint = torch.load(cfg.resume_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Successfully resumed from {cfg.resume_ckpt}")

    error_dict = {"rError": [], "tError": []}

    if cfg.visualize:
        from pytorch3d.structures import Pointclouds
        from pytorch3d.vis.plotly_vis import plot_scene
        from pytorch3d.implicitron.tools import model_io, vis_utils

        viz = vis_utils.get_visdom_connection(
            server=f"http://10.200.188.27", port=int(os.environ.get("VISDOM_PORT", 10088))
        )
        # viz = Visdom()

    sequence_list = test_dataset.sequence_list

    for seq_name in sequence_list:
        print("*" * 50 + f" Testing on Scene {seq_name} " + "*" * 50)

        # Load the data
        batch, image_paths = test_dataset.get_data(sequence_name=seq_name, return_path=True)

        # Send to GPU
        images = batch["image"].to(device)
        crop_params = batch["crop_params"].to(device)

        if cfg.load_gt:
            translation = batch["T"].to(device)
            rotation = batch["R"].to(device)
            fl = batch["fl"].to(device)
            pp = batch["pp"].to(device)

            # Prepare gt cameras
            gt_cameras = PerspectiveCameras(
                focal_length=fl.reshape(-1, 2),
                principal_point=pp.reshape(-1, 2),
                R=rotation.reshape(-1, 3, 3),
                T=translation.reshape(-1, 3),
                device=device,
            )

        # Unsqueeze to have batch size = 1
        images = images.unsqueeze(0)
        crop_params = crop_params.unsqueeze(0)

        batch_size = len(images)
        
        with torch.no_grad():
            # Run the model
            assert cfg.mixed_precision in ("None", "bf16", "fp16")
            if cfg.mixed_precision == "None":
                dtype = torch.float32
            elif cfg.mixed_precision == "bf16":
                dtype = torch.bfloat16
            elif cfg.mixed_precision == "fp16":
                dtype = torch.float16
            else:
                raise NotImplementedError(f"dtype {cfg.mixed_precision} is not supported now")
            
            predictions = run_one_scene(
                model,
                images,
                crop_params=crop_params,
                query_frame_num=cfg.query_frame_num,
                image_paths=image_paths,
                dtype = dtype,
                cfg=cfg,
            )

        # Export prediction as colmap format
        reconstruction_pycolmap = predictions["reconstruction"]
        output_path = os.path.join(seq_name, "output")
        os.makedirs(output_path, exist_ok=True)
        reconstruction_pycolmap.write(output_path)

        with open(os.path.join(output_path, "file_order.txt"), "w") as file:
            for s in image_paths:
                file.write(s + "\n")  # Write each string with a newline

        pred_cameras = predictions["pred_cameras"]

        if cfg.visualize:
            pcl = Pointclouds(points=predictions["points3D"][None])
            visual_dict = {"scenes": {"points": pcl, "cameras": pred_cameras}}
            # visual_dict = {"scenes": {"points": pcl}}

            fig = plot_scene(visual_dict, camera_scale=0.05)
            
            env_name = "demo_AAA"
            print(f"saving to {env_name}")
            viz.plotlyplot(fig, env=env_name, win="3D")

        # For more details about error computation,
        # You can refer to IMC benchmark
        # https://github.com/ubc-vision/image-matching-benchmark/blob/master/utils/pack_helper.py

        if cfg.load_gt:
            # Compute the error
            rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_cameras, gt_cameras, device, batch_size)

            print(f"    --  Mean Rot   Error (Deg) for this scene: {rel_rangle_deg.mean():10.2f}")
            print(f"    --  Mean Trans Error (Deg) for this scene: {rel_tangle_deg.mean():10.2f}")

            error_dict["rError"].extend(rel_rangle_deg.cpu().numpy())
            error_dict["tError"].extend(rel_tangle_deg.cpu().numpy())

    if cfg.load_gt:
        rError = np.array(error_dict["rError"])
        tError = np.array(error_dict["tError"])

        # you can choose either calculate_auc/calculate_auc_np, they lead to the same result
        Auc_30, normalized_histogram = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_3 = np.mean(np.cumsum(normalized_histogram[:3]))
        Auc_5 = np.mean(np.cumsum(normalized_histogram[:5]))
        Auc_10 = np.mean(np.cumsum(normalized_histogram[:10]))

        print(f"Testing Done")

        for _ in range(5):
            print("-" * 100)

        print("On the IMC dataset")
        print(f"Auc_3  (%): {Auc_3 * 100}")
        print(f"Auc_5  (%): {Auc_5 * 100}")
        print(f"Auc_10 (%): {Auc_10 * 100}")
        print(f"Auc_30 (%): {Auc_30 * 100}")

        for _ in range(5):
            print("-" * 100)

    return True


def run_one_scene(model, images, crop_params=None, query_frame_num=3, return_in_pt3d=True, image_paths=None, dtype=None, cfg=None):
    """
    images have been normalized to the range [0, 1] instead of [0, 255]
    """
    batch_num, frame_num, image_dim, height, width = images.shape
    device = images.device
    reshaped_image = images.reshape(batch_num * frame_num, image_dim, height, width)

    predictions = {}
    extra_dict = {}

    camera_predictor = model.camera_predictor
    track_predictor = model.track_predictor
    triangulator = model.triangulator

    # Find the query frames
    # First use DINO to find the most common frame among all the input frames
    # i.e., the one has highest (average) cosine similarity to all others
    # Then use farthest_point_sampling to find the next ones
    # The number of query frames is determined by query_frame_num
    
    with autocast(dtype=dtype):
        query_frame_indexes = find_query_frame_indexes(reshaped_image, camera_predictor, frame_num)

    image_paths = [os.path.basename(imgpath) for imgpath in image_paths]
    
    
    
    
    if cfg.center_order:
        # The code below switchs the first frame (frame 0) to the most common frame
        center_frame_index = query_frame_indexes[0]
        center_order = calculate_index_mappings(center_frame_index, frame_num, device=device)

        images, crop_params = switch_tensor_order([images, crop_params], center_order, dim=1)
        reshaped_image = switch_tensor_order([reshaped_image], center_order, dim=0)[0]

        image_paths = [image_paths[i] for i in center_order.cpu().numpy().tolist()]
        
        # Also update query_frame_indexes:
        query_frame_indexes = [center_frame_index if x == 0 else x for x in query_frame_indexes]
        query_frame_indexes[0] = 0


    # only pick query_frame_num 
    query_frame_indexes = query_frame_indexes[:query_frame_num]

    # Prepare image feature maps for tracker
    fmaps_for_tracker = track_predictor.process_images_to_fmaps(images)

    # Predict tracks
    with autocast(dtype=dtype):
        pred_track, pred_vis, pred_score = predict_tracks(track_predictor, images, fmaps_for_tracker, 
                                                      query_frame_indexes, frame_num, device, cfg)



        if cfg.comple_nonvis:
            print("using nonvis frames as queries")
            non_vis_frames = torch.nonzero((pred_vis.squeeze(0)>0.05).sum(-1) < 100).squeeze(-1).tolist()

            if len(non_vis_frames)>0:
                # if a frame has too few visible inlier, use it as a query  
                pred_track_comple, pred_vis_comple, pred_score_comple = predict_tracks(track_predictor, images, 
                                                                fmaps_for_tracker, non_vis_frames, 
                                                                frame_num, device, cfg)
                pred_track = torch.cat([pred_track, pred_track_comple], dim=2)
                pred_vis = torch.cat([pred_vis, pred_vis_comple], dim=2)
                pred_score = torch.cat([pred_score, pred_score_comple], dim=2)
                
                
        ########################################################################
        if False:
            from vggsfm.utils.visual import visualize_track
            from pytorch3d.implicitron.tools import model_io, vis_utils

            viz = vis_utils.get_visdom_connection(
                server=f"http://10.200.188.27", port=int(os.environ.get("VISDOM_PORT", 10088))
            )
            
            predictions= {}
            predictions["pred_tracks"] = pred_track_comple
            predictions["pred_vis"] = pred_vis_comple
            visualize_track(predictions, images, None, None, cfg, 0, viz, n_points=4096, selected_indices=None, total_points=4096, save_dir=None, visual_gt = False, )

    torch.cuda.empty_cache()

    # If necessary, force all the predictions at the padding areas as non-visible
    if crop_params is not None:
        boundaries = crop_params[:, :, -4:-2].abs().to(device)
        boundaries = torch.cat([boundaries, reshaped_image.shape[-1] - boundaries], dim=-1)
        hvis = torch.logical_and(
            pred_track[..., 1] >= boundaries[:, :, 1:2], pred_track[..., 1] <= boundaries[:, :, 3:4]
        )
        wvis = torch.logical_and(
            pred_track[..., 0] >= boundaries[:, :, 0:1], pred_track[..., 0] <= boundaries[:, :, 2:3]
        )
        force_vis = torch.logical_and(hvis, wvis)
        pred_vis = pred_vis * force_vis.float()

    # TODO: plot 2D matches
    
    if cfg.use_poselib:
        estimate_preliminary_cameras_fn = estimate_preliminary_cameras_poselib
    else:
        estimate_preliminary_cameras_fn = estimate_preliminary_cameras
    
    
    # Estimate preliminary_cameras by recovering fundamental/essential/homography matrix from 2D matches
    # By default, we use fundamental matrix estimation with 7p/8p+LORANSAC
    # All the operations are batched and differentiable (if necessary)
    # except when you enable use_poselib to save GPU memory
    _, preliminary_dict = estimate_preliminary_cameras_fn(
        pred_track,
        pred_vis,
        width,
        height,
        tracks_score=pred_score,
        max_error = cfg.fmat_thres,
        loopresidual=True,
        max_ransac_iters=cfg.max_ransac_iters,
    )
    
    pose_predictions = camera_predictor(reshaped_image, batch_size=batch_num)

    pred_cameras = pose_predictions["pred_cameras"]

    # Conduct Triangulation and Bundle Adjustment
    BA_cameras, extrinsics_opencv, intrinsics_opencv, points3D, reconstruction = triangulator(
        pred_cameras,
        pred_track,
        pred_vis,
        images,
        preliminary_dict,
        image_paths=image_paths,
        pred_score=pred_score,
        return_in_pt3d=return_in_pt3d,
        fmat_thres=cfg.fmat_thres,
        init_max_reproj_error=cfg.init_max_reproj_error,
        cfg= cfg,
    )

    # Switch back
    # NOTE we changed the image order previously, now we need to switch it back    
    # import pdb;pdb.set_trace()
    print("you have to switch it back!")
    predictions["pred_cameras"] = BA_cameras
    predictions["extrinsics_opencv"] = extrinsics_opencv
    predictions["intrinsics_opencv"] = intrinsics_opencv
    predictions["points3D"] = points3D
    predictions["reconstruction"] = reconstruction

    return predictions



def predict_tracks(track_predictor, images, fmaps_for_tracker, query_frame_indexes, frame_num, device, cfg=None):
    pred_track_list = []
    pred_vis_list = []
    pred_score_list = []

    for query_index in query_frame_indexes:
        # Find query_points at the query frame
        query_points = get_query_points(images[:, query_index], cfg.query_method, cfg.max_query_pts)

        # Switch so that query_index frame stays at the first frame
        # This largely simplifies the code structure of tracker
        new_order = calculate_index_mappings(query_index, frame_num, device=device)
        images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], new_order)


        # Feed into track predictor
        fine_pred_track, _, pred_vis, pred_score = track_predictor(images_feed, query_points, fmaps=fmaps_feed)

        # Switch back the predictions
        fine_pred_track, pred_vis, pred_score = switch_tensor_order([fine_pred_track, pred_vis, pred_score], new_order)

        # Append predictions for different queries
        pred_track_list.append(fine_pred_track)
        pred_vis_list.append(pred_vis)
        pred_score_list.append(pred_score)


    pred_track = torch.cat(pred_track_list, dim=2)
    pred_vis = torch.cat(pred_vis_list, dim=2)
    pred_score = torch.cat(pred_score_list, dim=2)

    return pred_track, pred_vis, pred_score



def find_query_frame_indexes(reshaped_image, camera_predictor, query_frame_num, image_size=336):
    # Downsample image to image_size x image_size
    # because we found it is unnecessary to use high resolution
    rgbs = F.interpolate(reshaped_image, (image_size, image_size), mode="bilinear", align_corners=True)
    rgbs = camera_predictor._resnet_normalize_image(rgbs)

    # Get the image features (patch level)
    frame_feat = camera_predictor.backbone(rgbs, is_training=True)
    frame_feat = frame_feat["x_norm_patchtokens"]
    frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

    # Compute the similiarty matrix
    frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
    similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
    similarity_matrix = similarity_matrix.mean(dim=0)
    distance_matrix = 1 - similarity_matrix.clone()

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(0)

    similarity_sum = similarity_matrix.sum(dim=1)

    # Find the most common frame
    most_common_frame_index = torch.argmax(similarity_sum).item()

    # Conduct FPS sampling
    # Starting from the most_common_frame_index,
    # try to find the farthest frame,
    # then the farthest to the last found frame
    # (frames are not allowed to be found twice)
    fps_idx = farthest_point_sampling(distance_matrix, query_frame_num, most_common_frame_index)

    return fps_idx



def get_query_points(query_image, query_method, max_query_num=4096, det_thres=0.005):
    # Run superpoint and sift on the target frame
    # Feel free to modify for your own

    methods = query_method.split('+')
    pred_points = []
    
    for method in methods:
        if "sp" in method:
            extractor = SuperPoint(max_num_keypoints=max_query_num, detection_threshold=det_thres).cuda().eval()
        elif "sift" in method:
            extractor = SIFT(max_num_keypoints=max_query_num).cuda().eval()            
        elif "aliked" in method:
            extractor = ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres).cuda().eval()
        else:
            raise NotImplementedError(f"query method {method} is not supprted now")
        
        query_points = extractor.extract(query_image)["keypoints"]
        pred_points.append(query_points)

    query_points = torch.cat(pred_points, dim=1)

    if query_points.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points.shape[1])[:max_query_num]
        query_points = query_points[:, random_point_indices, :]

    return query_points


def seed_all_random_engines(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    with torch.no_grad():
        test_fn()
