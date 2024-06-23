# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import cv2
import torch
import flow_vis

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random

def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()
    return np.stack(frames)


class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow" or mode == "sequence":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame: int = 0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        skip_track=False,
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(video, (self.pad_value, self.pad_value, self.pad_value, self.pad_value), "constant", 255)
        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        # res_video = self.cotracker_draw_tracks_on_video(
        #     video=video,
        #     tracks=tracks,
        #     visibility=visibility,
        #     segm_mask=segm_mask,
        #     gt_tracks=gt_tracks,
        #     query_frame=query_frame,
        #     compensate_for_camera_motion=compensate_for_camera_motion,
        # )

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            skip_track=skip_track,
        )

        if save_video:
            self.save_video(res_video, filename=filename, writer=writer, step=step)
        return res_video

    def visualize_withsig(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame: int = 0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        sigsq=None,
        skip_track=False,
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(video, (self.pad_value, self.pad_value, self.pad_value, self.pad_value), "constant", 255)
        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        res_video = self.draw_tracks_on_video_sq(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            sigsq=sigsq,
            skip_track=skip_track,
        )

        if save_video:
            self.save_video(res_video, filename=filename, writer=writer, step=step)
        return res_video

    def save_video(self, video, filename, writer=None, step=0):
        if writer is not None:
            writer.add_video(f"{filename}_pred_track", video.to(torch.uint8), global_step=step, fps=self.fps)
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
            clip = ImageSequenceClip(wide_list[2:-1], fps=self.fps)

            # Write the video file
            save_path = os.path.join(self.save_dir, f"{filename}_pred_track.mp4")
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame: int = 0,
        compensate_for_camera_motion=False,
        skip_track=False,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())

        # if skip_track:
        #     return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

        vector_colors = np.zeros((T, N, 3))
        if self.mode == "optical_flow":
            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif segm_mask is None:
            if self.mode == "sequence":
                y_min = 0
                y_max = N - 1
                norm = plt.Normalize(y_min, y_max)

                for n in range(N):
                    color = self.color_map(norm(n))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)

            elif self.mode == "rainbow":
                y_min, y_max = (tracks[query_frame, :, 1].min(), tracks[query_frame, :, 1].max())
                norm = plt.Normalize(y_min, y_max)

                # tracks[query_frame, 0, 1]
                for n in range(N):
                    color = self.color_map(norm(tracks[query_frame, n, 1]))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                y_min, y_max = (tracks[0, segm_mask > 0, 1].min(), tracks[0, segm_mask > 0, 1].max())
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if segm_mask[n] > 0:
                        color = self.color_map(norm(tracks[0, n, 1]))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(1, T):
                first_ind = max(0, t - self.tracks_leave_trace) if self.tracks_leave_trace >= 0 else 0
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (tracks[first_ind : t + 1, segm_mask <= 0] - tracks[t : t + 1, segm_mask <= 0]).mean(1)[
                        :, None
                    ]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(res_video[t], curr_tracks, curr_colors)
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(res_video[t], gt_tracks[first_ind : t + 1])

        #  draw points
        for t in range(T):
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (compensate_for_camera_motion and segm_mask[i] > 0):
                        if skip_track:
                            try:
                                res_video[t][coord[1], coord[0]] = vector_colors[t, i]
                            except:
                                m = 1
                        else:
                            if False:
                                x_center, y_center = coord  # coord is a tuple like (x_center, y_center)
                                half_size = 129 // 2  # Since the rectangle size is 129x129, half_size is half of that

                                # Calculate top-left corner
                                x1 = x_center - half_size
                                y1 = y_center - half_size

                                # Calculate bottom-right corner
                                x2 = x_center + half_size
                                y2 = y_center + half_size

                                # Now draw the rectangle on the image
                                cv2.rectangle(
                                    res_video[t], (x1, y1), (x2, y2), (255, 0, 0), thickness=2
                                )  # Change color and thickness as needed

                            cv2.circle(
                                res_video[t],
                                coord,
                                int(self.linewidth),
                                vector_colors[t, i].tolist(),
                                thickness=-1 if visibile else 2 - 1,
                            )

                            # if t>0:
                            #     cv2.circle(res_video[t], coord, int(self.linewidth), vector_colors[t, i].tolist(), thickness=-1 if visibile else 2 - 1)
                            # else:
                            #     res_video[t] = draw_cross(res_video[t], coord, vector_colors[t, i].tolist(), self.linewidth//2)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self, rgb: np.ndarray, tracks: np.ndarray, vector_colors: np.ndarray, alpha: float = 0.5
    ):  # H x W x 3  # T x 2
        T, N, _ = tracks.shape

        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    cv2.line(rgb, coord_y, coord_x, vector_color[i].tolist(), self.linewidth, cv2.LINE_AA)
            if self.tracks_leave_trace > 0:
                rgb = cv2.addWeighted(rgb, alpha, original, 1 - alpha, 0)
        return rgb

    def _draw_gt_tracks(self, rgb: np.ndarray, gt_tracks: np.ndarray):  # H x W x 3,  # T x 2
        T, N, _ = gt_tracks.shape
        color = np.array((211.0, 0.0, 0.0))

        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
                    cv2.line(rgb, coord_y, coord_x, color, self.linewidth, cv2.LINE_AA)
                    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
                    cv2.line(rgb, coord_y, coord_x, color, self.linewidth, cv2.LINE_AA)
        return rgb

    def draw_tracks_on_video_sq(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame: int = 0,
        compensate_for_camera_motion=False,
        sigsq=None,
        skip_track=False,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        sigsq = sigsq[0].detach().cpu().numpy()
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())

        vector_colors = np.zeros((T, N, 3))
        if self.mode == "optical_flow":
            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif segm_mask is None:
            if self.mode == "sequence":
                y_min = 0
                y_max = N - 1
                norm = plt.Normalize(y_min, y_max)

                for n in range(N):
                    color = self.color_map(norm(n))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)

            elif self.mode == "rainbow":
                y_min, y_max = (tracks[query_frame, :, 1].min(), tracks[query_frame, :, 1].max())
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    color = self.color_map(norm(tracks[query_frame, n, 1]))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                y_min, y_max = (tracks[0, segm_mask > 0, 1].min(), tracks[0, segm_mask > 0, 1].max())
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if segm_mask[n] > 0:
                        color = self.color_map(norm(tracks[0, n, 1]))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(1, T):
                first_ind = max(0, t - self.tracks_leave_trace) if self.tracks_leave_trace >= 0 else 0
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (tracks[first_ind : t + 1, segm_mask <= 0] - tracks[t : t + 1, segm_mask <= 0]).mean(1)[
                        :, None
                    ]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(res_video[t], curr_tracks, curr_colors)
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(res_video[t], gt_tracks[first_ind : t + 1])

        #  draw points
        for t in range(T):
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                scale = 15
                cov = (sigsq[t, i, 0] * scale, sigsq[t, i, 1] * scale)

                visibile = True
                # if visibility is not None:
                #     visibile = visibility[0, t, i]
                # if coord[0] != 0 and coord[1] != 0:
                #     if not compensate_for_camera_motion or (compensate_for_camera_motion and segm_mask[i] > 0):
                #         # axes_length = (10,30)
                #         # print("go here")
                #         # import pdb;pdb.set_trace()
                #         # cv2.ellipse(res_video[t], center=coord, axes=axes_length, angle=0, startAngle=0, endAngle=360, color=vector_colors[t, i].tolist(), thickness=2)
                #         if t>0:
                #             axes_length = cov
                #             res_video[t] = add_confidence_ellipses_to_image(res_video[t], [coord], [axes_length], [vector_colors[t, i].tolist()])
                #             # res_video[t] = draw_ellipse(res_video[t], coord, axes_length, angle=0, color=vector_colors[t, i].tolist())
                #             cv2.circle(res_video[t], coord, int(self.linewidth), vector_colors[t, i].tolist(), thickness=-1 if visibile else 2 - 1)
                #         else:
                #             linewidth = 5  # Example linewidth
                #             size = 10  # Radius for the circle, half-width/height for the rectangle
                #             top_left = (coord[0] - size, coord[1] - size)
                #             bottom_right = (coord[0] + size, coord[1] + size)

                #             cv2.rectangle(res_video[t], top_left, bottom_right, vector_colors[t, i].tolist(), thickness=linewidth)

                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (compensate_for_camera_motion and segm_mask[i] > 0):
                        if skip_track:
                            try:
                                axes_length = cov
                                res_video[t] = add_confidence_ellipses_to_image(
                                    res_video[t], [coord], [axes_length], [vector_colors[t, i].tolist()]
                                )
                                # res_video[t][coord[1], coord[0]] = vector_colors[t, i]
                            except:
                                m = 1
                        else:
                            import pdb

                            pdb.set_trace()
                            m = 1

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()


def add_confidence_ellipses_to_image(image, points, confidences, colors):
    modified_image = image.copy()
    for (x, y), (cx, cy), color in zip(points, confidences, colors):
        # Convert RGB color to BGR
        bgr_color = color[::-1]

        # Construct the covariance matrix
        cov_matrix = np.diag([cx, cy])

        # Eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(cov_matrix)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height of ellipse (2 standard deviations)
        axes = (int(2 * np.sqrt(vals[0])), int(2 * np.sqrt(vals[1])))

        # Get ellipse points
        ellipse_pts = cv2.ellipse2Poly((int(x), int(y)), axes, int(theta), 0, 360, 1)

        # Draw the ellipse
        cv2.polylines(modified_image, [ellipse_pts], isClosed=True, color=bgr_color, thickness=2)

        # Set the pixel at the ellipse's origin
        modified_image[int(y), int(x)] = bgr_color

    # for (x, y), (cx, cy), color in zip(points, confidences, colors):
    #     # bgr_color = color[::-1]

    #     bgr_color = color
    #     # Construct the covariance matrix
    #     cov_matrix = np.diag([cx, cy])

    #     # Eigenvalues and eigenvectors
    #     vals, vecs = np.linalg.eigh(cov_matrix)
    #     theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    #     # Width and height of ellipse (2 standard deviations)
    #     axes = (int(2 * np.sqrt(vals[0])), int(2 * np.sqrt(vals[1])))

    #     # Get ellipse points
    #     ellipse_pts = cv2.ellipse2Poly((int(x), int(y)), axes, int(theta), 0, 360, 1)

    #     # Draw the ellipse
    #     # cv2.polylines(modified_image, [ellipse_pts], isClosed=True, color=(0, 0, 255), thickness=2)
    #     cv2.polylines(modified_image, [ellipse_pts], isClosed=True, color=bgr_color, thickness=2)

    return modified_image


def draw_cross(rgb, gt_tracks, color, linewidth):
    length = linewidth * 2
    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
    cv2.line(rgb, coord_y, coord_x, color, linewidth, cv2.LINE_AA)
    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
    cv2.line(rgb, coord_y, coord_x, color, linewidth, cv2.LINE_AA)
    return rgb


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    draw.ellipse([left_up_point, right_down_point], fill=tuple(color) if visible else None, outline=tuple(color))
    return rgb





def visualize_track(
    predictions,
    images,
    tracks,
    tracks_visibility,
    cfg,
    step,
    viz,
    n_points=1,
    selected_indices=None,
    total_points=256,
    save_dir=None,
    visual_gt = False,
):
    if "pred_tracks" in predictions:
        track_visualizer = Visualizer(save_dir=save_dir, fps=2, show_first_frame=0, linewidth=8, mode="rainbow")
        image_subset = images[0:1]

        # the selected track number
        if selected_indices is None:
            selected_indices = sorted(random.sample(range(total_points), min(n_points, total_points)))

        # if cfg.debug:
            # selected_indices = list(range(24))

        if visual_gt:
            tracks_subset = tracks[0:1][:, :, selected_indices]
            tracks_vis_subset = tracks_visibility[0:1][:, :, selected_indices]
        else:
            tracks_subset = predictions["pred_tracks"][0:1][:, :, selected_indices]
            tracks_vis_subset = predictions["pred_vis"][0:1][:, :, selected_indices]
            tracks_vis_subset = tracks_vis_subset > 0.05

        res_video_gt = track_visualizer.visualize(
            255 * image_subset, tracks_subset, tracks_vis_subset, save_video=False
        )
        env_name = f"visual_debug"

        viz.images((res_video_gt[0] / 255).clamp(0, 1), env=env_name, win="tmp")

        # viz.images((res_video_gt[0] / 255).clamp(0, 1), env="debug", win="tmp")

        # viz.images(res_video_gt[0,1], env="tmp", win="tmp")

        # res_combined = res_video_gt
        # _, num_frames, channels, height, width = res_combined.shape
        # res_row = res_combined.squeeze(0).permute(1, 2, 0, 3).reshape(3, height, num_frames * width)
        # res_row_np = res_row.numpy()
        # res_row_np = ((res_row_np - res_row_np.min()) / (res_row_np.max() - res_row_np.min()) * 255).astype(np.uint8)
        # res_row_np = np.transpose(res_row_np, (1, 2, 0))
        # import cv2
        # cv2.imwrite('combined_frames.png', res_row_np)
        print(env_name)
        # import pdb

        # pdb.set_trace()

        return res_video_gt, save_dir

        """ TO BE CONTINUED 
        import pdb;pdb.set_trace()        
        m=1


        pred_vis_thresed = predictions["pred_vis"][0:1] > 0.5
        
        # total_points = predictions["pred_tracks"][-1].shape[2]
        pred_tracks_subset = predictions["pred_tracks"][-1][0:1][:, :, selected_indices]
        pred_sigsq_subset = predictions["sig_sq"][0:1][:, :, selected_indices]

        
        

        
        res_video_pred = track_visualizer.visualize(255 * image_subset, pred_tracks_subset, pred_vis_thresed[:, :, selected_indices], save_video=False)
        res_video_raw = track_visualizer.visualize(255 * image_subset, tracks_subset, tracks_vis_subset, save_video=False, skip_track= True)
        res_video_pixpoint = track_visualizer.visualize(255 * image_subset, pred_tracks_subset, pred_vis_thresed[:, :, selected_indices], save_video=False, skip_track= True)
        res_video_pix = track_visualizer.visualize_withsig(255 * image_subset, pred_tracks_subset, pred_vis_thresed[:, :, selected_indices], save_video=False, skip_track= True, sigsq=pred_sigsq_subset)

        # res_video_pix = track_visualizer.visualize(255 * image_subset, pred_tracks_subset, pred_vis_thresed[:, :, selected_indices], save_video=False, skip_track= True)


        # res_video_pred = add_colored_border(res_video_pred, 5, color=(0, 255, 255))
        # res_video_gt = add_colored_border(res_video_gt, 5, color=(255, 0, 0))
        res_combined = torch.cat([res_video_pred, res_video_gt, res_video_raw, res_video_pixpoint, res_video_pix], dim=-2)

        # env_name = f"perfect_{cfg.exp_name}_Single" if single_pt else f"perfect_{cfg.exp_name}"
        env_name = f"visual_{cfg.exp_name}"
        print(env_name)
        # 
        # viz.images((res_combined[0] / 255).clamp(0, 1), env=env_name, win="imgs")

        # track_visualizer.save_video(res_combined, filename=f"sample_{step}")

        # wide_list = list(video.unbind(1))
        # wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
        # clip = ImageSequenceClip(wide_list[2:-1], fps=self.fps)

        # # Write the video file
        # save_path = os.path.join(self.save_dir, f"{filename}_pred_track.mp4")
        # clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

        

        return res_combined, save_dir
        """


def add_colored_border(video_tensor, border_width, color=(0, 0, 255)):
    """
    Adds a colored border to a video represented as a PyTorch tensor.

    Parameters:
    video_tensor (torch.Tensor): A tensor of shape [batch_size, num_frames, num_channels, height, width].
    border_width (int): The width of the border to add.
    color (tuple): RGB values of the border color as a tuple (R, G, B).

    Returns:
    torch.Tensor: A new tensor with the colored border added, shape [batch_size, num_frames, num_channels, new_height, new_width].
    """
    # Extract original dimensions
    batch_size, num_frames, num_channels, original_height, original_width = video_tensor.shape

    # Calculate new dimensions
    new_height, new_width = original_height + 2 * border_width, original_width + 2 * border_width

    # Create new tensor filled with the specified color
    new_video_tensor = torch.zeros([batch_size, num_frames, num_channels, new_height, new_width])
    for i, c in enumerate(color):
        new_video_tensor[:, :, i, :, :] = c

    # Copy original video frames into new tensor
    new_video_tensor[
        :, :, :, border_width : border_width + original_height, border_width : border_width + original_width
    ] = video_tensor

    return new_video_tensor

