"""
The module produces a 3D layout .obj object from the model's prediction
and an aligned image.
Source: assessed on 04/10/2022 from:
https://github.com/sunset1995/HorizonNet/blob/master/layout_viewer.py

It is modified to return a 3D layout .obj object.
"""
import sys

sys.path.append("../horizon_net/")

import json
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.signal import correlate2d
from scipy.ndimage import shift

from misc.post_proc import np_coor2xy, np_coorx2u, np_coory2v
from eval_general import layout_2_depth


def convert_to_3D(aligned_image, inferenced_result):

    equirect_texture = np.array(aligned_image)
    H, W = equirect_texture.shape[:2]

    cor_id = np.array(inferenced_result["uv"], np.float32)
    cor_id[:, 0] *= W
    cor_id[:, 1] *= H
    # Convert corners to layout
    depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(
        cor_id, H, W, return_mask=True
    )
    coorx, coory = np.meshgrid(np.arange(W), np.arange(H))
    us = np_coorx2u(coorx, W)
    vs = np_coory2v(coory, H)
    zs = depth * np.sin(vs)
    cs = depth * np.cos(vs)
    xs = cs * np.sin(us)
    ys = -cs * np.cos(us)

    # Aggregate mask
    mask = np.ones_like(floor_mask)
    # ignore ceiling
    mask &= ~ceil_mask

    # Prepare ply's points and faces
    xyzrgb = np.concatenate(
        [xs[..., None], ys[..., None], zs[..., None], equirect_texture], -1
    )
    xyzrgb = np.concatenate([xyzrgb, xyzrgb[:, [0]]], 1)
    mask = np.concatenate([mask, mask[:, [0]]], 1)
    lo_tri_template = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]])
    up_tri_template = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 1]])
    ma_tri_template = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
    lo_mask = correlate2d(mask, lo_tri_template, mode="same") == 3
    up_mask = correlate2d(mask, up_tri_template, mode="same") == 3
    ma_mask = (
        (correlate2d(mask, ma_tri_template, mode="same") == 3) & (~lo_mask) & (~up_mask)
    )
    ref_mask = (
        lo_mask
        | (correlate2d(lo_mask, np.flip(lo_tri_template, (0, 1)), mode="same") > 0)
        | up_mask
        | (correlate2d(up_mask, np.flip(up_tri_template, (0, 1)), mode="same") > 0)
        | ma_mask
        | (correlate2d(ma_mask, np.flip(ma_tri_template, (0, 1)), mode="same") > 0)
    )
    points = xyzrgb[ref_mask]

    ref_id = np.full(ref_mask.shape, -1, np.int32)
    ref_id[ref_mask] = np.arange(ref_mask.sum())
    faces_lo_tri = np.stack(
        [
            ref_id[lo_mask],
            ref_id[shift(lo_mask, [1, 0], cval=False, order=0)],
            ref_id[shift(lo_mask, [1, 1], cval=False, order=0)],
        ],
        1,
    )
    faces_up_tri = np.stack(
        [
            ref_id[up_mask],
            ref_id[shift(up_mask, [1, 1], cval=False, order=0)],
            ref_id[shift(up_mask, [0, 1], cval=False, order=0)],
        ],
        1,
    )
    faces_ma_tri = np.stack(
        [
            ref_id[ma_mask],
            ref_id[shift(ma_mask, [1, 0], cval=False, order=0)],
            ref_id[shift(ma_mask, [0, 1], cval=False, order=0)],
        ],
        1,
    )
    faces = np.concatenate([faces_lo_tri, faces_up_tri, faces_ma_tri])
    reversed_faces = np.array([np.array(face[::-1]) for face in faces])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points[:, :3])
    mesh.vertex_colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.0)
    mesh.triangles = o3d.utility.Vector3iVector(np.concatenate([faces, reversed_faces]))

    return mesh
