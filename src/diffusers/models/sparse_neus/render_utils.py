import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
import trimesh
from icecream import ic

# from ops.back_project import cam2pixel
from diffusers.models.sparse_neus.ops.back_project import cam2pixel
import pdb


def sample_pdf(bins, weights, n_samples, det=False):
    '''
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    '''
    device = weights.device

    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]).to(device), cdf], -1)

    # if bins.shape[1] != weights.shape[1]:  # - minor modification, add this constraint
    #     cdf = torch.cat([torch.zeros_like(cdf[..., :1]).to(device), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(device)

    # Invert CDF
    u = u.contiguous()
    # inds = searchsorted(cdf, u, side='right')
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    # pdb.set_trace()
    return samples


def sample_ptsFeatures_from_featureVolume(pts, featureVolume, vol_dims=None, partial_vol_origin=None, vol_size=None):
    """
    sample feature of pts_wrd from featureVolume, all in world space
    :param pts: [N_rays, n_samples, 3]
    :param featureVolume: [C,wX,wY,wZ]
    :param vol_dims: [3] "3" for dimX, dimY, dimZ
    :param partial_vol_origin: [3]
    :return: pts_feature: [N_rays, n_samples, C]
    :return: valid_mask: [N_rays]
    """

    N_rays, n_samples, _ = pts.shape

    assert vol_dims is not None

    if vol_dims is None:
        pts_normalized = pts
    else:
        # normalized to (-1, 1)
        pts_normalized = 2 * (pts - partial_vol_origin[None, None, :]) / (vol_size * (vol_dims[None, None, :] - 1)) - 1
    
    # np.save('debug/pts_normalized.npy', pts_normalized.detach().cpu().numpy())

    valid_mask = (torch.abs(pts_normalized[:, :, 0]) < 1.0) & (
            torch.abs(pts_normalized[:, :, 1]) < 1.0) & (
                         torch.abs(pts_normalized[:, :, 2]) < 1.0)  # (N_rays, n_samples)
    
    # np.save('debug/valid_mask.npy', valid_mask.detach().cpu().numpy())

    pts_normalized = torch.flip(pts_normalized, dims=[-1])  # ! reverse the xyz for grid_sample

    # ! checked grid_sample, (x,y,z) is for (D,H,W), reverse for (W,H,D)
    pts_feature = F.grid_sample(featureVolume[None, :, :, :, :], pts_normalized[None, None, :, :, :],
                                padding_mode='zeros',
                                align_corners=True).view(-1, N_rays, n_samples)  # [C, N_rays, n_samples]

    pts_feature = pts_feature.permute(1, 2, 0)  # [N_rays, n_samples, C]
    return pts_feature, valid_mask


def sample_ptsFeatures_from_featureMaps(pts, featureMaps, w2cs, intrinsics, WH, proj_matrix=None, return_mask=False):
    """
    sample features of pts from 2d feature maps
    :param pts: [N_rays, N_samples, 3]
    :param featureMaps: [N_views, C, H, W]
    :param w2cs: [N_views, 4, 4]
    :param intrinsics: [N_views, 3, 3]
    :param proj_matrix: [N_views, 4, 4]
    :param HW:
    :return:
    """
    # normalized to (-1, 1)
    N_rays, n_samples, _ = pts.shape
    N_views = featureMaps.shape[0]

    if proj_matrix is None:
        proj_matrix = torch.matmul(intrinsics, w2cs[:, :3, :])

    pts = pts.permute(2, 0, 1).contiguous().view(1, 3, N_rays, n_samples).repeat(N_views, 1, 1, 1)
    pixel_grids = cam2pixel(pts, proj_matrix[:, :3, :3], proj_matrix[:, :3, 3:],
                            'zeros', sizeH=WH[1], sizeW=WH[0])  # (nviews, N_rays, n_samples, 2)

    valid_mask = (torch.abs(pixel_grids[:, :, :, 0]) < 1.0) & (
            torch.abs(pixel_grids[:, :, :, 1]) < 1.00)  # (nviews, N_rays, n_samples)
    
    pts_feature = F.grid_sample(featureMaps, pixel_grids,
                                padding_mode='zeros',
                                align_corners=True)  # [N_views, C, N_rays, n_samples]

    if return_mask:
        return pts_feature, valid_mask
    else:
        return pts_feature

@torch.no_grad()
def get_attn_mask(pts, N_views, w2cs, intrinsics, WH, proj_matrix=None, step=1):
    N_rays, n_samples, _ = pts.shape
    if proj_matrix is None:
        proj_matrix = torch.matmul(intrinsics, w2cs[:, :3, :])
    pts = pts.permute(2, 0, 1).contiguous().view(1, 3, N_rays, n_samples).repeat(N_views, 1, 1, 1)
    pixel_grids = cam2pixel(pts, proj_matrix[:, :3, :3], proj_matrix[:, :3, 3:],
                            'zeros', sizeH=WH[1], sizeW=WH[0])  # (nviews, N_rays, n_samples, 2)
    # attn_mask = compute_attn_mask(pixel_grids, WH[1], WH[0], step=step) # N_rays, N_view, H, W
    attn_mask = efficient_compute_attn_mask(pixel_grids, WH[1], WH[0], step=step) # N_rays, N_view, H, W
    return attn_mask

def compute_attn_mask(grid, H, W, step=1):
    """grid: nviews, N_ray, N_sample, 2
    """
    N_v, N_ray, N_sample, _ = grid.shape
    # grid = grid.view(N_v, -1, 2)
    x, y = grid[..., 0], grid[..., 1]

    x = 0.5 * (x + 1.0) * float(W - 1)
    y = 0.5 * (y + 1.0) * float(H - 1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    with torch.no_grad():
        attn_mask_all = torch.zeros(N_ray, N_v, H, W).to(grid.device)
        project_coords = torch.stack([x0, y0], dim=-1) # [N_v, N_ray, N_sample, 2]


        img_coord_x, img_coord_y = torch.meshgrid(torch.arange(H), torch.arange(W))
        img_coords = torch.stack([img_coord_x.T, img_coord_y.T], dim=-1).to(grid.device) # [H, W, 2]

        for i in range(N_v):
            coords = project_coords[i] # [N_ray, N_sample, 2]
            attn_mask = coords[:, None, None, :, :] - img_coords[None, :, :, None, :] # [N_ray, 1, 1, N_sample, 2] - [1, H, W, 1, 2] -> [N_ray, H, W, N_sample, 2]
            attn_mask = (torch.abs(attn_mask) <= step).all(-1) # [N_ray, H, W, N_sample]
            attn_mask = attn_mask.sum(-1) > 0 # [N_ray, H, W]
            attn_mask_all[:, i] = attn_mask
    
    return attn_mask_all

def efficient_compute_attn_mask(grid, H, W, step=1):
    """grid: nviews, N_ray, N_sample, 2
    """
    N_v, N_ray, N_sample, _ = grid.shape
    # grid = grid.view(N_v, -1, 2)
    x, y = grid[..., 0], grid[..., 1] # nv, nr, ns

    x = 0.5 * (x + 1.0) * float(W - 1)
    y = 0.5 * (y + 1.0) * float(H - 1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    flat_indices = torch.stack([(y0 * W + x0),
                                (y0 * W + x1),
                                (y1 * W + x0),
                                (y1 * W + x1)], dim=-1) # N_v, N_rays, N_sample, 4 
    
    with torch.no_grad():
        attn_mask_all = torch.zeros((N_ray, N_v, H*W), dtype=torch.bool).to(grid.device)

        for i in range(N_v):
            inds = flat_indices[i].view(N_ray, -1) # N_rays, N_sample, 4 -> N_rays, N_sample*4
            b_inds = torch.arange(N_ray, device=grid.device).view(-1, 1).expand_as(inds)
            # N_rays, H*W
            attn_mask_all[:, i][b_inds, inds] = True
    
    return attn_mask_all.view(N_ray, N_v, H, W)