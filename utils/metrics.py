import torch
import torch.nn.functional as F
import numpy as np
import scipy
import ot
import os
from matplotlib import pyplot as plt

"""
Useful Functions 
"""
def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0

    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
def iou_dist(m1, m2,label_range, eps=1e-8):
    intersection = torch.sum(m1 * m2, dim=[-1, -2])  # keep batch and class dimension
    sum = torch.sum(m1, dim=[-1, -2]) + torch.sum(m2, dim=[-1, -2])
    union = sum - intersection
    iou = (intersection + eps) / (union + eps)

    if label_range is not None and len(label_range) > 1:  # for cityscape
        iou[sum == 0] = np.nan
        return nanmean(1.0 - iou, -1)
    else:  # for lidc
        return torch.mean(1.0 - iou, -1)

def get_uniform_prob(N, device):
    return torch.empty(1, N, dtype=torch.float,
                       requires_grad=False, device=device).fill_(1.0 / N).squeeze(0)

def get_weight(prob_sample, prob_gt):
    return torch.bmm(prob_sample.unsqueeze(-1), prob_gt.unsqueeze(1))


def get_cost_matrix(sample_arr, gt_arr, M, N, d_sy, use_symmetric=False, label_range = [1], dist_fct = iou_dist):
    # dist_fct = monai.losses.DiceLoss(reduction='none', jaccard=True, include_background=False, smooth_nr = 0.0, smooth_dr = 0.0)
    for i in range(N):
        if i < M and use_symmetric:  # symmetric
            start, end = i, M
        else:
            start, end = 0, M
            use_symmetric = False
        for j in range(start, end):
            cij = (dist_fct(sample_arr[:, i, ...], gt_arr[:, j, ...], label_range=label_range))
            d_sy[:, i, j] = cij
            if use_symmetric:
                d_sy[:, j, i] = cij
    return d_sy

def cal_metrics_batch(sample_arr, gt_arr, prob_sample = None, prob_gt = None, to_one_hot = True,
                      nlabels=1, label_range = None, ret_h_score = True):

    """
    :param sample_arr: torch.tensor, expected shape (B,N,C,H,W) or (B,N,H,W)
    :param gt_arr:  torch.tensor, expected shape (B,M,C,H,W) or (B,M,H,W)
    :param prob_sample: torch.tensor, expected shape (B,N)
    :param prob_gt: torch.tensor, expected shape (B,M)
    :param to_one_hot: Set to True if the input arr is not in one hot format.
    :param nlabels: Class number. This is used for one_hot encoding.
                    For LIDC, nlabels = 2, for Cityscapes, nlabels = 19.
    :param label_range: The evaluated class, for LIDC use [1], for Cityscapes use the ten flipping class ids.
    :param ret_h_score: whether to return Matched-IoU. If set to False, this metric only return the GED score.

    :return: A list containing both GED score and M-IoU score (if ret_h_score is set to True).
    """

    if nlabels > 2:
        if to_one_hot:
            sample_arr = F.one_hot(sample_arr, nlabels).permute(0,1,-1,2,3)
            gt_arr = F.one_hot(gt_arr, nlabels).permute(0,1,-1,2,3)

        if label_range is not None:
            sample_arr = sample_arr[:,:,label_range]
            gt_arr = gt_arr[:, :,label_range]

    else:
        sample_arr = sample_arr.unsqueeze(2)
        gt_arr = gt_arr.unsqueeze(2)

    B,N,C,H,W = sample_arr.shape
    B,M,C,H,W = gt_arr.shape

    if ((sample_arr > 1)).any():
        # Means samples are in logit level, do softmax
        sample_arr = torch.nn.functional.softmax(sample_arr, dim=2)

    if prob_sample is None:
        prob_sample = get_uniform_prob(N, sample_arr.device).repeat(B,1)
    if prob_gt is None:
        prob_gt = get_uniform_prob(M, sample_arr.device).repeat(B,1)

    p_sy = get_weight(prob_sample, prob_gt)
    p_ss = get_weight(prob_sample, prob_sample)
    p_yy = get_weight(prob_gt, prob_gt)

    d_sy = torch.zeros((B,N,M), device = prob_sample.device)
    d_ss = torch.zeros((B,N,N), device = prob_sample.device)
    d_yy = torch.zeros((B,M,M), device = prob_sample.device)

    d_sy = get_cost_matrix(sample_arr, gt_arr, M, N, d_sy, label_range=label_range)
    d_ss = get_cost_matrix(sample_arr, sample_arr, N, N, d_ss, use_symmetric= True, label_range = label_range)
    d_yy = get_cost_matrix(gt_arr, gt_arr, M, M, d_yy, use_symmetric=True, label_range=label_range)

    iou = (d_sy*p_sy).sum([1,2])
    diversity = (d_ss*p_ss).sum([1,2])
    gt_diversity = (d_yy*p_yy).sum([1,2])

    ged = (2*iou - diversity -gt_diversity).double()# average batch
    avg_ged = ged.nanmean()

    return_list = [avg_ged]

    if ret_h_score:

        # If all weights are in uniform distribution, we follow previous work to use hungarian-matching algorithm
        if (prob_gt.diff() == 0 ).all() and (prob_sample.diff() == 0 ).all() and (N%M == 0): # uniform
            # repeat gt_arr
            n_repeats = N // M
            cost_matrix_array = d_sy.repeat( 1, 1, n_repeats).cpu().numpy().astype(np.float64)
            # loop for a batch
            h_score = 0.0
            for b in range(B):
                try:
                    h_score += (1 - cost_matrix_array[b])[
                        scipy.optimize.linear_sum_assignment(cost_matrix_array[b])].mean()
                except:
                    h_score += 0.0
            avg_h_score = torch.tensor(h_score / B, device=avg_ged.device, dtype=avg_ged.dtype)
        else:
            cost_matrix_array = d_sy.cpu().numpy().astype(np.float64)
            prob_sample_array = prob_sample.cpu().numpy().astype(np.float64)
            prob_gt_array = prob_gt.cpu().numpy().astype(np.float64)

            # loop for a batch
            h_score = 0.0

            for b in range(B):
                prob_sample_array[b] /= prob_sample_array[b].sum()
                prob_gt_array[b] /= prob_gt_array[b].sum()
                P = ot.emd(prob_sample_array[b], prob_gt_array[b], cost_matrix_array[b])
                h_score += np.sum(P * (1-cost_matrix_array[b]))

            avg_h_score = torch.tensor(h_score / B, device=avg_ged.device, dtype=avg_ged.dtype)

        return_list.append(avg_h_score)

    return return_list

if __name__ == '__main__':
    # debug
    B,C,M,N,H,W = 1,2,2,2,2,2

    sample_arr = torch.randn((B, N, C, H, W)).argmax(1)
    sample_prob = torch.nn.functional.softmax(torch.randn(B,N), dim = 1)

    gt_arr = torch.randn((B,C,M,H,W)).argmax(1)
    gt_prob = torch.nn.functional.softmax(torch.randn(B,M), dim = 1)

    metrics = cal_metrics_batch(sample_arr, gt_arr, sample_prob, gt_prob, nlabels=C)

    print(f'GED: {metrics[0].item()}, M-IoU: {metrics[1].item()}')