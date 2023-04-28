import torch
import torch.nn as nn
import torch.distributions as td
import utils.metrics as metrics
import math
from models.backbone import Unet, Conv2DSequence, weights_init, mu_init
import numpy as np

class Gating(nn.Module):
    """
    Get probabilities for experts using the gating network.
    """
    def __init__(self, num_filter, num_expert):
        super(Gating, self).__init__()
        self.num_expert = num_expert
        self.fc_layer = nn.Sequential(
            nn.Linear(num_filter, num_filter, bias=True),
            nn.ReLU(),
            nn.Linear(num_filter, num_filter, bias=True),
            nn.ReLU(),
            nn.Linear(num_filter, self.num_expert,  bias=False)
        )

    def forward(self, encoding):
        # Global average pooling
        z = self.fc_layer(encoding) # B,D
        prob = torch.nn.functional.softmax(z, dim=-1).float()
        return prob

class Noise_injector(nn.Module):
    """
    Concatenate the feature map and the sample taken from the latent space.
    """
    def __init__(self, n_hidden, z_dim, n_channels_out):

        super(Noise_injector, self).__init__()

        self.n_channels_out = n_channels_out
        self.n_hidden = n_hidden
        self.z_dim = z_dim

        self.residual = nn.Linear(self.z_dim, self.n_hidden)
        self.scale = nn.Linear(self.z_dim, self.n_hidden)
        self.last_layers = Conv2DSequence(self.n_hidden, self.n_channels_out, kernel=1, depth=3)

        self.residual.apply(weights_init)
        self.scale.apply(weights_init)

    def forward(self, feature_map, z):

        B, C, H, W = feature_map.shape
        _, N, D = z.shape

        feature_map = feature_map.expand(N, B, C, H, W).transpose(0, 1).reshape(B * N, C, H, W)
        z = z.reshape(B * N, D)

        residual = self.residual(z).view(z.shape[0], self.n_hidden, 1, 1)
        scale = self.scale(z).view(z.shape[0], self.n_hidden, 1, 1)

        feature_map = (feature_map + residual) * (scale + 1e-5)

        return self.last_layers(feature_map)

class MoSE(nn.Module):
    """
    The proposed Mixture of Stochastic Experts framework.
    """
    def __init__(self,
        # Params for network structure.
        input_channels = 1,
        num_classes = 2,
        num_filters = None,
        # Params for experts.
        latent_dim = 1,
        num_expert = 4,
        sample_per_mode = 4,
        # Params for gating.
        gating_feature_level=-1,
        gating_input_layer = 5,
        # Params for loss and metrics.
        softmax=True,
        loss_fn = None,
        masked_pred = False,
        eval_class_ids = None,
        eval_sample_num = None,
        ):
        super(MoSE, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.num_expert = num_expert
        self.latent_dim = latent_dim
        self.sample_per_mode = sample_per_mode
        self.eval_sample_num = eval_sample_num
        self.masked_pred = masked_pred
        self.eval_class_ids = eval_class_ids
        self.softmax = softmax

        self.mu = nn.Parameter(torch.tensor(mu_init(self.num_expert, self.latent_dim), dtype=torch.float),
                               requires_grad=True)
        self.log_sigma = nn.Parameter(torch.ones_like(self.mu) * math.log((1 / 8)), requires_grad=True)

        self.backbone = Unet(input_channels, num_classes, num_filters, global_layer = gating_input_layer,
                             apply_last_layer=False)
        self.gating = Gating(self.num_filters[gating_feature_level], self.num_expert)
        self.fuse = Noise_injector(self.num_filters[0], self.latent_dim, self.num_classes)

        self.loss_fn = loss_fn

    def forward(self, input, label=None, prob_gt = None, val=False):
        # Preparation
        B, _, H, W = input.shape
        CL, K, S, L = self.num_classes, self.num_expert, self.sample_per_mode, self.latent_dim
        if val and self.eval_sample_num is not None:
            N = self.eval_sample_num
        else:
            N = K * S

        # Forward

        # Get global feature and dense feature from the Segmentation Backbone
        u_g, u_d = self.backbone(input)

        # Get the expert probabilities form the gating module.
        expert_probs = self.gating(u_g)

        # Get latent codes and corresponding probabilities in either standard-sampling form or weighted form.
        if val and self.eval_sample_num is not None:
            # Standard-sampling. We only use this form during inference.
            latent_codes, sample_probs = self.get_latent_codes_sampling(expert_probs, self.mu, self.log_sigma, N, B)
        else:
            # Weighted form.
            dist = td.Normal(loc=self.mu, scale=torch.exp(self.log_sigma) + 1e-8)
            latent_codes = dist.rsample([B, S]).permute(0, 2, 1, 3).reshape(B, N, L)
            sample_probs = expert_probs.expand(S, B, K).permute(1, 2, 0).flatten(1) / S

        # Fuse the latent codes with the semantic features and get the final predictions.
        pred = self.fuse(u_d, latent_codes).reshape(B, N, CL, H, W)

        if self.softmax:
            pred = torch.softmax(pred, 2)
        if self.masked_pred: # For cityscapes only. We follow the convention to ignore void classes.
            pred = self.masked_pred_func(pred, label)

        if val: # For evaluation, calculate the metrics, otherwise, return the loss.
            metric = metrics.cal_metrics_batch((pred.argmax(2)).long(), (label).long(), sample_probs, prob_gt,
                                    nlabels=self.num_classes, label_range=self.eval_class_ids)
            return metric, pred, sample_probs
        else:
            losses = self.loss_fn(label, pred, sample_probs, prob_gt, sample_shape = (K,S)) # Our OT-based loss
            return losses

    def get_latent_codes_sampling(self, prob, mu, log_sigma, S, B):
        """
        The standard sampling process, which we first sample expert ids,
         and then sample from the expert-specific latent priors.
        """

        # Sample from the categorical distribution.
        categ_dist = td.one_hot_categorical.OneHotCategorical(probs=prob)
        expert_id = categ_dist.sample([S])  # S,B,N
        expert_id = expert_id.permute(1, 0, 2)  # B,S,N

        # Get expert-specific gaussian distribution params.
        mu_sa = expert_id.bmm(mu.expand(B, -1, -1))  # (B,S,N) x (B,N,-1) -> (B,S,-1)
        logsigma_sa = expert_id.bmm(log_sigma.expand(B, -1, -1))

        # Sample from the expert-specific gaussian distribution.
        gaussian_dist = td.Normal(loc=mu_sa, scale=torch.exp(logsigma_sa) + 1e-8)
        latent_codes = gaussian_dist.rsample()

        # Samples have the equal weights.
        prob = torch.nn.functional.softmax(torch.ones(size=(B, S), device=prob.device),
            dim=-1).float()

        return latent_codes, prob

    def masked_pred_func(self, pred, label):
        """
        Follow the convention, we do not calculate loss on the void classes on Cityscapes.
        """
        ignore_mask = torch.where(label[:, 0].unsqueeze(1).repeat(1, self.sample_per_mode, 1, 1) == 0)

        w = torch.ones_like(pred)
        r = torch.zeros_like(pred)
        w[ignore_mask[0], ignore_mask[1], :, ignore_mask[2], ignore_mask[3]] = 0.
        r[ignore_mask[0], ignore_mask[1], 0, ignore_mask[2], ignore_mask[3]] = 1.

        pred = pred * w + r
        return pred
