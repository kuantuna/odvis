import copy
import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import ImageList, Boxes

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.pos_neg_select import select_pos_neg

__all__ = ["ODVIS"]


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(x):
    return x is not None



@META_ARCH_REGISTRY.register()
class ODVIS(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.ODVIS.NUM_CLASSES

        self.backbone = build_backbone(cfg)

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.ODVIS.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.ODVIS.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        self.deep_supervision = cfg.MODEL.ODVIS.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.ODVIS.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.ODVIS.USE_FED_LOSS
        self.use_nms = cfg.MODEL.ODVIS.USE_NMS

        class_weight = cfg.MODEL.ODVIS.CLASS_WEIGHT
        giou_weight = cfg.MODEL.ODVIS.GIOU_WEIGHT
        l1_weight = cfg.MODEL.ODVIS.L1_WEIGHT
        no_object_weight = cfg.MODEL.ODVIS.NO_OBJECT_WEIGHT

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}


        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )
        losses = ["labels", "boxes", "mask"]
        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # self.reid_embed_head = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

    
    def forward(self, batched_inputs):
        if self.training:
            images, images_whwh = self.preprocess_image(batched_inputs)
            gt_instances = self.extract_gt_instances(batched_inputs)
            key_targets, ref_targets, diffused_boxes, _, t = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            diffused_boxes = diffused_boxes * images_whwh[:, None, :]

            # ?????
            src = self.backbone(images.tensor)
            features = [src[f] for f in self.features]
            key_features, ref_features = self.divide_features(features)
            # ????

            outputs_class, outputs_coord, outputs_kernel, key_propsal_features, mask_feat = self.head(key_features, diffused_boxes, t, None, is_key=True)
            # _, _, _, ref_proposal_features, _ = self.head(ref_features, diffused_boxes, t, None, is_key=False)

            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_kernels': outputs_kernel[-1], 'mask_feat':mask_feat}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, key_targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict


            """
            select_pos_neg(, , ref_targets, key_targets, self.reid_embed_head, key_propsal_features[-1], ref_proposal_features[-1], )

            f_mask = self.mask_branch(key_features)
            outputs_class, outputs_coord, outputs_kernel, k_out = self.decoder(key_features, diffused_boxes, t, None)
            pred_masks = self.mask_head(outputs_kernel, f_mask)

            r_out = self.decoder(ref_features, diffused_boxes, t, None)
            self.contrastive_head(k_out, r_out)
            """
            """
            if not self.training:
                results = self.ddim_sample(batched_inputs, features, images_whwh, images)
                return results
            """





    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        images_whwh = []
        for video in batched_inputs:
            for frame in video["image"]:
                h, w = frame.shape[-2:]
                images_whwh.append(torch.tensor([w, h, w, h]), dtype=torch.float32, device=self.device)
                images.append(self.normalizer(frame.to(self.device)))
        images = ImageList.from_tensors(images)
        images_whwh = torch.stack(images_whwh)
        return images, images_whwh

    def extract_gt_instances(self, batched_inputs):
        gt_instances = []
        for video in batched_inputs:
            for frame in video["instances"]:
                gt_instances.append(frame.to(self.device))
        return gt_instances

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for idx, targets_per_image in enumerate(targets):
            target = {}

            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            if idx % 2 == 0:
                d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
                diffused_boxes.append(d_boxes)
                noises.append(d_noise)
                ts.append(d_t)

            if len(targets_per_image) == 0:
                gt_masks = torch.zeros((h, w))
            else:
                gt_masks = targets_per_image.gt_masks
            
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["masks"] = gt_masks.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            inst_ids = targets_per_image.gt_ids
            valid_id = inst_ids!=-1  # if a object is disappeared，its gt_ids is -1
            target["inst_id"] = inst_ids
            target["valid"] = valid_id
            new_targets.append(target)
        bz = len(new_targets)//2
        key_ids = list(range(0,bz*2-1,2)) # evens
        ref_ids = list(range(1,bz*2,2))   # odds
        det_targets = [new_targets[_i] for _i in key_ids] # some of the new_targets go here
        ref_targets = [new_targets[_i] for _i in ref_ids] # some of the new_targets go here
        for i in range(bz):  # fliter empety object in key frame
            det_target = det_targets[i]
            ref_target = ref_targets[i]
            if False in det_target['valid']:
                valid_i = det_target['valid'].clone()
                for k,v in det_target.items():
                    det_target[k] = v[valid_i]
                for k,v in ref_target.items():
                    ref_target[k] = v[valid_i]
        return det_targets, ref_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)


    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4, # create random 497 boxes
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0) # concatenating newly created boxes
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def divide_features(self, features):
        """
        Fix this later!
        """
        key_features = 1
        ref_features = 2
        return key_features, ref_features

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x