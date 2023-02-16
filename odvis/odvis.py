import copy
import math
import random

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.layers import batched_nms
from detectron2.structures import ImageList, Boxes, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .tracker import IDOL_Tracker
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.pos_neg_select import select_pos_neg

__all__ = ["ODVIS"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


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

def clip_mask(Boxes,masks):
    boxes = Boxes.tensor.long()
    assert (len(boxes)==len(masks))
    m_out = []
    k = torch.zeros(masks[0].size()).long().to(boxes.device)
    for i in range(len(masks)):
        mask = masks[i]
        box = boxes[i]
        k[box[1]:box[3],box[0]:box[2]] = 1
        mask *= k
        m_out.append(mask)
        k *= 0
    return torch.stack(m_out)


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)
    num_instances = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(
        torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances)
    return weight_splits, bias_splits


def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mid_size, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    #import pdb;pdb.set_trace()
    output_boxes.clip(results.image_size)
    masks = results.pred_masks
    
    #masks = F.interpolate(masks.unsqueeze(1), size=new_size, mode='bilinear').squeeze(1)
    ################################################
    
    pred_global_masks = aligned_bilinear(masks.unsqueeze(1), 4)
    #import pdb;pdb.set_trace()
    pred_global_masks = pred_global_masks[:, :, :mid_size[0], :mid_size[1]]
    masks = F.interpolate(
                    pred_global_masks,
                    size=(new_size[0], new_size[1]),
                    mode='bilinear',
                    align_corners=False).squeeze(1)
    #################################################
    masks.gt_(0.5)
    #masks = masks.long()
    #import pdb;pdb.set_trace()
    masks = clip_mask(output_boxes,masks)
    #import pdb;pdb.set_trace()
    results.pred_masks = masks
    results = results[output_boxes.nonempty()]
    #import pdb;pdb.set_trace()
    #if results.has("pred_masks"):
        #if isinstance(results.pred_masks, ROIMasks):
        #    roi_masks = results.pred_masks
        #else:
            # pred_masks is a tensor of shape (N, 1, M, M)
        #    roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        #results.pred_masks = roi_masks.to_bitmasks(
        #    results.pred_boxes, output_height, output_width, mask_threshold
        #).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results



def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor
    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor,
                           size=(oh, ow),
                           mode='bilinear',
                           align_corners=True)
    tensor = F.pad(tensor,
                   pad=(factor // 2, 0, factor // 2, 0),
                   mode="replicate")
    return tensor[:, :, :oh - 1, :ow - 1]


@META_ARCH_REGISTRY.register()
class ODVIS(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.ODVIS.NUM_CLASSES
        self.num_proposals = cfg.MODEL.ODVIS.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.ODVIS.HIDDEN_DIM
        self.num_heads = cfg.MODEL.ODVIS.NUM_HEADS
        self.weight_nums = [64, 64, 8]
        self.bias_nums = [8, 8, 1]

        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        self.is_multi_cls = cfg.MODEL.ODVIS.MULTI_CLS_ON
        self.apply_cls_thres = cfg.MODEL.ODVIS.APPLY_CLS_THRES
        self.temporal_score_type = cfg.MODEL.ODVIS.TEMPORAL_SCORE_TYPE
        self.inference_select_thres = cfg.MODEL.ODVIS.INFERENCE_SELECT_THRES
        self.inference_fw = cfg.MODEL.ODVIS.INFERENCE_FW
        self.inference_tw = cfg.MODEL.ODVIS.INFERENCE_TW
        self.memory_len = cfg.MODEL.ODVIS.MEMORY_LEN
        self.nms_pre = cfg.MODEL.ODVIS.NMS_PRE
        self.add_new_score = cfg.MODEL.ODVIS.ADD_NEW_SCORE 
        self.batch_infer_len = cfg.MODEL.ODVIS.BATCH_INFER_LEN

        self.mask_on = cfg.MODEL.MASK_ON

        self.coco_pretrain = cfg.INPUT.COCO_PRETRAIN


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
        self.merge_on_cpu = cfg.MODEL.ODVIS.MERGE_ON_CPU

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
        reid_weight = cfg.ODVIS.REID_WEIGHT
        mask_weight = cfg.ODVIS.MASK_WEIGHT
        dice_weight = cfg.ODVIS.DICE_WEIGHT

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight,
                       "loss_reid": reid_weight/8, "loss_reid_aux": reid_weight*1.5, "loss_mask": mask_weight, "loss_dice": dice_weight}


        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )
        losses = ["labels", "boxes", "mask", "reid"]
        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.merge_device = "cpu" if self.merge_on_cpu else self.device
        self.reid_embed_head = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord, outputs_kernel, proposal_features, mask_feat = self.head(backbone_feats, x_boxes, t, None, True)
        outputs_inst_embed = self.reid_embed_head(proposal_features) #problem on inst_embed fix
        
        #torch.Size([6, 1, 500, 80]), torch.Size([6, 1, 500, 153]), torch.Size([1, 8, 200, 304])
        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord, outputs_kernel, mask_feat, outputs_inst_embed

    
    def forward(self, batched_inputs):
        if self.training:
            key_images, ref_images, images_whwh = self.preprocess_image(batched_inputs)
            gt_instances = self.extract_gt_instances(batched_inputs)
            key_targets, ref_targets, diffused_boxes, _, t = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            diffused_boxes = diffused_boxes * images_whwh[:, None, :]

            key_src = self.backbone(key_images.tensor)
            key_features = [key_src[f] for f in self.in_features]
            
            ref_src = self.backbone(ref_images.tensor)
            ref_features = [ref_src[f] for f in self.in_features]

            outputs_class, outputs_coord, outputs_kernel, key_propsal_features, mask_feat = self.head(key_features, diffused_boxes, t, None, is_key=True)
            _, _, _, ref_proposal_features, _ = self.head(ref_features, diffused_boxes, t, None, is_key=False)

            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_kernels': outputs_kernel[-1], 'mask_feat':mask_feat}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b, 'pred_kernels':c, 'mask_feat':mask_feat}
                                         for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_kernel[:-1])]

            outputs_without_aux = {k: v for k, v in output.items() if k != 'aux_outputs'}
            _, matched_ids = self.criterion.matcher(outputs_without_aux, key_targets)
            ref_cls = self.head.head_series[-1].class_logits(ref_proposal_features).sigmoid()

            contrast_items = select_pos_neg(outputs_coord[-1], matched_ids, ref_targets, key_targets, self.reid_embed_head, key_propsal_features, ref_proposal_features, ref_cls)
            output['pred_qd'] = contrast_items

            loss_dict = self.criterion(output, key_targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        elif self.coco_pretrain:
            images, images_whwh = self.preprocess_coco_image(batched_inputs)
            src = self.backbone(images.tensor)
            features = [src[f] for f in self.in_features]
            output = self.inference_forward(features, images_whwh)
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            kernel_pred = output["pred_kernels"]
            mask_feat = output["mask_feat"]
            results = self.coco_inference(box_cls, box_pred, kernel_pred, mask_feat, images.image_sizes, batched_inputs)
            return results

        else:
            images, _, _ = self.preprocess_image(batched_inputs)
            video_len = len(batched_inputs[0]['file_names'])
            clip_length = self.batch_infer_len
            #split long video into clips to form a batch input 
            if video_len > clip_length:
                num_clips = math.ceil(video_len/clip_length)
                logits_list, boxes_list, embed_list, masks_list, kernels_list, mask_feat_list = [], [], [], [], [], []
                for c in range(num_clips):
                    start_idx = c*clip_length
                    end_idx = (c+1)*clip_length
                    clip_inputs = [{'image':batched_inputs[0]['image'][start_idx:end_idx]}]
                    clip_images, images_whwh, _ = self.preprocess_image(clip_inputs)
                    src = self.backbone(clip_images.tensor)
                    features = [src[f] for f in self.in_features]
                    clip_output = self.inference_forward(features, images_whwh)
                    logits_list.append(clip_output['pred_logits'])
                    boxes_list.append(clip_output['pred_boxes'])
                    embed_list.append(clip_output['pred_inst_embed'])
                    # masks_list.append(clip_output['pred_masks'].to(self.merge_device))
                    kernels_list.append(clip_output['pred_kernels'])
                    mask_feat_list.append(clip_output['mask_feat'])
                output = {
                    'pred_logits':torch.cat(logits_list,dim=0),
                    'pred_boxes':torch.cat(boxes_list,dim=0),
                    'pred_inst_embed':torch.cat(embed_list,dim=0),
                    # 'pred_masks':torch.cat(masks_list,dim=0),
                    'pred_kernels':torch.cat(kernels_list,dim=0),
                    'mask_feat':torch.cat(mask_feat_list,dim=0),
                }    

            else:
                images, images_whwh, _ = self.preprocess_image(batched_inputs)
                src = self.backbone(images.tensor)
                features = [src[f] for f in self.in_features]
                output = self.inference_forward(features, images_whwh)

            idol_tracker = IDOL_Tracker(
                    init_score_thr= 0.2,
                    obj_score_thr=0.1,
                    nms_thr_pre=self.nms_pre,  #0.5
                    nms_thr_post=0.05,
                    addnew_score_thr = self.add_new_score, #0.2
                    memo_tracklet_frames = 10,
                    memo_momentum = 0.8,
                    long_match = self.inference_tw,
                    frame_weight = (self.inference_tw|self.inference_fw),
                    temporal_weight = self.inference_tw,
                    memory_len = self.memory_len
                    )
        
            height = batched_inputs[0]['height']
            width = batched_inputs[0]['width']
            video_output = self.inference(output, idol_tracker, (height, width), images.image_sizes[0])  # (height, width) is resized size,images. image_sizes[0] is original size

            return video_output


    def inference_forward(self, backbone_feats, images_whwh, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        #tensor([ -1., 999.])
        times = list(reversed(times.int().tolist()))
        
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds, outputs_class, outputs_coord, outputs_kernel, mask_feat, outputs_inst_embed = self.model_predictions(backbone_feats, images_whwh, img, time_cond,self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                #true
                score_per_image = outputs_class[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat((img, torch.randn(batch, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
        
        return {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_inst_embed': outputs_inst_embed, 'pred_kernels': outputs_kernel[-1], 'mask_feat': mask_feat}
            

    def coco_inference(self, box_cls, box_pred, kernel_pred, mask_feat, image_sizes, batched_inputs, do_postprocess=True):

        assert len(box_cls) == len(image_sizes)
        results = []

        for i, (logits_per_image, box_pred_per_image, kernel_pred_per_image, mas, image_size) in enumerate(zip(
            box_cls, box_pred, kernel_pred, mask_feat, image_sizes
        )):

            num_instance = len(kernel_pred_per_image)
            weights, biases = parse_dynamic_params(
                kernel_pred_per_image, #[500, 153]
                8,
                self.weight_nums,
                self.bias_nums)
            mask_feat_head = mas.unsqueeze(0).repeat(1, num_instance, 1, 1) #[1, 4000, 120, 216]
            mask_logits = self.mask_heads_forward( #[1, 500, 120, 216]
                mask_feat_head, 
                weights, 
                biases, 
                num_instance)
            mask_pred = mask_logits.reshape(-1, 1, mas.size(1), mas.size(2)) #[500, 1, 120, 216]


            prob = logits_per_image.sigmoid()
            nms_scores,idxs = torch.max(prob,1)
            # boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
            keep_indices = batched_nms(box_pred_per_image,nms_scores,idxs,0.7)  
            prob = prob[keep_indices]
            box_pred_per_image = box_pred_per_image[keep_indices]
            mask_pred_i = mask_pred[keep_indices]

            topk_values, topk_indexes = torch.topk(prob.view(-1), 100, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
            # topk_boxes = topk_indexes // logits_per_image.shape[1]
            labels = topk_indexes % logits_per_image.shape[1]
            scores_per_image = scores
            labels_per_image = labels

            box_pred_per_image = box_pred_per_image[topk_boxes]
            mask_pred_i = mask_pred_i[topk_boxes]

            result = Instances(image_size)
            result.pred_boxes = Boxes(box_pred_per_image)
            result.pred_masks = mask_pred_i.squeeze(1).sigmoid()
            # result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                mid_size = image_size
                r = detector_postprocess(results_per_image, height, width,mid_size)
                processed_results.append({"instances": r})
            return processed_results


    def inference(self, outputs, tracker, ori_size, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # results = []
        video_dict = {}
        vido_logits = outputs['pred_logits']
        # video_output_masks = outputs['pred_masks']
        pred_kernels = outputs['pred_kernels']
        mask_feat = outputs['mask_feat']
        output_h, output_w = mask_feat.shape[-2:]
        video_output_boxes = outputs['pred_boxes']
        video_output_embeds = outputs['pred_inst_embed']
        vid_len = len(vido_logits)
        for i_frame, (logits, kernel, output_boxes, output_embed, mas) in enumerate(zip(
            vido_logits, pred_kernels, video_output_boxes, video_output_embeds, mask_feat
         )):
            scores = logits.sigmoid().cpu().detach()  #[500, 40]
            max_score, _ = torch.max(logits.sigmoid(),1) #[500]
            indices = torch.nonzero(max_score>self.inference_select_thres, as_tuple=False).squeeze(1)#[x]
            if len(indices) == 0:
                topkv, indices_top1 = torch.topk(scores.max(1)[0],k=1)
                indices_top1 = indices_top1[torch.argmax(topkv)]
                indices = [indices_top1.tolist()]
            else:
                nms_scores,idxs = torch.max(logits.sigmoid()[indices],1)
                # boxes_before_nms = box_cxcywh_to_xyxy(output_boxes[indices])
                keep_indices = batched_nms(output_boxes[indices],nms_scores,idxs,0.9)#.tolist()
                indices = indices[keep_indices]
            # # fix the box format
            # scale_x, scale_y = (
            #     ori_size[1] / image_sizes[1],
            #     ori_size[0] / image_sizes[0],
            # )
            # o_boxes = Boxes(output_boxes)
            # o_boxes.scale(scale_x, scale_y)
            # o_boxes.clip(ori_size)
            # output_boxes = o_boxes.tensor       
            box_score = torch.max(logits.sigmoid()[indices],1)[0] #[x] or [1]
            det_bboxes = torch.cat([output_boxes[indices],box_score.unsqueeze(1)],dim=1)
            det_labels = torch.argmax(logits.sigmoid()[indices],dim=1)
            track_feats = output_embed[indices]

            num_instance = len(kernel)
            weights, biases = parse_dynamic_params(
                kernel, #[500, 153]
                8,
                self.weight_nums,
                self.bias_nums)
            mask_feat_head = mas.unsqueeze(0).repeat(1, num_instance, 1, 1) #[1, 4000, 120, 216]
            mask_logits = self.mask_heads_forward( #[1, 500, 120, 216]
                mask_feat_head, 
                weights, 
                biases, 
                num_instance)
            output_mask = mask_logits.reshape(-1, 1, mas.size(1), mas.size(2)) #[500, 1, 120, 216]
            det_masks = output_mask[indices] # 

            bboxes, labels, ids, indices = tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                masks = det_masks,
                track_feats=track_feats,
                frame_id=i_frame,
                indices = indices
                )
            indices = torch.tensor(indices)[ids>-1].tolist()
            ids = ids[ids > -1]
            ids = ids.tolist()
            for query_i, id in zip(indices,ids):
                if id in video_dict.keys():
                    video_dict[id]['masks'].append(output_mask[query_i])
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1
                else:
                    video_dict[id] = {
                        'masks':[None for fi in range(i_frame)], 
                        'boxes':[None for fi in range(i_frame)], 
                        'scores':[None for fi in range(i_frame)], 
                        'valid':0}
                    video_dict[id]['masks'].append(output_mask[query_i])
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1

            for k,v in video_dict.items():
                if len(v['masks'])<i_frame+1: #padding None for unmatched ID
                    v['masks'].append(None)
                    v['scores'].append(None)
                    v['boxes'].append(None)
            check_len = [len(v['masks']) for k,v in video_dict.items()]
            # print('check_len',check_len)

            #  filtering sequences that are too short in video_dict (noise)，the rule is: if the first two frames are None and valid is less than 3
            if i_frame>8:
                del_list = []
                for k,v in video_dict.items():
                    if v['masks'][-1] is None and  v['masks'][-2] is None and v['valid']<3:
                        del_list.append(k)   
                for del_k in del_list:
                    video_dict.pop(del_k)                      

        del outputs
        logits_list = []
        masks_list = []

        for inst_id,m in  enumerate(video_dict.keys()):
            score_list_ori = video_dict[m]['scores']
            scores_temporal = []
            for k in score_list_ori:
                if k is not None:
                    scores_temporal.append(k)
            logits_i = torch.stack(scores_temporal)
            if self.temporal_score_type == 'mean':
                logits_i = logits_i.mean(0)
            elif self.temporal_score_type == 'max':
                logits_i = logits_i.max(0)[0]
            else:
                print('non valid temporal_score_type')
                import sys;sys.exit(0)
            logits_list.append(logits_i)
            
            # category_id = np.argmax(logits_i.mean(0))
            masks_list_i = []
            for n in range(vid_len):
                mask_i = video_dict[m]['masks'][n]
                if mask_i is None:    
                    zero_mask = None # padding None instead of zero mask to save memory
                    masks_list_i.append(zero_mask)
                else:
                    pred_global_masks = aligned_bilinear(mask_i.unsqueeze(1).sigmoid(), 4)
                    pred_global_masks = pred_global_masks[:, :, :image_sizes[0], :image_sizes[1]]
                    masks = F.interpolate(
                                    pred_global_masks,
                                    size=(ori_size[0], ori_size[1]),
                                    mode='bilinear',
                                    align_corners=False).squeeze(1)[0].cpu()
                    masks.gt_(0.5)
                    masks_list_i.append(masks)
            masks_list.append(masks_list_i)
        if len(logits_list)>0:
            pred_cls = torch.stack(logits_list)
        else:
            pred_cls = []

        if len(pred_cls) > 0:
            if self.is_multi_cls:
                is_above_thres = torch.where(pred_cls > self.apply_cls_thres)
                scores = pred_cls[is_above_thres]
                labels = is_above_thres[1]
                out_masks = [masks_list[valid_id] for valid_id in is_above_thres[0]]
            else:
                scores, labels = pred_cls.max(-1)
                out_masks = masks_list
            out_scores = scores.tolist()
            out_labels = labels.tolist()
        else:
            out_scores = []
            out_labels = []
            out_masks = []
        video_output = {
            "image_size": ori_size,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output



    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        images_whwh = []
        for video in batched_inputs:
            for frame in video["image"]:
                h, w = frame.shape[-2:]
                images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
                images.append(self.normalizer(frame.to(self.device)))  
        if self.training:
            bz = len(images)//2
            key_ids = list(range(0,bz*2-1,2)) # evens
            ref_ids = list(range(1,bz*2,2))   # odds
            key_images = [images[_i] for _i in key_ids] # some of the new_targets go here
            ref_images = [images[_i] for _i in ref_ids] # some of the new_targets go here
            key_images = ImageList.from_tensors(key_images, self.size_divisibility)
            ref_images = ImageList.from_tensors(ref_images, self.size_divisibility)
            images_whwh = [images_whwh[_i] for _i in key_ids]
            images_whwh = torch.stack(images_whwh)
            return key_images, ref_images, images_whwh
        else:
            images = ImageList.from_tensors(images, self.size_divisibility)
            images_whwh = torch.stack(images_whwh)
            return images, images_whwh, None


    def preprocess_coco_image(self, batched_inputs):
        images = []
        images_whwh = []
        for x in batched_inputs:
            image = x["image"]
            h, w = image.shape[-2:]
            images.append(self.normalizer(image.to(self.device)))
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images = ImageList.from_tensors(images, self.size_divisibility)
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
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
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
                    if k != "image_size_xyxy":
                        det_target[k] = v[valid_i]
                for k,v in ref_target.items():
                    if k != "image_size_xyxy":
                        ref_target[k] = v[valid_i]
        return det_targets, ref_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)


    def mask_heads_forward(self, features, weights, biases, num_instances):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x,
                        w,
                        bias=b,
                        stride=1,
                        padding=0,
                        groups=num_instances)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

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


def segmentation_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
    ):

    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        mask = F.interpolate(results.pred_masks.float(), size=(output_height, output_width), mode='nearest')
        mask = mask.squeeze(1).byte()
        results.pred_masks = mask

    return results
