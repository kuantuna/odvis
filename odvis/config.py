# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN

def add_odvis_config(cfg):
    """
    Add config for ODVIS.
    """
    cfg.MODEL.ODVIS = CN()
    cfg.MODEL.ODVIS.NUM_CLASSES = 80
    cfg.MODEL.ODVIS.NUM_PROPOSALS = 300

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10
    cfg.INPUT.SAMPLING_INTERVAL = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    cfg.INPUT.COCO_PRETRAIN = False
    cfg.INPUT.PRETRAIN_SAME_CROP = False

    # LOSS
    cfg.MODEL.ODVIS.MASK_WEIGHT = 2.0
    cfg.MODEL.ODVIS.DICE_WEIGHT = 5.0
    cfg.MODEL.ODVIS.GIOU_WEIGHT = 2.0
    cfg.MODEL.ODVIS.L1_WEIGHT = 5.0
    cfg.MODEL.ODVIS.CLASS_WEIGHT = 2.0
    cfg.MODEL.ODVIS.REID_WEIGHT = 2.0
    cfg.MODEL.ODVIS.DEEP_SUPERVISION = True
    cfg.MODEL.ODVIS.MASK_STRIDE = 4
    cfg.MODEL.ODVIS.MATCH_STRIDE = 4
    cfg.MODEL.ODVIS.FOCAL_ALPHA = 0.25
    cfg.MODEL.ODVIS.NO_OBJECT_WEIGHT = 0.1

  
    # Focal Loss.
    cfg.MODEL.ODVIS.USE_FOCAL = True
    cfg.MODEL.ODVIS.USE_FED_LOSS = False
    cfg.MODEL.ODVIS.ALPHA = 0.25
    cfg.MODEL.ODVIS.GAMMA = 2.0
    cfg.MODEL.ODVIS.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.ODVIS.OTA_K = 5

    # Diffusion
    cfg.MODEL.ODVIS.SNR_SCALE = 2.0
    cfg.MODEL.ODVIS.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.ODVIS.USE_NMS = True

    # TRANSFORMER
    cfg.MODEL.ODVIS.NHEADS = 8
    cfg.MODEL.ODVIS.DROPOUT = 0.0
    cfg.MODEL.ODVIS.DIM_FEEDFORWARD = 2048
    cfg.MODEL.ODVIS.ACTIVATION = 'relu'
    cfg.MODEL.ODVIS.ENC_LAYERS = 6
    cfg.MODEL.ODVIS.DEC_LAYERS = 6

    cfg.MODEL.ODVIS.HIDDEN_DIM = 256
    cfg.MODEL.ODVIS.NUM_OBJECT_QUERIES = 300
    cfg.MODEL.ODVIS.DEC_N_POINTS = 4
    cfg.MODEL.ODVIS.ENC_N_POINTS = 4
    cfg.MODEL.ODVIS.NUM_FEATURE_LEVELS = 4
    cfg.MODEL.ODVIS.NUM_CLS = 1
    cfg.MODEL.ODVIS.NUM_REG = 3
    cfg.MODEL.ODVIS.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.ODVIS.NUM_DYNAMIC = 2
    cfg.MODEL.ODVIS.DIM_DYNAMIC = 64


    # Evaluation
    cfg.MODEL.ODVIS.CLIP_STRIDE = 1
    cfg.MODEL.ODVIS.MERGE_ON_CPU = True
    cfg.MODEL.ODVIS.MULTI_CLS_ON = True
    cfg.MODEL.ODVIS.APPLY_CLS_THRES = 0.05

    cfg.MODEL.ODVIS.TEMPORAL_SCORE_TYPE = 'mean' # mean or max score for sequence masks during inference,
    cfg.MODEL.ODVIS.INFERENCE_SELECT_THRES = 0.1  # 0.05 for ytvis
    cfg.MODEL.ODVIS.NMS_PRE =  0.5
    cfg.MODEL.ODVIS.ADD_NEW_SCORE = 0.2
    cfg.MODEL.ODVIS.INFERENCE_FW = True #frame weight
    cfg.MODEL.ODVIS.INFERENCE_TW = True  #temporal weight
    cfg.MODEL.ODVIS.MEMORY_LEN = 3
    cfg.MODEL.ODVIS.BATCH_INFER_LEN = 10

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0  # 0.1

    ## support Swin backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # find_unused_parameters
    cfg.FIND_UNUSED_PARAMETERS = True

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])