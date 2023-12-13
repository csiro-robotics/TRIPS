# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np


def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    hparams = {}

    hparams["data_augmentation"] = (True, True)
    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["resnet18"] = (False, False)
    hparams["resnet34"] = (False, False)
    hparams["resnet_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    hparams["class_balanced"] = (False, False)
    hparams["optimizer"] = ("adam", "adam")

    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet

    if dataset not in SMALL_IMAGES:  # PACS, OfficeHome, DomainNet
        hparams["lr"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        if dataset == "DomainNet":
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5)))
        else:
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5.5)))
        if algorithm == "ARM":
            hparams["batch_size"] = (8, 8)
    else:
        hparams["lr"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
        hparams["batch_size"] = (64, int(2 ** random_state.uniform(3, 9)))

    if dataset in SMALL_IMAGES:
        hparams["weight_decay"] = (0.0, 0.0)
    else:  # PACS, OfficeHome, DomainNet
        hparams["weight_decay"] = (0.0, 10 ** random_state.uniform(-6, -2))

    # -----------------------------------------------------------------
    if "TRIPLET_DIST" in algorithm:
        hparams["backbone_use_bottleneck"] = (True, True)
        hparams["backbone_normalize_feature"] = (True, True)
    elif "ERM_DIST" in algorithm:
        hparams["backbone_use_bottleneck"] = (True, True)
        hparams["backbone_normalize_feature"] = (True, True)
    else:
        hparams["backbone_use_bottleneck"] = (False, False)
        hparams["backbone_normalize_feature"] = (False, False)
        
    hparams["LOSS_mannual_setting"] = (False, False)
    hparams["lambda_c"] = (1.0, 1.0) 
    hparams["lambda_d"] = (1.0, 1.0) 
    hparams["lambda_t"] = (1.0, 1.0) 
    hparams["normalised_feature_offset"] = (1.0, 1.0) 

    hparams["exemplar_mixup"] = (False, False)
    hparams["exemplar_mixup_times"] = (4.0, 4.0) 

    hparams["focal_loss"] = (False, False)
    hparams["focal_loss_gamma"] = (2.0, 2.0) 

    # -----------------------------------------------------------------

    if algorithm in ["DANN", "CDANN"]:
        if dataset not in SMALL_IMAGES:
            hparams["lr_g"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
            hparams["lr_d"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        else:
            hparams["lr_g"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
            hparams["lr_d"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))

        if dataset in SMALL_IMAGES:
            hparams["weight_decay_g"] = (0.0, 0.0)
        else:
            hparams["weight_decay_g"] = (0.0, 10 ** random_state.uniform(-6, -2))

        hparams["lambda"] = (1.0, 10 ** random_state.uniform(-2, 2))
        hparams["weight_decay_d"] = (0.0, 10 ** random_state.uniform(-6, -2))
        hparams["d_steps_per_g_step"] = (1, int(2 ** random_state.uniform(0, 3)))
        hparams["grad_penalty"] = (0.0, 10 ** random_state.uniform(-2, 1))
        hparams["beta1"] = (0.5, random_state.choice([0.0, 0.5]))
        hparams["mlp_width"] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams["mlp_depth"] = (3, int(random_state.choice([3, 4, 5])))
        hparams["mlp_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    elif algorithm == "RSC":
        hparams["rsc_f_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
        hparams["rsc_b_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
    elif algorithm == "SagNet":
        hparams["sag_w_adv"] = (0.1, 10 ** random_state.uniform(-2, 1))
    elif algorithm == "IRM":
        hparams["irm_lambda"] = (1e2, 10 ** random_state.uniform(-1, 5))
        hparams["irm_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )
    elif algorithm in ["Mixup", "OrgMixup"]:
        hparams["mixup_alpha"] = (0.2, 10 ** random_state.uniform(-1, -1))
    elif algorithm == "GroupDRO":
        hparams["groupdro_eta"] = (1e-2, 10 ** random_state.uniform(-3, -1))
    elif algorithm in ("MMD", "CORAL"):
        hparams["mmd_gamma"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm in ("MLDG", "SOMLDG"):
        hparams["mldg_beta"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm == "MTL":
        hparams["mtl_ema"] = (0.99, random_state.choice([0.5, 0.9, 0.99, 1.0]))
    elif algorithm == "VREx":
        hparams["vrex_lambda"] = (1e1, 10 ** random_state.uniform(-1, 5))
        hparams["vrex_penalty_anneal_iters"] = (500, int(10 ** random_state.uniform(0, 4)))
    elif algorithm == "SAM":
        hparams["rho"] = (0.05, random_state.choice([0.01, 0.02, 0.05, 0.1]))
    elif algorithm == "CutMix":
        hparams["beta"] = (1.0, 1.0)
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        hparams["cutmix_prob"] = (1.0, 1.0)
    elif "TRIPLET" in algorithm:
        hparams["w_cross_entropy"] = (False, False)
        hparams["margin"] = (0, 0)
        hparams["CE_loss_no_exemplar"] = (False, False)
        hparams["CE_loss_only_new_cls"] = (False, False)
        hparams["w_triplet"] = (True, True)

    if "DIST" in algorithm:
        hparams["apply_CE_2_all_cls"] = (False, False)
        hparams["DIST_loss_feature"] = (False, False)
        hparams["DIST_loss_feature_CSCCT"] = (False, False)
        hparams["DIST_loss_feature_CSC_offset"] = (0.0, 0.0)
        hparams["DIST_loss_tar_normalized_old_and_new"] = (False, False)
        hparams["DIST_loss_only_exemplar"] = (False, False)
        hparams["DIST_loss_only_new_data"] = (False, False)
        hparams["DIST_loss_rated"] = (False, False)
        hparams["DIST_loss_ratio_for_exemplar"] = (0.25, 0.25)
    
    if "MOV" in algorithm:
        hparams["beta"] = (0.96, 0.96)
    
    if "PROTO" in algorithm:
        hparams['using_delta'] = (False, False)
        hparams['using_gamma_moving_avg'] = (False, False)
        hparams['sigma'] = (0.2, 0.2)
        hparams['PROTO_mean_Balance_beta'] = (1.0, 1.0)
        hparams['PROTO_mean_MovingAvg_eta'] = (0.0, 0.0)
        hparams["PROTO_semantic_shifting"] = (False, False)
        hparams["PROTO_radius_value"] = (-1.0, -1.0)
        hparams["w_proto"] = (True, True)
        hparams["PROTO_augmentation"] = (False, False)
        hparams["PROTO_augmentation_w_COV"] = (False, False)
        hparams["PROTO_cov_Shrinkage_alpha"] = (0.05, 0.05)
        hparams["PROTO_cov_MovingAvg_eta"] = (0.1, 0.1)
        hparams["PROTO_cov_Balance_beta"] = (1.0, 1.0)
        hparams["PROTO_cov_sketching"] = (False, False)
        hparams["PROTO_cov_sketchingRatio"] = (1, 1)
        

    hparams["Data_Augmentation"] = (False, False)
    hparams["Data_Augmentation_Cutout"] = (False, False)
    hparams["Data_Augmentation_Rotation"] = (False, False)

    return hparams


def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}
