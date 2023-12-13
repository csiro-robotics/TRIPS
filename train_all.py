import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
from domainbed.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="datadir/")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument("--trial_seed", type=int, default=0, help="Trial number (used for seeding split_dataset and random_hparams). Use 0, 1, and 2.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps. Default is dataset-dependent.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for each domain. Default is dataset-dependent.")
    parser.add_argument("--checkpoint_freq", type=int, default=None, help="Checkpoint every N steps. Default is dataset-dependent.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate used to train the model.")  
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer used to train the model. Choosing from: adam, sgd")
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)  
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument("--evalmode", default="fast", help="[fast, all]. if fast, ignore train_in datasets in evaluation time.")
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    # -------------------------------------------------
    parser.add_argument("--off_swad", action="store_true", help="Turn off the SWAD training strategy.")
    parser.add_argument("--current_session", default=0, type=int, help="Current session for incremental learning. Base session is step 0.")
    parser.add_argument("--num_old_cls", default=0, type=int, help="Number of old class for incremental learning.")
    parser.add_argument("--num_new_cls", default=0, type=int, help="Number of new class for incremental learning.")
    # -------------------------------------------------
    parser.add_argument("--Data_Augmentation", action="store_true", help="Training data augmentation (rotation).")
    # -------------------------------------------------
    parser.add_argument("--num_of_exemplar", default=0, type=int, help="Number of exemplar per class per domain for each old class.")
    parser.add_argument("--CE_loss_no_exemplar", action="store_true", help="Cross Entropy Loss: old exemplar is not passed to cross entropy loss.")
    parser.add_argument("--CE_loss_only_new_cls", action="store_true", help="Cross Entropy Loss: cross entropy is only applied to new class.")
    parser.add_argument("--apply_CE_2_all_cls", action="store_true", help="No matter whether exemplar provided, apply cross entropy loss to all classes.")
    parser.add_argument("--DIST_loss_feature", action="store_true", help="Distillation Loss: apply distillation at feature from the backbone.")
    parser.add_argument("--DIST_loss_feature_CSCCT", action="store_true", help="Distillation Loss: apply CSCCT feature loss instead of L2 distillation at feature from the backbone.")
    parser.add_argument("--DIST_loss_feature_CSC_offset", default=0, type=float, help="Hyperparameter - csc feature distillation loss offset.")
    parser.add_argument("--DIST_loss_tar_normalized_old_and_new", action="store_true", help="Distillation Loss: target output is normalized over both old and new classes.")
    parser.add_argument("--DIST_loss_only_exemplar", action="store_true", help="Distillation Loss: Only exemplar is passed to distillation loss.")
    parser.add_argument("--DIST_loss_only_new_data", action="store_true", help="Distillation Loss: Only new data is passed to distillation loss.")
    parser.add_argument("--DIST_loss_rated", action="store_true", help="Distillation Loss: Both exemplar and new data are passed to distillation loss but rated.")
    parser.add_argument("--DIST_loss_ratio_for_exemplar", default=0.25, type=float, help="Hyperparameter - distillation loss ratio for exemplars.")
    # -------------------------------------------------
    parser.add_argument("--load_old_info", type=str, default=None, help="Load old parameter to the new model.")
    parser.add_argument("--model_type", type=str, default='iid', help="Chosen from: iid, oracle, last_step")
    # -------------------------------------------------
    parser.add_argument("--store_ewc_importance", action="store_true", help="Store the EWC importance matrix.")
    parser.add_argument("--store_mas_importance", action="store_true", help="Store the MAS importance matrix")
    # -------------------------------------------------
    parser.add_argument("--metric_type", type=str, default='l2', help="Chosen from: l1 (absolute distance), l2 (euclidean distance), or cosine_similarity.")
    # -------------------------------------------------
    parser.add_argument("--TRIPLET_w_cross_entropy", action="store_true", help="Using cross-entropy when using triplet loss.")
    parser.add_argument("--margin", type=float, default=0, help="Hyperparameter - margin, margin for triplet loss.")
    parser.add_argument("--triplet_dist_type", type=str, default="cosine_dist", help="Chosen from cosine_dist or euclidean_dist")
    parser.add_argument("--TRIPLET_feature_offset", default=1, type=int, help="Hyperparameter - offset, offsetting the normalized feature.")
    parser.add_argument("--No_TRIPLET", action="store_true", help="Do not use triplet loss.")
    # -------------------------------------------------
    parser.add_argument("--hparam_temperature", default=1, type=int, help="Hyperparameter - temperature, used for knowledge distillation type method.")
    parser.add_argument("--hparam_ewc_scale", default=1000, type=int, help="Hyperparameter - ewc_scale, used for balancing different loss terms for ewc method.")
    parser.add_argument("--hparam_mas_scale", default=1000, type=int, help="Hyperparameter - mas_scale, used for balancing different loss terms for mas method.")
    # -------------------------------------------------
    parser.add_argument("--No_Proto", action="store_true", help="Do not use prototype.")
    parser.add_argument("--PROTO_class_wise_domain_wise", action="store_true", help="Utilizing which type of prototypes: class_wise, or class_wise_domain_wise.")
    parser.add_argument("--PROTO_semantic_shifting", action="store_true", help="Dynamic updating the location of old prototypes by mimicing new data shifting trace.")
    parser.add_argument("--PROTO_AUG_before_shifting", action="store_true", help="Do old feature augmentation first before shifting")
    parser.add_argument("--PROTO_using_delta", action="store_true", help="Using delta to calculate shifting or directly using features.")
    parser.add_argument("--hparam_PROTO_sigma", default=0.5, type=float, help="Hyperparameter - sigma, used for calculating the shifted prototype.")
    parser.add_argument("--hparam_PROTO_mean_Balance_beta", default=1.0, type=float, help="Hyperparameter - PROTO_mean_Balance_beta (alpha), used for calculating the shifted prototype.")
    parser.add_argument("--hparam_PROTO_mean_MovingAvg_eta", default=0.1, type=float, help="Hyperparameter - PROTO_mean_MovingAvg_eta (gamma), used for calculating the shifted prototype.")
    # ----------
    parser.add_argument("--PROTO_augmentation", action="store_true", help="Sampling plenty of samples for the stored average prototypes.")
    parser.add_argument("--PROTO_augmentation_w_COV", action="store_true", help="Sampling plenty of samples for the stored average prototypes with covariance.")
    parser.add_argument("--hparam_PROTO_cov_Shrinkage_alpha", type=float, default=0.05, help="Shrinkage value used for prototype augmentation.")
    parser.add_argument("--hparam_PROTO_cov_MovingAvg_eta", type=float, default=0.1, help="Moving average value used for prototype augmentation.")
    parser.add_argument("--hparam_PROTO_cov_Balance_beta", type=float, default=1.0, help="Old and new balance value used for prototype augmentation.")
    # ----------
    parser.add_argument("--PROTO_cov_sketching", action="store_true", help="Using projection to reduce the covariance size.")
    parser.add_argument("--PROTO_cov_sketchingRatio", type=int, default=1, help="Projection ratio to reduce the covariance size. \
    Play with n_size_in/2, n_size_in/4, n_size_in/8, n_size_in/16, n_size_in/32, n_size_in/64 and even n_size_in/128")
    # ----------
    parser.add_argument("--hparam_MOV_beta", default=0.96, type=float, help="Hyperparameter - beta, used for exponentially moving updating of the model.")
    # -------------------------------------------------
    parser.add_argument("--hparam_LOSS_mannual_setting", action="store_true", help="Manual set the loss term hyparameters.")
    parser.add_argument("--hparam_LOSS_lambda_c", default=1.0, type=float, help="Hyperparameter - lambda_c, used for balancing different loss terms.")
    parser.add_argument("--hparam_LOSS_lambda_d", default=1.0, type=float, help="Hyperparameter - lambda_d, used for balancing different loss terms.")
    parser.add_argument("--hparam_LOSS_lambda_t", default=1.0, type=float, help="Hyperparameter - lambda_t, used for balancing different loss terms.")
    # -------------------------------------------------
    parser.add_argument("--Check", action="store_true", help="For debugging.")
    parser.add_argument("--Skip_training", action="store_true", help="The model is not trained and for evaluation or debugging only")
    parser.add_argument("--Regular_checkpoint", action="store_true", help="For regular storing model parameters.")
    parser.add_argument("--Regular_checkpoint_freq", default=100, type=int, help="Step frequency for regular storing model parameters.")
    # -------------------------------------------------

    args, left_argv = parser.parse_known_args()

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    if args.Data_Augmentation:
        hparams["Data_Augmentation"] = args.Data_Augmentation

    hparams['lambda_c'] = args.hparam_LOSS_lambda_c
    hparams['lambda_d'] = args.hparam_LOSS_lambda_d
    hparams['lambda_t'] = args.hparam_LOSS_lambda_t
    
    if args.TRIPLET_w_cross_entropy:
        hparams["w_cross_entropy"] = True
        hparams["margin"] = args.margin
    if args.No_TRIPLET:
        hparams["w_triplet"] = False
    if args.No_Proto:
        hparams["w_proto"] = False
    
    if args.CE_loss_no_exemplar:
        hparams["CE_loss_no_exemplar"] = True
    if args.CE_loss_only_new_cls:
        hparams["CE_loss_only_new_cls"] = True
    if args.DIST_loss_tar_normalized_old_and_new:
        hparams["DIST_loss_tar_normalized_old_and_new"] = True
    if args.DIST_loss_only_exemplar:
        hparams["DIST_loss_only_exemplar"] = True
    if args.DIST_loss_only_new_data:
        hparams["DIST_loss_only_new_data"] = True
    if args.DIST_loss_rated:
        hparams["DIST_loss_rated"] = True
        hparams["DIST_loss_ratio_for_exemplar"] = args.DIST_loss_ratio_for_exemplar
    if args.DIST_loss_feature:
        hparams["DIST_loss_feature"] = args.DIST_loss_feature

    if "MOV" in args.algorithm:
        hparams["beta"] = args.hparam_MOV_beta
    
    if "PROTO" in args.algorithm:
        hparams['sigma'] = args.hparam_PROTO_sigma
        hparams['PROTO_mean_Balance_beta'] = args.hparam_PROTO_mean_Balance_beta
        hparams['PROTO_mean_MovingAvg_eta'] = args.hparam_PROTO_mean_MovingAvg_eta
    if args.PROTO_using_delta:
        hparams['using_delta'] = True
    if args.PROTO_augmentation:
        hparams["PROTO_augmentation"] = True
    if args.PROTO_semantic_shifting:
        hparams["PROTO_semantic_shifting"] = True
    if args.PROTO_augmentation_w_COV:
        hparams["PROTO_augmentation_w_COV"] = True
        hparams["PROTO_cov_Shrinkage_alpha"]= args.hparam_PROTO_cov_Shrinkage_alpha
        hparams["PROTO_cov_MovingAvg_eta"]= args.hparam_PROTO_cov_MovingAvg_eta
        hparams["PROTO_cov_Balance_beta"]= args.hparam_PROTO_cov_Balance_beta
    
    if args.PROTO_cov_sketching:
        hparams["PROTO_cov_sketching"] =True
        hparams["PROTO_cov_sketchingRatio"] = args.PROTO_cov_sketchingRatio
    
    if args.hparam_LOSS_mannual_setting:
        hparams["LOSS_mannual_setting"] = True
    if args.DIST_loss_feature_CSCCT:
        hparams["DIST_loss_feature_CSCCT"] = True
        hparams["DIST_loss_feature_CSC_offset"] = args.DIST_loss_feature_CSC_offset

    hparams['normalised_feature_offset'] = args.TRIPLET_feature_offset
    
    if args.optimizer == "adam":
        hparams["optimizer"] = "adam"
    elif args.optimizer == "sgd":
        hparams["optimizer"] = "sgd"
    if args.learning_rate:
        hparams["lr"] = args.learning_rate
    if args.batch_size:
        hparams["batch_size"] = args.batch_size

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    timestamp = misc.timestamp()
    # args.unique_name = f"{timestamp}_{args.name}"
    args.unique_name = f"{args.name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / Path("train_output") / args.dataset / "trial_seed_{0}".format(args.trial_seed)
    # args.out_root = args.work_dir / Path("train_output") / args.dataset
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True)

    writer = get_writer(args.out_root / "runs" / args.unique_name)
    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))

    # Different to DomainBed, we support CUDA only.
    assert torch.cuda.is_available(), "CUDA is not available"

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))

    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)

    if args.show:
        exit()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    # Dummy datasets for logging information.
    # Real dataset will be re-assigned in train function.
    # test_envs only decide transforms; simply set to zero.
    dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)

    # print dataset information
    print('Dummy datasets for logging information.')
    print('Real dataset will be re-assigned in train function.')
    logger.nofmt("Dataset:")
    if 'INC' in args.dataset and args.current_session > 0:
        logger.nofmt(f"\t[{args.dataset}] #envs={(len(dataset)-1)}, #classes={dataset.num_classes}")
    else:
        logger.nofmt(f"\t[{args.dataset}] #envs={len(dataset)}, #classes={dataset.num_classes}")
    for i, env_property in enumerate(dataset.environments):
        logger.nofmt(f"\tenv{i}: {env_property} (#{len(dataset[i])})")
    logger.nofmt("")
    
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    org_n_steps = n_steps
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    if not args.test_envs:
        if 'INC' in args.dataset and args.current_session > 0 and args.num_of_exemplar > 0:
            args.test_envs = [[te] for te in range(len(dataset)-1)]
        else:
            args.test_envs = [[te] for te in range(len(dataset))]
    logger.info(f"Target test envs = {args.test_envs}")

    ###########################################################################
    # Run
    ###########################################################################
    all_records = []
    results = collections.defaultdict(list)

    for test_env in args.test_envs:
        res, records = train(test_env, args=args, hparams=hparams, n_steps=n_steps, checkpoint_freq=checkpoint_freq, logger=logger, writer=writer)
        all_records.append(records)
        for k, v in res.items():
            results[k].append(v)

    # log summary table
    logger.info("=== Summary ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("Unique name: %s" % args.unique_name)
    logger.info("Out path: %s" % args.out_dir)
    logger.info("Algorithm: %s" % args.algorithm)
    logger.info("Dataset: %s" % args.dataset)

    table = PrettyTable(["Selection"] + dataset.environments + ["Avg."])
    for key, row in results.items():
        row.append(np.mean(row))
        row = [f"{acc:.3%}" for acc in row]
        table.add_row([key] + row)
    logger.nofmt(table)


if __name__ == "__main__":
    main()
