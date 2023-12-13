import collections
import json
import time
import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.lib.prototype import load_prototype
from domainbed.lib.utils import write_to_txt
from domainbed.pre_trainer import pre_train
from domainbed.post_trainer import post_train

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")

    # pre-training
    if args.current_session > 0 and args.load_old_info == None:
        raise ValueError("Old session model needs to be loaded for incremental learning.")
        
    if args.load_old_info:
        old_parameters, old_prototype_dict, precision_matrix = pre_train(args, test_envs)
    else:
        old_parameters, old_prototype_dict, precision_matrix = None, None, None

    # main training procedure
    ret, records, eval_meta, algorithm, updated_iid_prototype_dict, updated_oracle_prototype_dict = main_training(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env, old_parameters, old_prototype_dict, precision_matrix)

    # post-training
    iid_prototype_dict = old_prototype_dict
    oracle_prototype_dict = old_prototype_dict
    if args.current_session > 0:
        iid_prototype_dict["cls_wise_avg_feature"] = updated_iid_prototype_dict["mean"]
        iid_prototype_dict["cls_wise_cov"] = updated_iid_prototype_dict["covariance"]
        oracle_prototype_dict["cls_wise_avg_feature"] = updated_oracle_prototype_dict["mean"]
        oracle_prototype_dict["cls_wise_cov"] = updated_oracle_prototype_dict["covariance"]
    post_train(test_envs, args, eval_meta, algorithm, args.current_session, iid_prototype_dict, oracle_prototype_dict, precision_matrix)

    return ret, records


def main_training(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None, old_parameters=None, old_prototype_dict=None, precision_matrix=None):
    iid_prototype_dict = {}
    oracle_prototype_dict = {}

    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    print('~~~~~~~~~ get dataset ~~~~~~~~~')
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    test_splits = []
    if hparams.indomain_test > 0.0:
        logger.info("!!! In-domain test mode On !!!")
        assert hparams["val_augment"] is False, (
            "indomain_test split the val set into val/test sets. "
            "Therefore, the val set should be not augmented."
        )
        val_splits = []
        for env_i, (out_split, _weights) in enumerate(out_splits):
            n = len(out_split) // 2
            seed = misc.seed_hash(args.trial_seed, env_i)
            val_split, test_split = split_dataset(out_split, n, seed=seed)
            val_splits.append((val_split, None))
            test_splits.append((test_split, None))
            logger.info(
                "env %d: out (#%d) -> val (#%d) / test (#%d)"
                % (env_i, len(out_split), len(val_split), len(test_split))
            )
        out_splits = val_splits

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)
    if args.num_of_exemplar > 0 and args.Mixup_old_exemplar and args.current_session > 0:
        print('trainer.py | before exemplar_mixup | batch_sizes: {0}'.format(batch_sizes))
        batch_sizes[-1] = batch_sizes[-1] / args.hparam_Mixup_old_exemplar_times * 2
        print('trainer.py | after exemplar_mixup | batch_sizes: {0}'.format(batch_sizes))

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")

    # -------------------------------------------------------------------------
    # setup train loaders
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS,
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]  # 128
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    num_domains = len(dataset) - len(test_envs)
    if args.algorithm == 'ERM_DIST' or args.algorithm == 'MSL_MOV_DIST':  # distillation
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, num_domains, hparams, args.current_session, args.num_of_exemplar, args.num_old_cls, args.hparam_temperature)
    elif args.algorithm == 'EWC':  
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, num_domains, hparams, args.current_session, old_parameters, precision_matrix, args.hparam_ewc_scale)
    elif args.algorithm == 'MAS':
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, num_domains, hparams, args.current_session, old_parameters, precision_matrix, args.hparam_mas_scale)
    elif 'TRIPLET_DIST_W_PROTO' == args.algorithm:
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, num_domains, hparams, args.current_session, args.num_of_exemplar, args.num_old_cls, args.hparam_temperature, old_prototype_dict, args.triplet_dist_type, args.Check)
    elif args.algorithm == 'ERM' or args.algorithm == 'CORAL':
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, num_domains, hparams, args.current_session, args.num_of_exemplar)
    else:  # MSL, MixStyle2
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, num_domains, hparams, args.current_session)

    algorithm.to(device)

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # load old model
    #######################################################
    if args.current_session > 0 and ('DIST' in args.algorithm) and (not args.load_old_info):
        raise RuntimeError('For distillation method there must be old model loaded.')

    if args.load_old_info:
        old_info_path = "{0}/checkpoints".format(args.load_old_info)
        old_model_path = os.path.join(args.out_root, old_info_path)
        algorithm.load_previous_model_param(old_model_path, test_envs, args.model_type)
    
    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(test_envs, eval_meta, n_envs, logger, args.evalmode, args.debug, target_env, args.num_old_cls, args.num_new_cls, dataset.environments)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"

    best_iid_acc = 0.0
    best_oracle_acc = 0.0
    best_iid_model_flag = False
    best_oracle_model_flag = False
    
    for step in range(n_steps):
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(train_minibatches_iterator)
        # batches: {data_key: [env0_tensor, ...], ...}
        batches = misc.merge_dictlist(batches_dictlist)

        # to device
        batches_2_device = {
            key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batches.items() if key != "img_id"
        }
        inputs = {**batches_2_device, "step": step, "img_id": batches["img_id"], "envs": dataset.environments}

        if 'DIST' in args.algorithm:
            algorithm.train_mode()
        else:
            algorithm.train()
        
        # *** updating the model ***
        # print('step: {0}'.format(step))
        step_vals = algorithm.update(**inputs)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            accuracies, summaries, harmonic_accuracies, avg_cls_accuracies, cls_wise_accuracies, old_cls_accuracies, new_cls_accuracies = evaluator.evaluate(algorithm)  # *** evaluating the model ***
            # accuracies, summaries = evaluator.evaluate(algorithm)  # *** evaluating the model ***
            results["eval_time"] = time.time() - eval_start_time

            # print('--- check source model output.)
            # evaluator.source_model_validation(algorithm)

            # results = (epochs, loss, step, step_time)
            # results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + sorted(harmonic_accuracies.keys()) + \
                           sorted(avg_cls_accuracies.keys()) + sorted(cls_wise_accuracies.keys()) + list(results.keys())

            # merge results
            results.update(summaries)
            results.update(accuracies)
            results.update(harmonic_accuracies)
            results.update(avg_cls_accuracies)
            results.update(cls_wise_accuracies)
            results.update(old_cls_accuracies)
            results.update(new_cls_accuracies)

            # print
            display_keys = ['step', 'epoch', 'loss', 'train_out_domain_cls_avg', 'test_out_domain_cls_avg', 'test_in_domain_cls_avg', 'test_in_harmonic']
            if display_keys != last_results_keys:
                logger.info(misc.to_row(display_keys, colwidth=25))
                last_results_keys = display_keys
            logger.info(misc.to_row([results[key] for key in display_keys], colwidth=25))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")  # add evaluation information to tensorboard
            writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")  # add evaluation information to tensorboard

            #######################################################
            # store related model
            #######################################################
            ckpt_dir = args.out_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            test_env_str = ",".join(map(str, test_envs))

            if results['train_out_domain_cls_avg'] > best_iid_acc:  # train_out
                filename = "TE{}_best_iid.pth".format(test_env_str)
                path = ckpt_dir / filename
                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)
                best_iid_acc = results['train_out_domain_cls_avg']  # train_out
                iid_prototype_dict["mean"] = algorithm.updated_mean
                iid_prototype_dict["covariance"] = algorithm.updated_cov
                print('* Find best iid model, step = {0}, best_iid_acc = {1}'.format(step, best_iid_acc))
                best_iid_model_flag = True
                txt_filename = "TE{}_best_iid_info.txt".format(test_env_str)
                write_to_txt(results, ckpt_dir, txt_filename)

            if results['test_out_domain_cls_avg'] > best_oracle_acc:  # test_out
                filename = "TE{}_best_oracle.pth".format(test_env_str)
                path = ckpt_dir / filename
                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)
                best_oracle_acc = results['test_out_domain_cls_avg']  # test_out
                oracle_prototype_dict["mean"] = algorithm.updated_mean
                oracle_prototype_dict["covariance"] = algorithm.updated_cov
                print('* Find best oracle model, step = {0}, best_oracle_acc = {1}'.format(step, best_oracle_acc))
                best_oracle_model_flag = True
                txt_filename = "TE{}_best_oracle_info.txt".format(test_env_str)
                write_to_txt(results, ckpt_dir, txt_filename)

            if args.model_save and step >= args.model_save:  # regularlly save the model
                # filename = "TE{}_{}.pth".format(test_env_str, step)
                filename = "TE{}_last_step.pth".format(test_env_str)
                if len(test_envs) > 1 and target_env is not None:
                    train_env_str = ",".join(map(str, train_envs))
                    filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
                path = ckpt_dir / filename

                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)
            
        if step % args.tb_freq == 0:
            # add step values only for tb log
            writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")  # add training information to tensorboard ('loss' shows total loss value)
    
    # check model storage
    print('--- best_iid_model_flag: {0}'.format(best_iid_model_flag))
    print('--- best_oracle_model_flag: {0}'.format(best_oracle_model_flag))
    if best_iid_model_flag == False:
        print('*** No best iid model found!')
        filename = "TE{}_best_iid.pth".format(test_env_str)
        path = ckpt_dir / filename
        save_dict = {
            "args": vars(args),
            "model_hparams": dict(hparams),
            "test_envs": test_envs,
            "model_dict": algorithm.cpu().state_dict(),
        }
        algorithm.cuda()
        if not args.debug:
            torch.save(save_dict, path)
        else:
            logger.debug("DEBUG Mode -> no save (org path: %s)" % path)    
    
    if best_oracle_model_flag == False:
        print('*** No best oracle model found!')
        filename = "TE{}_best_oracle.pth".format(test_env_str)
        path = ckpt_dir / filename
        save_dict = {
            "args": vars(args),
            "model_hparams": dict(hparams),
            "test_envs": test_envs,
            "model_dict": algorithm.cpu().state_dict(),
        }
        algorithm.cuda()
        if not args.debug:
            torch.save(save_dict, path)
        else:
            logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

    # find best
    logger.info("---")
    records = Q(records)
    """
    oracle_best = records.argmax("test_out")["test_in"]
    iid_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]
    """
    oracle_best = records.argmax("test_out_domain_cls_avg")["test_in_domain_cls_avg"]
    oracle_best_harmonic = records.argmax("test_out_domain_cls_avg")["test_in_harmonic"]
    iid_best = records.argmax("train_out_domain_cls_avg")["test_in_domain_cls_avg"]
    iid_best_harmonic = records.argmax("train_out_domain_cls_avg")["test_in_harmonic"]
    last = records[-1]["test_in_domain_cls_avg"]
    last_harmonic = records[-1]["test_in_harmonic"]

    if hparams.indomain_test:
        # if test set exist, use test set for indomain results
        in_key = "train_inTE"
    else:
        # in_key = "train_out"
        in_key = "train_out_domain_cls_avg"
        in_key_harmonic = "train_out_harmonic"

    iid_best_indomain = records.argmax("train_out_domain_cls_avg")[in_key]
    last_indomain = records[-1][in_key]
    iid_best_indomain_harmonic = records.argmax("train_out_domain_cls_avg")[in_key_harmonic]
    last_indomain_harmonic = records[-1][in_key_harmonic]


    ret = {
        "oracle": oracle_best,
        "iid": iid_best,
        "last": last,
        "last (inD)": last_indomain,
        "iid (inD)": iid_best_indomain,
        "oracle_harmonic": oracle_best_harmonic,
        "iid_harmonic": iid_best_harmonic,
        "last_harmonic": last_harmonic,
        "last_harmonic (inD)": last_indomain_harmonic,
        "iid_harmonic (inD)": iid_best_indomain_harmonic,
    }
   
    return ret, records, eval_meta, algorithm, iid_prototype_dict, oracle_prototype_dict