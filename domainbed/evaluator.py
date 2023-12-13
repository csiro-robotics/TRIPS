import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def accuracy_from_loader(algorithm, loader, weights, debug=False):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    if 'DIST' in algorithm.name:
        algorithm.eval_mode()
    else:
        algorithm.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)

        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    if 'DIST' in algorithm.name:
        algorithm.train_mode()
    else:
        algorithm.train()

    acc = correct / total
    loss = losssum / total

    return acc, loss

def cls_wise_accuracy_from_loader(algorithm, loader, weights, debug=False, num_old_cls=0, num_new_cls=0):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0
    cls_wise_data_num = []
    cls_wise_correct_num = []
    cls_wise_acc = []
    num_total_cls = num_old_cls + num_new_cls
    old_cls_acc = 0
    new_cls_acc = 0
    avg_cls_acc = 0
    for i in range(num_total_cls):
        cls_wise_data_num.append(0)
        cls_wise_correct_num.append(0)
        cls_wise_acc.append(0)

    if 'DIST' in algorithm.name:
        algorithm.eval_mode()
    else:
        algorithm.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)

        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        for i in range(len(y)):
            prediction = logits[i].argmax()
            if y[i] == prediction:
                cls_wise_correct_num[y[i]] = cls_wise_correct_num[y[i]] + 1
            cls_wise_data_num[y[i]] = cls_wise_data_num[y[i]] + 1

        if debug:
            break

    if 'DIST' in algorithm.name:
        algorithm.train_mode()
    else:
        algorithm.train()

    acc = correct / total
    loss = losssum / total

    for i in range(len(cls_wise_data_num)):
        if cls_wise_data_num[i] > 0:
            cls_wise_acc[i] = cls_wise_correct_num[i] / cls_wise_data_num[i]
        else:
            cls_wise_acc[i] = 0.0
        # print('cls_wise_accuracy_from_loader | cls: {0} | cls_wise_correct_num: {1}, cls_wise_data_num: {2}, cls_wise_acc: {3}'.format(i, cls_wise_correct_num[i], cls_wise_data_num[i], cls_wise_acc[i]))
        if i < num_old_cls:
            old_cls_acc = old_cls_acc + cls_wise_acc[i]
        else:
            new_cls_acc = new_cls_acc + cls_wise_acc[i]
        avg_cls_acc = avg_cls_acc + cls_wise_acc[i]

    if num_old_cls != 0:
        old_cls_acc = old_cls_acc / num_old_cls
    else:
        old_cls_acc = 0.0
    avg_cls_acc = avg_cls_acc / num_total_cls
    new_cls_acc = new_cls_acc / num_new_cls
    if old_cls_acc + new_cls_acc > 0.0:
        harmonic_acc = (2 * old_cls_acc * new_cls_acc) / (old_cls_acc + new_cls_acc)
    else:
        harmonic_acc = 0.0
    # print('cls_wise_accuracy_from_loader | old_cls_acc: {0}, new_cls_acc: {1}, harmonic_acc: {2}, avg_cls_acc: {3}'.format(old_cls_acc, new_cls_acc, harmonic_acc, avg_cls_acc))

    return acc, loss, harmonic_acc, avg_cls_acc, cls_wise_acc, old_cls_acc, new_cls_acc

def accuracy(algorithm, loader_kwargs, weights, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)

    # return accuracy_from_loader(algorithm, loader, weights, **kwargs)
    return cls_wise_accuracy_from_loader(algorithm, loader, weights, **kwargs)


class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None, num_old_cls=0, num_new_cls=0, environments=None):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug
        self.environments = environments
        if num_new_cls == 0:
            raise ValueError('Evaluator | Something wrong with the num_of_cls.')
        self.num_old_cls = num_old_cls
        self.num_new_cls = num_new_cls

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)

        assert n_test_envs == 1
        summaries = collections.defaultdict(float)

        # for domain-wise accuracy - for each domain: (total_num_of_correct_prediction_for_all_classes/total_num_of_test_images_for_all_classes)
        summaries["test_in_domain_avg"] = 0.0
        summaries["test_out_domain_avg"] = 0.0
        summaries["train_in_domain_avg"] = 0.0
        summaries["train_out_domain_avg"] = 0.0

        # for domain-wise class-wise accuracy - for each domain: (sum_and_average(num_of_correct_prediction_per_class/num_of_test_images_per_class))
        summaries["test_in_domain_cls_avg"] = 0.0
        summaries["test_out_domain_cls_avg"] = 0.0
        summaries["train_in_domain_cls_avg"] = 0.0
        summaries["train_out_domain_cls_avg"] = 0.0

        # for domain-wise harmonic accuracy
        summaries["test_in_harmonic"] = 0.0
        summaries["test_out_harmonic"] = 0.0
        summaries["train_in_harmonic"] = 0.0
        summaries["train_out_harmonic"] = 0.0

        accuracies = {}
        losses = {}
        cls_wise_accuracies = {}
        harmonic_accuracies = {}
        avg_cls_accuracies = {}
        old_cls_accuracies = {}
        new_cls_accuracies = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])
            # print('evaluate | env_name: {0}, inout: {1}, env_num: {2}'.format(env_name, inout, env_num))

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, loss, harmonic_acc, avg_cls_acc, cls_wise_acc, old_cls_acc, new_cls_acc = accuracy(algorithm, loader_kwargs, weights, debug=self.debug, num_old_cls=self.num_old_cls, num_new_cls=self.num_new_cls)
            # acc, loss = accuracy(algorithm, loader_kwargs, weights, debug=self.debug)
            accuracies[name + "_domain_avg_acc"] = acc
            losses[name] = loss
            harmonic_accuracies[name + "_harmonic_acc"] = harmonic_acc
            avg_cls_accuracies[name + "_domain_cls_avg_acc"] = avg_cls_acc
            cls_wise_accuracies[name + "_cls_wise_acc"] = cls_wise_acc
            old_cls_accuracies[name + "_old_cls_avg_acc"] = old_cls_acc
            new_cls_accuracies[name + "_new_cls_avg_acc"] = new_cls_acc

            if env_num in self.train_envs:
                summaries["train_" + inout + "_domain_avg"] += acc / n_train_envs
                summaries["train_" + inout + "_domain_cls_avg"] += avg_cls_acc / n_train_envs
                summaries["train_" + inout + "_harmonic"] += harmonic_acc / n_train_envs
                if inout == "out":
                    summaries["train_" + inout + "_loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout + "_domain_avg"] += acc / n_test_envs
                summaries["test_" + inout + "_domain_cls_avg"] += avg_cls_acc / n_test_envs
                summaries["test_" + inout + "_harmonic"] += harmonic_acc / n_test_envs

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries, harmonic_accuracies, avg_cls_accuracies, cls_wise_accuracies, old_cls_accuracies, new_cls_accuracies
            # return accuracies, summaries

    def source_model_validation(self, algorithm):
        n_test_envs = len(self.test_envs)

        assert n_test_envs == 1

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            if name != 'env0_in':
                continue
            print('name: {0}'.format(name))

            if isinstance(loader_kwargs, dict):
                loader = FastDataLoader(**loader_kwargs)
            elif isinstance(loader_kwargs, FastDataLoader):
                loader = loader_kwargs
            else:
                raise ValueError(loader_kwargs)

            for i, batch in enumerate(loader):
                x = batch["x"].to(device)
                y = batch["y"].to(device)

            with torch.no_grad():
                logits = algorithm.source_predict(x)
            # print('logits size: {0}'.format(logits.size()))
            # print('logits[:3, :]: {0}'.format(logits[:3, :]))
