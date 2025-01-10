import numpy as np
import torch
import json
import torch.nn.functional as F
from utils import utils
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn


@torch.no_grad()
def mds(args, test_logits, test_hidden, loader, task_mask):
    test_samples = torch.Tensor(test_hidden)
    score_in = utils.maha_score(args, test_samples, args.precision_list, args.feat_mean_list, task_mask)
    return score_in


@torch.no_grad()
def mls(args, test_logits, test_hidden, loader, task_mask):
    logits = test_logits
    return torch.tensor(logits).max(-1)[0]


def calculate_mask(self, w):
    contrib = self.mean_act[None, :] * w.data.squeeze().cpu().numpy()
    self.thresh = np.percentile(contrib, self.p)
    mask = torch.Tensor((contrib > self.thresh)).cuda()
    self.masked_w = w * mask


def BTP(args, test_logits, test_hidden, task_mask, mlp):
    test_hidden_tensor = torch.tensor(test_hidden, dtype=torch.float32)
    batch_size = 256
    data_loader = DataLoader(test_hidden_tensor, batch_size=batch_size, shuffle=False)
    mlp = mlp.cuda().eval()
    id_probabilities_list = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.cuda()
            mlp_outputs = mlp(batch)
            id_probabilities = 1 - torch.sigmoid(0.5 * mlp_outputs[:, 0])
            id_probabilities_list.append(id_probabilities.cpu())
    id_probabilities = torch.cat(id_probabilities_list)
    logit_score = torch.tensor(test_logits)
    energy = torch.logsumexp(logit_score, dim=1)

    test_samples = torch.Tensor(test_hidden)
    maha_distance = utils.maha_score(args, test_samples, args.precision_list, args.feat_mean_list, task_mask)

    return maha_distance, energy, id_probabilities


def baseline(args, results):
    metric = {}
    ood_label = {}
    maha_distance_tasks = {}
    energy_tasks = {}
    sum_ = 0

    for eval_t in range(args.task + 1):

        metric[eval_t] = {}
        ood_label[eval_t] = {}
        maha_distance_tasks[eval_t] = {}
        energy_tasks[eval_t] = {}

        logits = np.transpose(results[eval_t]['logits'], (1, 0, 2))  # (task_mask, sample, logit)
        softmax = torch.softmax(torch.from_numpy(logits / 1.0), dim=-1)
        maha_distance, energy, id_probabilities = torch.zeros(logits.shape[:2]), torch.zeros(logits.shape[:2]), torch.zeros(logits.shape[:2])

        for task_mask in range(args.task + 1):
            test_logits = results[eval_t]['logits'][task_mask]
            test_hidden = results[eval_t]['hidden'][task_mask]
            maha_distance[:,task_mask], energy[:,task_mask], id_probabilities[:,task_mask] = BTP(args, test_logits, test_hidden, task_mask, args.per_task_mlp[task_mask])

        energy_probabilities = torch.softmax(energy, dim=1)
        task_probabilities = maha_distance * energy_probabilities * id_probabilities

        if args.task != 0:
            prediction = (softmax.contiguous() * task_probabilities.unsqueeze(dim=-1)).reshape(task_probabilities.shape[0],-1).max(-1)[1].cpu().numpy().tolist()
        else:
            prediction = softmax.max(-1)[1].cpu().numpy().tolist()

        metric[eval_t]['acc'] = utils.acc(prediction, results[eval_t]['references'])

        sum_ += metric[eval_t]['acc']

    print("baseline: ", baseline)
    print(metric)
    metric['average'] = sum_ / (args.task + 1) * 100
    print(f"Acc: {metric['average']:.2f}")

    import os

    with open(os.path.join(args.output_dir, f'{baseline}_results'), 'a') as f:
        f.write(json.dumps(metric) + '\n')

    for eval_t in range(args.task + 1):
        utils.write_result_eval(metric[eval_t]['acc'], eval_t, args, metric)