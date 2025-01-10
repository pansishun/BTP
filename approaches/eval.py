from modulefinder import IMPORT_NAME
from approaches import before_train, after_train
from tqdm.auto import tqdm
import torch.nn as nn
import os
import shutil
import torch
import json
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
from transformers import get_scheduler
from utils import utils, baseline
import faiss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


logger = logging.getLogger(__name__)


class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def eval(self, model, train_loaders, test_loaders, replay_loader, accelerator, test_transformer):

        model = accelerator.prepare(model)
        model.eval()

        results = {}
        train_hidden = {}
        train_labels = {}
        train_logits = {}
        train_logits_mlp = {}
        model.eval()

        for eval_t in tqdm(range(self.args.task + 1)):

            results[eval_t] = {
                'predictions': [],  # [N x data], prediction of N task mask
                'references': [],  # [data]
                'hidden': [],  # [N x data]
                'logits': [],  # [N x data]
                'softmax_prob': [],  # [N x data]
                'total_num': 0,
                'adjusted_probabilities': []
            }
            train_hidden[eval_t] = []
            train_labels[eval_t] = []
            train_logits[eval_t] = []
            train_logits_mlp[eval_t] = []
            test_loader, train_loader = accelerator.prepare(test_loaders[eval_t], train_loaders[eval_t])
            load_path = os.path.join(self.args.output_dir, 'per_task_mlp.pt')
            per_task_mlp = torch.load(load_path)
            self.args.per_task_mlp = per_task_mlp

            for task_mask in range(self.args.task + 1):

                hidden_list = []
                prediction_list = []
                logits_list = []
                softmax_list = []

                for _, batch in enumerate(test_loader):
                    with torch.no_grad():
                        # Forward pass through the feature extractor and classifier
                        features, _ = model.forward_features(batch[0].cuda(), task_mask, s=self.args.smax)
                        output = model.forward_classifier(features, task_mask)
                        output = output[:, task_mask * self.args.class_num: (task_mask + 1) * self.args.class_num]

                        # Calculate softmax probabilities for model predictions
                        model_probabilities = torch.softmax(output, dim=1)

                        hidden_list += (features).cpu().numpy().tolist()
                        softmax_list += model_probabilities.cpu().numpy().tolist()
                        logits_list += output.cpu().numpy().tolist()

                        if task_mask == 0:
                            results[eval_t]['total_num'] += batch[0].shape[0]
                            results[eval_t]['references'] += batch[1].cpu().numpy().tolist()

                results[eval_t]['hidden'].append(hidden_list)
                results[eval_t]['predictions'].append(prediction_list)
                results[eval_t]['logits'].append(logits_list)

            for _, batch in enumerate(train_loader):
                with torch.no_grad():
                    features, _ = model.forward_features(batch[0], eval_t, s=self.args.smax)
                    output = model.forward_classifier(features, eval_t)
                    output = output[:, eval_t * self.args.class_num: (eval_t + 1) * self.args.class_num]
                    train_logits[eval_t] += output.cpu().numpy().tolist()
                    train_hidden[eval_t] += (features).cpu().numpy().tolist()
                    train_labels[eval_t] += (batch[1] - self.args.class_num * eval_t).cpu().numpy().tolist()


        ## train data
        self.args.train_logits = train_logits
        self.args.train_logits_mlp = train_logits_mlp
        self.args.train_labels = train_labels
        self.args.train_hidden = train_hidden
        self.args.model = model
        self.args.test_loaders = test_loaders

        ## maha feat
        self.args.feat_mean_list, self.args.precision_list = utils.load_maha(self.args, train_hidden)

        baseline.baseline(self.args, results)