from modulefinder import IMPORT_NAME
from approaches import before_train, after_train
from tqdm.auto import tqdm
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch
import random
from torch import nn
import heapq
import copy
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import logging
import math
from torchvision import transforms
import torch.nn.functional as F
from utils.sgd_hat import HAT_reg, compensation, compensation_clamp
from utils.sgd_hat import SGD_hat as SGD
from utils import utils

logger = logging.getLogger(__name__)

EPSILON = 1e-8
class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def train(self, model, train_loader, test_loaders, replay_loader,replay_loader1,replay_loader_transformer):
        class BufferDataset(Dataset):
            def __init__(self, buffer_dict, transform=None):
                self.data = []
                self.transform = transform
                for label in buffer_dict:
                    self.data.extend(buffer_dict[label])

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                # 提取数据和标签
                x, y = self.data[idx]
                if self.transform:
                    x = self.transform(x)
                return x, y

        # load task level memory
        ###################################
        if self.args.task != 0:
            buffer_path = os.path.join(os.path.join(self.args.prev_output, f'buffer_dict.pt'))

            buffer_dict = torch.load(buffer_path)
            dataset = BufferDataset(buffer_dict,transform=replay_loader_transformer)

            batch_size = 64
            replay_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=0)

        if 'C10_5T' in self.args.sequence_file:
            model.ood_classifier = nn.Linear(384, 1)  # OOD classifier
        elif 'C100_' in self.args.sequence_file:
            hidden = 256
            model.ood_classifier = nn.Sequential(
                nn.Linear(384, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            ).cuda()
        else:
            hidden = 384
            model.ood_classifier = nn.Sequential(
                nn.Linear(384, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            ).cuda()

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        optimizer = SGD(
            [
                {'params': model.adapter_parameters(), 'weight_decay': 5e-4},
                {'params': model.ood_classifier.parameters()}  # 只训练 OOD 分类头的参数
            ],
            lr=self.args.learning_rate,
            momentum=0.9,
            nesterov=True
        )
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        model = model.cuda()
        if 'derpp' in self.args.baseline:
            self.args.teacher_model = self.args.teacher_model.cuda()
            for p in self.args.teacher_model.parameters():
                p.requires_grad = False

        if replay_loader is not None:
            replay_iterator = iter(replay_loader)

        before_train.prepare(self.args, model)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = {}".format(len(train_loader) * self.args.batch_size))
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}, checkpoint Model = {self.args.model_name_or_path}")
        logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Learning Rate = {self.args.learning_rate}")
        logger.info(
            f"  Seq ID = {self.args.idrandom}, Task id = {self.args.task}, Task Name = {self.args.task_name}, Num task = {self.args.ntasks}")

        progress_bar = tqdm(range(self.args.max_train_steps))
        completed_steps = 0
        starting_epoch = 0

        for epoch in range(starting_epoch, self.args.num_train_epochs):
            model.train()

            for step, batch in enumerate(train_loader):

                s = (self.args.smax - 1 / self.args.smax) * step / len(
                    train_loader) + 1 / self.args.smax

                if replay_loader is not None:
                    try:
                        replay_batch = next(replay_iterator)
                        batch[0] = torch.cat((batch[0], replay_batch[0]), dim=0)
                        batch[1] = torch.cat(
                            (batch[1], torch.ones_like(replay_batch[1]) * (self.args.task + 1) * self.args.class_num),
                            dim=0)
                    except:
                        replay_iterator = iter(replay_loader)
                        replay_batch = next(replay_iterator)
                        batch[0] = torch.cat((batch[0], replay_batch[0]), dim=0)
                        batch[1] = torch.cat(
                            (batch[1], torch.ones_like(replay_batch[1]) * (self.args.task + 1) * self.args.class_num),
                            dim=0)

                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()

                # Forward pass
                features, masks = model.forward_features(batch[0], self.args.task, s=s)
                outputs = model.forward_classifier(features, self.args.task)

                # ID Classification Loss
                id_mask = (batch[1] < (self.args.task + 1) * self.args.class_num)  # ID data mask
                id_loss = nn.functional.cross_entropy(outputs[id_mask], batch[1][id_mask])

                # OOD Classification Head (assuming model.ood_classifier is defined)
                ood_outputs = model.ood_classifier(features)  # OOD classification head output
                ood_labels = (batch[1] >= (self.args.task + 1) * self.args.class_num).long()  # OOD labels

                if self.args.task == 0:
                    id_ood_loss = torch.tensor(0)
                else:
                    ood_labels = ood_labels.float()
                    id_ood_loss = F.binary_cross_entropy_with_logits(ood_outputs[:,0], ood_labels)

                tau = 0.07

                features_normalized = nn.functional.normalize(features, p=2, dim=1)

                id_indices = torch.where(ood_labels == 0)[0]

                id_features = features_normalized[id_indices]

                similarity_matrix = torch.matmul(id_features, features_normalized.T)

                positive_mask = (ood_labels == 0).float().unsqueeze(0).repeat(len(id_indices), 1)
                for idx_i, idx_in_batch in enumerate(id_indices):
                    positive_mask[idx_i, idx_in_batch] = 0

                negative_mask = (ood_labels == 1).float().unsqueeze(0).repeat(len(id_indices), 1)

                exp_sim = torch.exp(similarity_matrix / tau)
                numerator = torch.sum(exp_sim * positive_mask, dim=1)
                denominator = torch.sum(exp_sim * (positive_mask + negative_mask), dim=1)

                nce_loss = -torch.log(numerator / denominator)
                nce_loss = nce_loss.mean() * 0.05

                loss = id_loss + id_ood_loss + nce_loss

                if 'hat' in self.args.baseline:
                    loss += HAT_reg(self.args, masks)
                elif 'derpp' in self.args.baseline and replay_loader is not None:
                    with torch.no_grad():
                        prev_feature = self.args.teacher_model.forward_features(replay_batch[0].cuda())
                    loss += nn.functional.mse_loss(features[-prev_feature.shape[0]:, ...], prev_feature)

                loss.backward()

                if step % self.args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:

                    if 'hat' in self.args.baseline:
                        compensation(model, self.args, thres_cosh=self.args.thres_cosh, s=s)
                        optimizer.step(hat=(self.args.task > 0))
                        compensation_clamp(model, thres_emb=6)

                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_description(
                        'Train Iter (Epoch=%3d, loss=%5.3f, id_loss=%5.3f , id_ood_loss=%5.3f, nce_loss=%5.3f)' % (
                        epoch, loss.item(), id_loss.item(), id_ood_loss.item(), nce_loss.item()))

            if completed_steps >= self.args.max_train_steps:
                break

        if replay_loader is not None:
            replay_loader.dataset.transform = test_loaders[0].dataset.transform
        replay_loader = self._construct_exemplar(model, train_loader, replay_loader, self.args.replay_buffer_size, test_loaders[0].dataset.transform)
        self.get_mlp(self.args, model, replay_loader)
        del model.ood_classifier
        after_train.compute(self.args, model)

    def eval_cil(self, model, test_loaders, eval_t):
        model.eval()
        dataloader = test_loaders[eval_t]
        label_list = []
        cil_prediction_list, til_prediction_list = [], []
        total_num = 0

        for _, batch in enumerate(dataloader):
            with torch.no_grad():
                features = model.forward_features(batch[0].cuda())
                logits = model.forward_classifier(features)
                cil_outputs = logits[..., : (self.args.task + 1) * self.args.class_num]
                til_outputs = logits[..., eval_t * self.args.class_num: (eval_t + 1) * self.args.class_num]
                _, cil_prediction = torch.max(torch.softmax(cil_outputs, dim=1), dim=1)
                _, til_prediction = torch.max(torch.softmax(til_outputs, dim=1), dim=1)
                til_prediction += eval_t * self.args.class_num

                references = batch[1]
                total_num += batch[0].shape[0]

                label_list += references.cpu().numpy().tolist()
                cil_prediction_list += cil_prediction.cpu().numpy().tolist()
                til_prediction_list += til_prediction.cpu().numpy().tolist()

        cil_accuracy = sum(
            [1 if label_list[i] == cil_prediction_list[i] else 0 for i in range(total_num)]
        ) / total_num

        til_accuracy = sum(
            [1 if label_list[i] == til_prediction_list[i] else 0 for i in range(total_num)]
        ) / total_num

        tp_accuracy = sum(
            [1 if cil_prediction_list[i] // self.args.class_num == eval_t else 0 for i in range(total_num)]
        ) / total_num

        results = {
            'til_accuracy': round(til_accuracy, 4),
            'cil_accuracy': round(cil_accuracy, 4),
            'TP_accuracy': round(tp_accuracy, 4)
        }
        return results

    def eval_hat(self, model, test_loaders, eval_t):

        model.eval()
        dataloader = test_loaders[eval_t]
        label_list = []
        prediction_list = []
        taskscore_list = []
        total_num = 0
        for task_mask in range(self.args.task + 1):
            total_num = 0
            task_pred = []
            task_confidence = []
            task_label = []
            for _, batch in enumerate(dataloader):
                with torch.no_grad():
                    features, _ = model.forward_features(batch[0].cuda(), task_mask, s=self.args.smax)
                    outputs = model.forward_classifier(features, task_mask)[
                              :, task_mask * self.args.class_num: (task_mask + 1) * self.args.class_num]
                    score, prediction = torch.max(torch.softmax(outputs, dim=1), dim=1)

                    predictions = prediction + task_mask * self.args.class_num
                    references = batch[1]

                    total_num += batch[0].shape[0]
                    task_confidence += score.cpu().numpy().tolist()
                    task_label += references.cpu().numpy().tolist()
                    task_pred += predictions.cpu().numpy().tolist()

            label_list = task_label
            prediction_list.append(task_pred)
            taskscore_list.append(np.array(task_confidence))

        task_pred = np.argmax(np.stack(taskscore_list, axis=0), axis=0)
        cil_pred = [prediction_list[task_pred[i]][i] for i in range(total_num)]
        til_pred = [prediction_list[eval_t][i] for i in range(total_num)]

        cil_accuracy = sum(
            [1 if label_list[i] == cil_pred[i] else 0 for i in range(total_num)]
        ) / total_num
        til_accuracy = sum(
            [1 if label_list[i] == til_pred[i] else 0 for i in range(total_num)]
        ) / total_num
        TP_accuracy = sum(
            [1 if task_pred[i] == eval_t else 0 for i in range(total_num)]
        ) / total_num

        results = {
            'til_accuracy': round(til_accuracy, 4),
            'cil_accuracy': round(cil_accuracy, 4),
            'TP_accuracy': round(TP_accuracy, 4)
        }
        return results

    def _construct_exemplar(self, model, dataloader, replay_loader, buf_num=2000, test_transform=None):
        model.eval()
        if self.args.task != 0:
            buffer_path = os.path.join(self.args.prev_output, 'buffer_dict.pt')
            # Load buffer_dict
            buffer_dict = torch.load(buffer_path)
        else:
            buffer_dict = {}

        buffer_per_task = buf_num // (self.args.task + 1)
        buffer_remainder = buf_num % (self.args.task + 1)

        features_all = []
        label_all = []
        dataset = dataloader.dataset
        dataset.transform = test_transform
        dataloader_no_aug = DataLoader(dataset, batch_size=32, shuffle=False)
        for _, batch in enumerate(dataloader_no_aug):
            inputs, labels = batch
            labels = labels.cpu()
            with torch.no_grad():
                # Get model features
                features, _ = model.forward_features(inputs.cuda(), self.args.task, s=self.args.smax)
                features_all.append(features.cpu())
                label_all.append(labels)

        dataset = dataloader.dataset
        if hasattr(dataset, 'dataset'):
            dataset.dataset.transform = None
            # dataset.dataset.transforms = None
        dataset.transform = None
        data_all_org_img = []

        def pil_collate_fn(batch):
            images, labels = zip(*batch)
            return list(images), list(labels)

        dataloader_img = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=pil_collate_fn)
        for _, batch in enumerate(dataloader_img):
            inputs, labels = batch
            data_all_org_img.extend(inputs)

        features_all = torch.cat(features_all, dim=0)
        label_all = torch.cat(label_all, dim=0)

        # Update old buffer_dict
        for task_idx in range(0, self.args.task):
            if task_idx >= (self.args.task + 1) - buffer_remainder:
                m = buffer_per_task + 1
            else:
                m = buffer_per_task
            buffer_dict[task_idx] = buffer_dict[task_idx][:m]

        if self.args.task >= (self.args.task + 1) - buffer_remainder:
            m = buffer_per_task + 1
        else:
            m = buffer_per_task

        if self.args.task == 0:
            m_now = buf_num
        else:
            m_now = buf_num // self.args.task

        # Normalize feature vectors
        vectors = features_all / (features_all.norm(dim=1, keepdim=True) + 1e-8)
        task_mean = vectors.mean(dim=0)

        # Calculate class capacities
        classes = torch.unique(label_all).tolist()
        number_of_classes = len(classes)
        per_class_capacity = m_now // number_of_classes
        remainder = m_now % number_of_classes

        # Initialize class capacity and selected counts
        class_capacity = {cls: per_class_capacity for cls in classes}
        for i in range(remainder):
            class_capacity[classes[i]] += 1

        N = vectors.shape[0]
        available_mask = torch.ones(N, dtype=torch.bool)
        exemplar_indices = []
        exemplar_vectors = []

        # Select samples considering class balance
        for k in range(1, m_now + 1):
            if not available_mask.any():
                break
            if exemplar_vectors:
                S = torch.stack(exemplar_vectors).sum(dim=0)
            else:
                S = torch.zeros_like(vectors[0])
            mu_p = (vectors[available_mask] + S) / k
            distances = (mu_p - task_mean).norm(dim=1)
            i = torch.argmin(distances)
            selected_idx = torch.nonzero(available_mask)[i].item()
            exemplar_indices.append(selected_idx)
            exemplar_vectors.append(vectors[selected_idx])
            available_mask[selected_idx] = False

        new_exemplars = []
        for idx in exemplar_indices:
            data_item = data_all_org_img[idx]
            target = label_all[idx].item()
            new_exemplars.append((data_item, target))

        buffer_dict[self.args.task] = new_exemplars[:m]

        if replay_loader is not None:
            # Update replay_loader
            replay_loader.dataset.data += [(ex[0], ex[1]) for ex in new_exemplars]

        save_path = os.path.join(self.args.output_dir, 'buffer_dict.pt')
        torch.save(buffer_dict, save_path)

        return replay_loader

    def get_mlp(self, args, model, replay_loader):
        model.eval()
        mlp = model.ood_classifier
        if self.args.task == 0:
            per_task_mlp = {0: mlp}
            save_path = os.path.join(args.output_dir, f'per_task_mlp.pt')
            torch.save(per_task_mlp, save_path)
            return per_task_mlp
        else:
            load_path = os.path.join(self.args.prev_output, f'per_task_mlp.pt')
            per_task_mlp = torch.load(load_path)
            per_task_mlp[self.args.task] = mlp

        for task_mask in range(self.args.task + 1):
            mlp = per_task_mlp[task_mask]
            mlp.eval()
            task_features = {}
            for step, batch in enumerate(replay_loader):
                batch[0] = batch[0].cuda()
                with torch.no_grad():
                    features, _ = model.forward_features(batch[0], task_mask, s=self.args.smax)
                    labels = batch[1].cpu().numpy() // self.args.class_num


                    for feature, label in zip(features.cpu(), labels):
                        if label not in task_features:
                            task_features[label] = []
                        task_features[label].append(feature)

            # MLP Model
            buffer_per_task = len(task_features[0])
            ood_features = []
            id_features = []
            for task_id in task_features:
                features = torch.stack(task_features[task_id])[:buffer_per_task]
                if task_id != task_mask:
                    ood_features.extend(features)
                else:
                    id_features.extend(features)

            id_features = torch.stack(id_features)

            if len(ood_features) != 0:
                ood_features = torch.stack(ood_features)
                id_labels = torch.zeros(id_features.size(0), dtype=torch.int64)
                ood_labels = torch.ones(ood_features.size(0), dtype=torch.int64)

                features = torch.cat((id_features, ood_features), dim=0)
                labels = torch.cat((id_labels, ood_labels), dim=0)
            else:
                features = id_features
                labels = torch.zeros(id_features.size(0), dtype=torch.int64)

            # DataLoader for combined dataset
            dataset = TensorDataset(features, labels)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            mlp = per_task_mlp[task_mask]

            num_epochs = 10
            lr_rate = self.args.learning_rate

            mlp.train()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.SGD(mlp.parameters(), lr=lr_rate, momentum=0.9)

            # Training loop
            for epoch in range(num_epochs):
                total_loss = 0
                correct = 0
                total = 0

                for data, labels in loader:
                    outputs = mlp(data.cuda())
                    loss = criterion(outputs[:,0], labels.cuda().float())
                    total_loss += loss.item() * data.size(0)

                    probabilities = torch.sigmoid(outputs[:,0])
                    predicted = (probabilities > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels.cuda()).sum().item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                avg_loss = total_loss / total
                train_accuracy = 100 * correct / total

                print(
                    f'task {task_mask}: Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%')

            # Store MLP for each class
            per_task_mlp[task_mask] = mlp

            save_path = os.path.join(args.output_dir, f'per_task_mlp.pt')
            torch.save(per_task_mlp, save_path)

        return per_task_mlp