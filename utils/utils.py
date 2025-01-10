import os
from copy import deepcopy
import numpy as np
import torch
import sklearn.covariance
from sklearn import metrics
import json
import faiss
import torch.nn.functional as F
from tqdm.auto import tqdm


def normalize(v):
    v = np.array(v)
    norm = np.linalg.norm(v, axis=-1).reshape(-1, 1)
    return v / (norm + 1e-9)


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def write_result(results, eval_t, args):
    progressive_main_path = os.path.join(
        args.output_dir + '/../', 'progressive_main_' + str(args.seed)
    )
    progressive_til_path = os.path.join(
        args.output_dir + '/../', 'progressive_til_' + str(args.seed)
    )
    progressive_tp_path = os.path.join(
        args.output_dir + '/../', 'progressive_tp_' + str(args.seed)
    )

    if os.path.exists(progressive_main_path):
        eval_main = np.loadtxt(progressive_main_path)
    else:
        eval_main = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
    if os.path.exists(progressive_til_path):
        eval_til = np.loadtxt(progressive_til_path)
    else:
        eval_til = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
    if os.path.exists(progressive_tp_path):
        eval_tp = np.loadtxt(progressive_tp_path)
    else:
        eval_tp = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)

    try:
        eval_main[args.task][eval_t] = results['accuracy']
        eval_til[args.task][eval_t] = results['accuracy']
        eval_tp[args.task][eval_t] = results['accuracy']
    except:
        eval_main[args.task][eval_t] = results['cil_accuracy']
        eval_til[args.task][eval_t] = results['til_accuracy']
        eval_tp[args.task][eval_t] = results['TP_accuracy']

    np.savetxt(progressive_main_path, eval_main, '%.4f', delimiter='\t')
    np.savetxt(progressive_til_path, eval_til, '%.4f', delimiter='\t')
    np.savetxt(progressive_tp_path, eval_tp, '%.4f', delimiter='\t')

    if args.task == args.ntasks - 1:
        final_main = os.path.join(args.output_dir + '/../', 'final_main_' + str(args.seed))
        forward_main = os.path.join(args.output_dir + '/../', 'forward_main_' + str(args.seed))

        final_til = os.path.join(args.output_dir + '/../', 'final_til_' + str(args.seed))
        forward_til = os.path.join(args.output_dir + '/../', 'forward_til_' + str(args.seed))

        final_tp = os.path.join(args.output_dir + '/../', 'final_tp_' + str(args.seed))
        forward_tp = os.path.join(args.output_dir + '/../', 'forward_tp_' + str(args.seed))

        with open(final_main, 'w') as final_main_file, open(forward_main, 'w') as forward_main_file:
            for j in range(eval_main.shape[1]):
                final_main_file.writelines(str(eval_main[-1][j]) + '\n')
                forward_main_file.writelines(str(eval_main[j][j]) + '\n')

        with open(final_til, 'w') as final_til_file, open(forward_til, 'w') as forward_til_file:
            for j in range(eval_til.shape[1]):
                final_til_file.writelines(str(eval_til[-1][j]) + '\n')
                forward_til_file.writelines(str(eval_til[j][j]) + '\n')

        with open(final_tp, 'w') as final_tp_file, open(forward_tp, 'w') as forward_tp_file:
            for j in range(eval_tp.shape[1]):
                final_tp_file.writelines(str(eval_tp[-1][j]) + '\n')
                forward_tp_file.writelines(str(eval_tp[j][j]) + '\n')


def write_result_eval(results, eval_t, args,metric):
    progressive_main_path = os.path.join(
        args.output_dir + '/../', 'progressive_main_' + str(args.seed)
    )

    if os.path.exists(progressive_main_path):
        eval_main = np.loadtxt(progressive_main_path)
    else:
        eval_main = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)

    eval_main[args.task][eval_t] = results
    if args.task == eval_t:
        AIA = np.mean([np.mean(eval_main[i][:i + 1]) for i in range(args.task + 1)]) * 100
        print(f"AIA: {AIA:.2f}")

    np.savetxt(progressive_main_path, eval_main, '%.4f', delimiter='\t')

    if args.task == args.ntasks - 1:
        final_main = os.path.join(args.output_dir + '/../', 'final_main_' + str(args.seed))
        forward_main = os.path.join(args.output_dir + '/../', 'forward_main_' + str(args.seed))
        final_AIA_ACC_path = os.path.join(args.output_dir + '/../', 'final_AIA_ACC_' + str(args.seed))

        with open(final_main, 'w') as final_main_file, open(forward_main, 'w') as forward_main_file, open(final_AIA_ACC_path, 'w') as final_AIA_ACC_file:
            for j in range(eval_main.shape[1]):
                final_main_file.writelines(str(eval_main[-1][j]) + '\n')
                forward_main_file.writelines(str(eval_main[j][j]) + '\n')

            if args.task == eval_t:
                # Save AIA and ACC as two lines
                final_AIA_ACC_file.write(f"AIA: {AIA:.2f}\n")
                final_AIA_ACC_file.write(f"ACC: {metric['average']:.2f}\n")


def prepare_sequence_eval(args):
    with open(os.path.join('./sequence', args.sequence_file), 'r') as f:
        data = f.readlines()[args.idrandom]
        data = data.split()

    args.all_tasks = data
    args.ntasks = len(data)

    ckpt = args.base_dir + '/seq' + str(args.class_order) + "/seed" + str(args.seed) + \
           '/' + str(args.baseline) + '/' + str(data[args.task]) + '/model'
    args.output_dir = args.base_dir + "/seq" + \
                      str(args.class_order) + "/seed" + str(args.seed) + "/" + str(args.baseline) + '/' + str(
        data[args.task])

    args.prev_output = None
    args.model_name_or_path = ckpt

    print('output_dir: ', args.output_dir)
    print('prev_output: ', args.prev_output)
    print('args.model_name_or_path: ', args.model_name_or_path)

    return args


def prepare_sequence_train(args):
    with open(os.path.join('./sequence', args.sequence_file), 'r') as f:
        data = f.readlines()[args.idrandom]
        data = data.split()

    args.task_name = data[args.task]
    args.all_tasks = data
    args.ntasks = len(data)
    args.output_dir = args.base_dir + '/seq' + \
                      str(args.class_order) + "/seed" + str(args.seed) + '/' + str(args.baseline) + '/' + str(
        data[args.task])
    ckpt = args.base_dir + '/seq' + str(args.class_order) + "/seed" + str(args.seed) + \
           '/' + str(args.baseline) + '/' + str(data[args.task - 1]) + '/model'

    if args.task > 0:
        args.prev_output = args.base_dir + "/seq" + \
                           str(args.class_order) + "/seed" + str(args.seed) + "/" + str(args.baseline) + '/' + str(
            data[args.task - 1])
        args.model_name_or_path = ckpt
    else:
        args.prev_output = None
        args.model_name_or_path = None

    print('output_dir: ', args.output_dir)
    print('prev_output: ', args.prev_output)
    print('args.model_name_or_path: ', args.model_name_or_path)

    return args


def load_in661_pretrain(args, target_model):
    """
        target_model: the model we want to replace the parameters (most likely un-trained)
    """
    if os.path.isfile(f'{args.base_dir}/deit_in661/best_checkpoint.pth'):
        checkpoint = torch.load(f'{args.base_dir}/deit_in661/best_checkpoint.pth', map_location='cpu')
    else:
        raise NotImplementedError("Cannot find pre-trained model")
    target = target_model.state_dict()
    pretrain = checkpoint['model']
    transfer = {k: v for k, v in pretrain.items() if k in target and 'head' not in k}
    target.update(transfer)
    target_model.load_state_dict(target)


def lookfor_model(args):
    ## load visual encoder ##
    if 'deit_small_patch16' in args.visual_encoder:
        from networks.vit_hat import deit_small_patch16_224
        model = deit_small_patch16_224(pretrained=False, num_classes=args.class_num *
                                                                     args.ntasks, latent=args.latent, args=args,
                                       hat='hat' in args.baseline)
    elif 'vit_small_patch16' in args.visual_encoder:
        from networks.vit_hat import vit_small_patch16_224
        model = vit_small_patch16_224(pretrained=False, num_classes=args.class_num *
                                                                    args.ntasks, latent=args.latent, args=args,
                                      hat='hat' in args.baseline)
    elif 'vit_base_patch16' in args.visual_encoder:
        from networks.vit_hat import vit_base_patch16_224
        model = vit_base_patch16_224(pretrained=False, num_classes=args.class_num *
                                                                   args.ntasks, latent=args.latent, args=args,
                                     hat='hat' in args.baseline)
    elif 'vit_tiny_patch16' in args.visual_encoder:
        from networks.vit_hat import vit_tiny_patch16_224
        model = vit_tiny_patch16_224(pretrained=False, num_classes=args.class_num *
                                                                   args.ntasks, latent=args.latent, args=args,
                                     hat='hat' in args.baseline)
    elif 'deit_tiny_patch16' in args.visual_encoder:
        from networks.vit_hat import deit_tiny_patch16_224
        model = deit_tiny_patch16_224(pretrained=False, num_classes=args.class_num *
                                                                    args.ntasks, latent=args.latent, args=args,
                                      hat='hat' in args.baseline)
    else:
        raise NotImplementedError

    checkpoint = torch.load(f'{args.base_dir}/pretrained/{args.visual_encoder}.pth', map_location='cpu')
    target = model.state_dict()

    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']

    transfer = {k: v for k, v in checkpoint.items() if k in target and 'head' not in k}
    target.update(transfer)
    model.load_state_dict(target)

    ## load adapter or hat mask##
    if 'hat' in args.baseline:
        for _ in range(args.task):
            model.append_embeddings()

        if not args.training:  # inference for the t-th task
            model.append_embeddings()

        if args.task > 0 or not args.training:  # load the trained weights
            model.load_state_dict(torch.load(os.path.join(args.model_name_or_path), map_location='cpu'))

        if args.training:  # training for the t-th task
            model.append_embeddings()

    if 'derpp' in args.baseline:
        if args.task > 0:
            model.load_state_dict(torch.load(os.path.join(args.model_name_or_path), map_location='cpu'))
        args.teacher_model = deepcopy(model)  # used for get representation for replay sample

    return model


def auroc(predictions, references):
    fpr, tpr, _ = metrics.roc_curve(references, predictions, pos_label=1)
    return metrics.auc(fpr, tpr)


def acc(predictions, references):
    acc = metrics.accuracy_score(references, predictions)
    return acc


def aupr(predictions, references):
    ind_indicator = np.zeros_like(references)
    ind_indicator[references != -1] = 1

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(ind_indicator, predictions)

    aupr_in = metrics.auc(recall_in, precision_in)

    return aupr_in


def fpr95(predictions, references, tpr=0.95):
    gt = np.ones_like(references)
    gt[references == -1] = 0
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, predictions)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
    return fpr

@torch.no_grad()
def Mahainit(args, train_hidden):
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    feat_mean_list = {}
    precision_list = {}

    for train_t in tqdm(range(args.task + 1)):
        features = train_hidden[train_t]
        features_array = np.array([np.array(f).flatten() for f in features])
        feat_mean = np.mean(features_array, axis=0)
        feat_mean_list[train_t] = [feat_mean.tolist()]
        X = features_array - feat_mean
        group_lasso.fit(X)
        precision = group_lasso.precision_
        precision_list[train_t] = precision.tolist()

    return feat_mean_list, precision_list

def maha_score(args, test_sample, precision_list, feat_mean_list, task_mask):
    zero_f = test_sample - torch.Tensor(feat_mean_list[task_mask])
    term_gau = 20.0 / torch.mm(torch.mm(zero_f, torch.Tensor(precision_list[task_mask])),
                               zero_f.t()).diag()
    noise_gaussian_score = term_gau.view(-1, 1)
    score_in = noise_gaussian_score[:,0]
    return score_in


def load_maha(args, train_hidden):
    print("start mahainit...")
    feat_mean_list, precision_list = Mahainit(args, train_hidden)
    print("finish mahainit!!")
    return feat_mean_list, precision_list