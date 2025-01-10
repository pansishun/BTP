import numpy as np
import logging
import config
from utils import utils
import os
from accelerate.utils import set_seed
from dataloader.data import get_dataset
from torch.utils.data import DataLoader
from approaches.train import Appr
from approaches.noncl import Appr as Appr_noncl
import torch
from torch.utils.data import ConcatDataset

logger = logging.getLogger(__name__)

args = config.parse_args()
args = utils.prepare_sequence_train(args)
## set seed
if args.seed is not None:
    set_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if args.output_dir is not None:
    os.makedirs(args.output_dir, exist_ok=True)

dataset = get_dataset(args)
model = utils.lookfor_model(args)

if 'full' in args.baseline:
    train_loader = DataLoader(ConcatDataset([dataset[t]['train'] for t in range(args.task+1)]), batch_size=args.batch_size, shuffle=True, num_workers=0)
else:
    train_loader = DataLoader(dataset[args.task]['train'], batch_size=args.batch_size, shuffle=True, num_workers=0)

test_loaders = []

for eval_t in range(args.ntasks):
    test_dataset = dataset[eval_t]['test']
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loaders.append(test_dataloader)

replay_loader = None
if dataset[args.task]['replay'] is not None:
    replay_loader = DataLoader(dataset[args.task]['replay'], batch_size=32, shuffle=True, num_workers=0)
replay_loader1 = DataLoader(dataset[args.task+1]['replay'], batch_size=32, shuffle=True, num_workers=0)

if 'full' in args.baseline:
    appr = Appr_noncl(args)
else:
    appr = Appr(args)

if args.task != 0:
    transform = dataset[args.task]['replay'].transform
else:
    transform = None
appr.train(model, train_loader, test_loaders, replay_loader,replay_loader1,transform)