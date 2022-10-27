import argparse
import os

import torch
from torch import optim

torch.cuda.empty_cache()
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import Dataset
from models import CP, ComplEx, RSME, ComplExMDR
from regularizers import F2, N3
from optimizers import KBCOptimizer
from datetime import datetime

import json
import numpy as np
import time
import ast
from utils import avg_both

# os.environ["CUDA_VISIBLE_DEVICES"] = device
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

big_datasets = ['WN9', 'WN18', 'FB15K-237']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    default='FB15K-237',
    help="Dataset in {}".format(datasets)
)

models = ['CP', 'ComplEx', 'RSME', 'ComplExMDR']
parser.add_argument(
    '--model', choices=models,
    default='ComplExMDR',
    help="Model in {}".format(models)
)

parser.add_argument(
    '--alpha', default=1, type=float,
    help="Modality embedding ratio in modality_structure fusion. Default=1 means dscp/img emb does not fuse structure emb."
)

parser.add_argument(
    '--modality_split', default=True, type=ast.literal_eval,
    help="Whether split modalities."
)

parser.add_argument(
    '--fusion_img', default=True, type=ast.literal_eval,
    help="Whether fusion img modality graph."
)

parser.add_argument(
    '--fusion_dscp', default=True, type=ast.literal_eval,
    help="Whether fusion description modality graph."
)

parser.add_argument(
    '--scale', default=20, type=float,
    help="temp parameter"
)

regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=2000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0.01, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
parser.add_argument(
    '--note', default=None,
    help="model setting or ablation note for ckpt save"
)
parser.add_argument(
    '--early_stopping', default=10, type=int,
    help="stop training until Hits10 stop increasing after early stopping epoches"
)
parser.add_argument(
    '--ckpt_dir', default='../ckpt/'
)
parser.add_argument(
    '--img_info', default='../data/FB15K-237/img_vec.pickle'
)
parser.add_argument(
    '--dscp_info', default='../data/FB15K-237/dscp_vec.pickle'
)

args = parser.parse_args()
print("running setting args: ", args)

dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

print(dataset.get_shape())
model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    'RSME': lambda: RSME(dataset.get_shape(), args.rank, args.init),
    'ComplExMDR': lambda: ComplExMDR(dataset.get_shape(), args.rank, args.init, args.modality_split, args.fusion_img,
                                     args.fusion_dscp, args.alpha,
                                     args.dataset, args.scale, args.img_info, args.dscp_info)
}[args.model]()

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]

device = 'cuda'
model.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size, args.modality_split, args.fusion_img,
                         args.fusion_dscp)
scheduler = ReduceLROnPlateau(optim_method, 'min', factor=0.5, verbose=True, patience=10, threshold=1e-3)


# scheduler = StepLR(optim_method, step_size=10, gamma=0.5)


def create_subfolder(log_dir):
    index = len(os.listdir(log_dir))
    try:
        new_folder_path = os.path.join(log_dir, str(index))
        os.makedirs(new_folder_path)
    except Exception as e:
        print(e)
        print(new_folder_path)
    return new_folder_path


ckpt_dir = args.ckpt_dir
if not ckpt_dir.endswith('/'):
    ckpt_dir = ckpt_dir + '/'
run_dir = create_subfolder(ckpt_dir)

cur_loss = 0
best_loss = 10000
curve = {'train': [], 'valid': [], 'test': []}
curve_loss = []
best_mr = 100000
best_mrr = 0
best_hits = [0, 0, 0]
best_epoch = 0
best_val_model_test_result = {}
train, test, valid = [0, 0, 0]
model_path = run_dir + '/m-' + datetime.now().strftime("%Y%m%d_%H%M") + '-n-' + str(args.note) + '.pth'
since = time.time()
for e in range(args.max_epochs):
    # valid, test, train = [
    #     avg_both(*dataset.eval(model, split, 10))
    #     for split in ['valid', 'test', 'train']
    # ]
    cur_loss = optimizer.epoch(examples).tolist()
    curve_loss.append(cur_loss)
    # scheduler.step()
    if cur_loss < best_loss:
        best_loss = cur_loss
    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 3000))
            for split in ['valid', 'test', 'train']
        ]
        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("epoch: %d" % (e + 1))
        print(args.note)
        print("\t TRAIN: ", train)
        print("\t TEST : ", test)
        print("\t VALID : ", valid)
        if args.modality_split:
            print("Note that with modality split, "
                  "the result during training is more like an upper bound but not final performance:)")
        if valid['hits@[1,3,10]'][-1] > best_hits[-1]:
            best_mrr = valid['MRR']
            best_mr = valid['MR']
            best_hits = valid['hits@[1,3,10]']
            best_epoch = e + 1
            best_val_model_test_result = test
            torch.save(model, model_path)

        scheduler.step(valid['hits@[1,3,10]'][2])
        print("Learning rate at epoch {}: {}".format(e + 1, scheduler._last_lr))

    if (e + 1 - best_epoch) > args.early_stopping:
        break

time_elapsed = time.time() - since
sec_per_epoch = time_elapsed / float(e + 1)
print('Time consuming: {:.3f}s, average sec per epoch: {:.3f}s'.format(time_elapsed, sec_per_epoch))
print('last_lr: ', scheduler._last_lr)
print('Test result on best Valid MRR model: ', best_val_model_test_result)

result = {'epoch': e + 1,
          'best_loss': best_loss, 'best_epoch': best_epoch,
          'best_mrr': best_mrr, 'best_mr': best_mr, 'best_hits10': best_hits,
          'curve': curve, 'curve_loss': curve_loss,
          'train': train, 'test': test, 'val': valid, 'final_result': best_val_model_test_result,
          'sec_per_epoch': sec_per_epoch, 'last_lr': scheduler._last_lr, 'run_dir': run_dir}
result.update(args.__dict__)

print(result)

if run_dir:
    print(run_dir)
    with open(os.path.join(run_dir,
                           'result.json'),
              'w') as f:
        json.dump(result, f, indent=2)
