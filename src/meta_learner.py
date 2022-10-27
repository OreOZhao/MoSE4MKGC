import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from tqdm import tqdm

from datasets import Dataset
from utils import avg_both

seed = 20220530
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class MLP(torch.nn.Module):
    def __init__(self, mod_size, hidden_size):
        """
        inputs : T, E, M
        output: T, M, 1 -> T, M
        """
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(mod_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        print("hidden_size", hidden_size)

    def forward(self, inputs):
        hid = F.relu(self.fc1(inputs))  # T, E, H
        output = F.relu(self.fc2(hid))  # T, E, 1
        output = torch.squeeze(output)  # T, E
        return output


class MetaOptimizer(object):
    def __init__(self, model, optimizer, batch_size=256):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size

    def epoch(self, examples, kg_triples):
        # actual_examples = examples[torch.randperm(examples.shape[0]), :]  # 随机数
        # examples: T, E, M
        my_rand = torch.randperm(examples.shape[0])
        actual_examples = examples[my_rand, :]
        actual_triples = kg_triples[my_rand, :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm(total=examples.shape[0], unit='ex', ncols=80) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                inputs_batch = actual_examples[
                               b_begin:b_begin + self.batch_size
                               ].cuda()  # [batch, 3]
                inputs_batch_triples = actual_triples[
                                       b_begin:b_begin + self.batch_size
                                       ].cuda()
                truth = inputs_batch_triples[:, 2]  # batch size
                preds = self.model.forward(inputs_batch)  # B, E
                l = loss(preds, truth)
                self.optimizer.zero_grad()
                l.backward()

                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(inputs_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
        return l


def meta_test(model, inputs, batch_size=256):
    # inputs size T,E,M
    output = torch.zeros((inputs.shape[0], inputs.shape[1])).squeeze().cuda()  # T,E
    b_begin = 0
    while b_begin < inputs.shape[0]:
        inputs_batch = inputs[b_begin: b_begin + batch_size].cuda()
        output_batch = model(inputs_batch)
        output[b_begin: b_begin + batch_size] += output_batch
        b_begin += batch_size
    return output


def main():
    print(model_path)
    model = torch.load(model_path)
    dataset = Dataset(dataset_name)
    train_valid = dataset.get_examples('valid')
    train_valid_examples = torch.from_numpy(train_valid.astype('int64')).cuda()
    test = dataset.get_examples('test')
    test_examples = torch.from_numpy(test.astype('int64')).cuda()
    valid = dataset.get_examples('valid')
    valid_examples = torch.from_numpy(valid.astype('int64')).cuda()

    num_ents = dataset.get_shape()[0]
    num_rels = dataset.get_shape()[1]

    missing = ['rhs', 'lhs']

    at = (1, 3, 10)

    train_valid_ent_scores = {}
    train_valid_target_scores = {}

    valid_ent_scores = {}
    valid_target_scores = {}

    for m in missing:
        q = train_valid_examples.clone()
        if m == 'lhs':
            tmp = torch.clone(q[:, 0])
            q[:, 0] = q[:, 2]
            q[:, 2] = tmp
            q[:, 1] += dataset.n_predicates // 2
        train_valid_ent_scores[m], train_valid_target_scores[m] = model.get_ranking_score(q, dataset.to_skip[m],
                                                                                          batch_size=1000,
                                                                                          filt=False)  # filtered scores and targets

        train_valid_ent_scores[m] = train_valid_ent_scores[m].permute(1, 2, 0)  # T, E, M

    # filter=False, so to_skip is invalid
    # (M,T,E), (M,T)

    for m in missing:
        q = valid_examples.clone()
        if m == 'lhs':
            tmp = torch.clone(q[:, 0])
            q[:, 0] = q[:, 2]
            q[:, 2] = tmp
            q[:, 1] += dataset.n_predicates // 2
        valid_ent_scores[m], valid_target_scores[m] = model.get_ranking_score(q, dataset.to_skip[m], batch_size=1000,
                                                                              filt=True)  # (M,T,E), (M,T)

    test_ent_scores = {}
    test_target_scores = {}

    for m in missing:
        q = test_examples.clone()
        if m == 'lhs':
            tmp = torch.clone(q[:, 0])
            q[:, 0] = q[:, 2]
            q[:, 2] = tmp
            q[:, 1] += dataset.n_predicates // 2
        test_ent_scores[m], test_target_scores[m] = model.get_ranking_score(q, dataset.to_skip[m], batch_size=1000,
                                                                            filt=True)  # (M,T,E), (M,T)

    hs = 32
    meta_learner = MLP(mod_size=3, hidden_size=hs)
    # hidden_size (4, 8, 16, 32, 64)
    meta_learner.to('cuda')
    lr = 0.1
    print("lr: ", lr)

    optim_method = optim.Adagrad(meta_learner.parameters(), lr=lr)
    optimizer = MetaOptimizer(meta_learner, optim_method)
    print("batch_size", optimizer.batch_size)
    best_hits10 = 0.0

    max_epochs = 20
    print("max_epochs: ", max_epochs)

    meta_test_scores = {}
    meta_test_targets = {}
    meta_test_ranks = {}
    meta_test_mr = {}
    meta_test_mrr = {}
    meta_test_hits = {}
    meta_learner_path = '/'.join(model_path.split('/')[:-1]) + '/meta_mlp.pth'
    for e in range(max_epochs):
        print(e)
        for m in missing:
            q = train_valid_examples.clone()
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += dataset.n_predicates // 2
            cur_loss = optimizer.epoch(train_valid_ent_scores[m], q)

        for m in missing:
            q = test_examples.clone()
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += dataset.n_predicates // 2
            with torch.no_grad():
                meta_test_scores[m] = meta_test(meta_learner, test_ent_scores[m].permute(1, 2, 0))
                meta_test_targets[m] = meta_test(meta_learner, test_target_scores[m].permute(1, 2, 0))
                meta_test_ranks[m] = model.get_meta_score_filtered_ranking(meta_test_scores[m],
                                                                           meta_test_targets[m],
                                                                           q, dataset.to_skip[m])
                meta_test_mr[m] = torch.mean(meta_test_ranks[m]).item()
                meta_test_mrr[m] = torch.mean(1. / meta_test_ranks[m]).item()
                meta_test_hits[m] = torch.FloatTensor((list(map(
                    lambda x: torch.mean((meta_test_ranks[m] <= x).float()).item(),
                    at
                ))))

        print("TEST: ")
        metrics = avg_both(meta_test_mr, meta_test_mrr, meta_test_hits)
        print(metrics)
        if metrics['hits@[1,3,10]'][2] > best_hits10:
            best_hits10 = metrics['hits@[1,3,10]'][2]
            meta_learner_path = '/'.join(model_path.split('/')[:-1]) + '/' + datetime.now().strftime(
                "%Y%m%d_%H%M") + '_hs_' + str(hs) + '_hits10_' + str(best_hits10) + '_MLP_relu.pth'
            torch.save(meta_learner, meta_learner_path)
            print('save meta learner to ', meta_learner_path)
    print("done")


big_datasets = ['WN9', 'WN18', 'FB15K-237']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Meta learner Inference"
)
parser.add_argument(
    '--dataset', choices=datasets,
    default='FB15K-237',
    help="Dataset in {}".format(datasets)
)
parser.add_argument(
    '--model_path',
    help="Your model path saved in training learn.py"
)

args = parser.parse_args()
print("running setting args: ", args)

model_path = args.model_path
dataset_name = args.dataset
main()
