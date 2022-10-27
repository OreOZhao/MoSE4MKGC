import argparse
import ast

import torch

from datasets import Dataset
from ensemble import get_modality_weight_for_relation
from utils import avg_both

big_datasets = ['WN9', 'WN18', 'FB15K-237']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Boosting Inference"
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
parser.add_argument(
    '--boosting', default=True, type=ast.literal_eval,
    help="Whether fusion description modality graph."
)
args = parser.parse_args()
print("running setting args: ", args)

model_path = args.model_path
dataset_name = args.dataset

print(model_path)
model = torch.load(model_path)

dataset = Dataset(dataset_name)
valid = dataset.get_examples('valid')
test = dataset.get_examples('test')
valid_examples = torch.from_numpy(valid.astype('int64')).cuda()
test_examples = torch.from_numpy(test.astype('int64')).cuda()

num_ents = dataset.get_shape()[0]
num_rels = dataset.get_shape()[1]

missing = ['rhs', 'lhs']

valid_ranks = {}
valid_ent_ranks = {}
valid_ent_scores = {}
valid_target_scores = {}
valid_mean_rank = {}
valid_mean_reciprocal_rank = {}
valid_hits_at = {}

test_ranks = {}
test_ent_ranks = {}
test_ent_scores = {}
test_target_scores = {}
test_mean_rank = {}
test_mean_reciprocal_rank = {}
test_hits_at = {}

at = (1, 3, 10)

for m in missing:
    q = valid_examples.clone()
    if m == 'lhs':
        tmp = torch.clone(q[:, 0])
        q[:, 0] = q[:, 2]
        q[:, 2] = tmp
        q[:, 1] += dataset.n_predicates // 2
    valid_ent_scores[m], valid_target_scores[m] = model.get_ranking_score(q, dataset.to_skip[m], batch_size=2000,
                                                                          filt=True)  # filtered scores and targets

for m in missing:
    q = test_examples.clone()
    if m == 'lhs':
        tmp = torch.clone(q[:, 0])
        q[:, 0] = q[:, 2]
        q[:, 2] = tmp
        q[:, 1] += dataset.n_predicates // 2
    test_ent_scores[m], test_target_scores[m] = model.get_ranking_score(q, dataset.to_skip[m], batch_size=2000,
                                                                        filt=True)

rel_score = {}
rel_rank = {}
rel_test_mean_rank = {}
rel_test_mean_reciprocal_rank = {}
rel_test_hits_at = {}

for m in missing:
    print(m)
    q = test_examples.clone()
    if m == 'lhs':
        tmp = torch.clone(q[:, 0])
        q[:, 0] = q[:, 2]
        q[:, 2] = tmp
        q[:, 1] += dataset.n_predicates // 2

    # ------------------------ boosting ------------------------------
    rels_id = q[:, 1]
    if args.boosting:
        rel_score[m], mod_tri_weight = get_modality_weight_for_relation(dataset_name, test_ent_scores[m], rels_id,
                                                                        valid_ent_scores[m], valid_target_scores[m],
                                                                        'boosting')
    else:
        rel_score[m], mod_tri_weight = get_modality_weight_for_relation(dataset_name, test_ent_scores[m], rels_id,
                                                                        valid_ent_scores[m], valid_target_scores[m],
                                                                        'average')
    rel_rank[m] = model.get_ensemble_score_filtered_ranking(rel_score[m], mod_tri_weight, q,
                                                            dataset.to_skip[m],
                                                            batch_size=2000)
    rel_test_mean_rank[m] = torch.mean(rel_rank[m]).item()
    rel_test_mean_reciprocal_rank[m] = torch.mean(1. / rel_rank[m]).item()
    rel_test_hits_at[m] = torch.FloatTensor((list(map(
        lambda x: torch.mean((rel_rank[m] <= x).float()).item(),
        at
    ))))

print(rel_test_mean_rank)
print(rel_test_mean_reciprocal_rank)
print(rel_test_hits_at)
print(avg_both(rel_test_mean_rank, rel_test_mean_reciprocal_rank, rel_test_hits_at))
print("done")
