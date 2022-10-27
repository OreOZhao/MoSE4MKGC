import torch
from tqdm import tqdm

from datasets import Dataset
from weight_learning import learn_rel_specific_modal_weights


def get_boost_weight_for_relation(dataset_name, valid_ent_scores, valid_target_scores, boosting=True):
    # valid_ent_scores M,T,E
    # valid_target_scores M,T
    sizes = valid_ent_scores.shape
    M = sizes[0]  # modality count
    T = sizes[1]  # triples count
    E = sizes[2]  # entity count
    dataset = Dataset(dataset_name)
    valid = dataset.get_examples('valid')
    valid_examples = torch.from_numpy(valid.astype('int64')).cuda()
    num_rels = dataset.get_shape()[1]
    if not boosting:  # average inference
        rels_modal_weights = torch.ones(size=(M, num_rels))
        return rels_modal_weights / 3
    rels_modal_weights = torch.zeros(size=(M, num_rels))  # M,R
    missing = ['rhs', 'lhs']
    rel_range = {'rhs': range(0, num_rels // 2), 'lhs': range(num_rels // 2, num_rels)}
    for m in missing:
        q = valid_examples.clone()  # T
        if m == 'lhs':
            tmp = torch.clone(q[:, 0])
            q[:, 0] = q[:, 2]
            q[:, 2] = tmp
            q[:, 1] += dataset.n_predicates // 2
        for r in tqdm(rel_range[m], ncols=80):
            triples = q[q[:, 1] == r]
            rel_ent_scores = valid_ent_scores[:, q[:, 1] == r, :]  # M, T_r, E
            rel_target_scores = valid_target_scores[:, q[:, 1] == r]  # M, T_r, 1
            r_modal_weights = learn_rel_specific_modal_weights(triples, rel_ent_scores, rel_target_scores)
            # print(r, r_modal_weights)
            rels_modal_weights[:, r] += r_modal_weights
    return rels_modal_weights  # M,R


def get_modality_weight_for_relation(dataset_name, test_score, rels_id,
                                     valid_ent_scores, valid_target_scores,
                                     mode='boosting'):
    if mode == 'boosting':
        modality_weight = get_boost_weight_for_relation(dataset_name, valid_ent_scores, valid_target_scores, True)
    elif mode == 'average':
        modality_weight = get_boost_weight_for_relation(dataset_name, valid_ent_scores, valid_target_scores, False)
    elif mode == 'boosting_mask':
        modality_weight = get_boost_weight_for_relation(dataset_name, valid_ent_scores, valid_target_scores, True)
        modality_weight = (modality_weight == modality_weight.max(dim=0)[0]).float()
        modality_weight /= modality_weight.sum(dim=0)

    #
    # modality_weight [M, Rels]
    # test_score [M,T,E]
    # rels_id [T]
    # mod_tri_weight = [M,T]
    mod_tri_weight = modality_weight[:, rels_id]  # M,T
    mod_tri_weight = mod_tri_weight.unsqueeze(dim=-1)  # M,T,1
    score_ensemble = test_score * mod_tri_weight  # M,T,E
    score_ensemble = score_ensemble.sum(dim=0)  # T,E
    # print(mod_tri_weight)
    return score_ensemble, mod_tri_weight
    # import pandas as pd
    # pdweight = pd.DataFrame(modality_weight)
    # pdweight.to_csv('wn18_rel_weight_rhs.csv',index=False,header=False)
