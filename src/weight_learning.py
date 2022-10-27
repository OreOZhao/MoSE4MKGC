import torch


def learn_rel_specific_modal_weights(triples, rel_ent_scores, rel_target_scores):
    """
    triples: [h,r,t] (T, 3), triples of valid sets
    rel_ent_ranks: M,T,E
    return [M]

    1. init D_m =1/|P|
    2. construct right or wrong matrix, [p]_m
    3. model weight, w_i^m = f(D^m, [p]_m)
    4. triple weight, D_m+1 = g(D^m, w^m,[p]_m)
    """
    M = rel_ent_scores.shape[0]
    T = rel_ent_scores.shape[1]
    E = rel_ent_scores.shape[2]
    assert T == len(triples)
    if T < 10:
        return torch.ones(size=(M,)) / torch.sum(torch.ones(size=(M,)))

    # 1. D_0
    weight_matrix = torch.ones(size=(T, E))  # D_m entity pair weights D_0, (T, E)
    weight_matrix = weight_matrix / weight_matrix.sum(dim=1, keepdim=True)  # normalize to 1/|P|
    # 2. [p]_m, right 1, wrong -1 (M,T,E)
    rank_matrix = get_rank_pm_matrix(rel_ent_scores, rel_target_scores)

    last_modal_weights = torch.zeros(size=(M,))
    for m in range(M):
        # 3. w_m^i
        modal_weights = get_modal_weights_for_one_round(weight_matrix, rank_matrix)  # w_m, M
        modal_order = torch.argsort(modal_weights, descending=True)  # modality idx as weight top down

        for mod in modal_order:
            if last_modal_weights[mod] <= 0:  # not selected before
                best_modal = mod
                break
        best_modal_weight = modal_weights[best_modal]
        last_modal_weights[best_modal] += best_modal_weight

        # 4. update D_m weight_matrix
        coeff = torch.exp(-best_modal_weight * rank_matrix[best_modal])  # T,E
        weight_matrix = weight_matrix * coeff  # T,E
        weight_matrix = weight_matrix / torch.sum(weight_matrix)  # Z_m is the sum of all (T,E)

    return last_modal_weights


def get_rank_pm_matrix(rel_ent_scores, rel_target_scores):
    # rel_ent_scores M,T,E
    # rel_target_scores M,T
    M = rel_ent_scores.shape[0]
    T = rel_ent_scores.shape[1]
    E = rel_ent_scores.shape[2]
    rank_matrix = torch.zeros(size=(M, T, E))
    # target_scores = rel_target_scores.unsqueeze(dim=-1)
    rank_matrix[rel_ent_scores >= rel_target_scores] = -1
    rank_matrix[rel_ent_scores < rel_target_scores] = 1
    # rel_ent_scores is filtered, so the score of target entity is -infinite, rank_matrix[target]=1
    return rank_matrix


def get_modal_weights_for_one_round(weight_matrix, rank_matrix):
    # weight_matrix (T,E), weight for each entity pair
    # rank_matrix (M,T,E)
    M = rank_matrix.shape[0]
    T = rank_matrix.shape[1]
    E = rank_matrix.shape[2]
    wrong = torch.zeros(size=(M,))
    correct = torch.zeros(size=(M,))
    for m in range(M):
        rank = rank_matrix[m]  # T,E
        D_m_sum_wrong = torch.sum(weight_matrix[rank == -1])
        D_m_sum_correct = torch.sum(weight_matrix[rank == 1])
        wrong[m] = wrong[m] + D_m_sum_wrong + 1e-10
        correct[m] = correct[m] + D_m_sum_correct + 1e-10
    modal_weights = 0.5 * torch.log(correct / wrong)  # M
    return modal_weights
