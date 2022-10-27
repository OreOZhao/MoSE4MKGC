import pickle
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)


class ComplExMDR(KBCModel):
    """
    modality decoupled relation
    """

    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,
            modality_split=True, fusion_img=True, fusion_dscp=True,
            alpha=1,
            dataset='FB15K-237',
            scale=16,
            img_info='../data/FB15K-237/img_vec.pickle',
            dscp_info='../data/FB15K-237/dscp_vec.pickle'
    ):
        super(ComplExMDR, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.r_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])

        self.r_embeddings[0].weight.data *= init_size  # entity
        self.r_embeddings[1].weight.data *= init_size  # relation

        self.modality_split = modality_split
        self.fusion_img = fusion_img
        self.fusion_dscp = fusion_dscp

        self.alpha = alpha  # alpha = 1 means image/text emb does not fuse structure modality

        self.temp = scale

        print("modality_split: {}, fusion_img: {}, fusion_dscp: {}"
              .format(self.modality_split, self.fusion_img, self.fusion_dscp))

        if self.fusion_img:
            self.img_dimension = 1000
            self.img_info = pickle.load(open(img_info, 'rb'))
            self.img_vec = torch.from_numpy(self.img_info).float().cuda()
            self.img_post_mats = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
            # 图映射到和structure emb一个大小
            nn.init.xavier_uniform(self.img_post_mats)  # 每层网络保证输入输出的方差相同

            self.img_rel_embeddings = nn.Embedding(sizes[1], 2 * rank, sparse=True)
            self.img_rel_embeddings.weight.data *= init_size

        if self.fusion_dscp:
            self.dscp_dimension = 768
            self.dscp_info = pickle.load(open(dscp_info, 'rb'))
            self.dscp_vec = torch.from_numpy(self.dscp_info).float().cuda()
            self.dscp_post_mats = nn.Parameter(torch.Tensor(self.dscp_dimension, 2 * rank), requires_grad=True)
            nn.init.xavier_uniform(self.dscp_post_mats)  # 每层网络保证输入输出的方差相同

            self.dscp_rel_embeddings = nn.Embedding(sizes[1], 2 * rank, sparse=True)
            self.dscp_rel_embeddings.weight.data *= init_size

    def score(self, x):
        lhs = self.r_embeddings[0](x[:, 0])
        rel = self.r_embeddings[1](x[:, 1])
        rhs = self.r_embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        score_str = torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )  # ComplEx的score function
        return_value = []
        return_value.append(score_str)
        if self.modality_split:
            if self.fusion_img:
                img_embeddings = self.img_vec.mm(self.img_post_mats)
                str_img_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
                str_img_embeddings /= self.temp
                lhs_i = str_img_embeddings[(x[:, 0])]
                rel_i = self.img_rel_embeddings(x[:, 1])
                rhs_i = str_img_embeddings[(x[:, 2])]

                lhs_i = lhs_i[:, :self.rank], lhs_i[:, self.rank:]
                rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]
                rhs_i = rhs_i[:, :self.rank], rhs_i[:, self.rank:]
                score_img = torch.sum(
                    (lhs_i[0] * rel_i[0] - lhs_i[1] * rel_i[1]) * rhs_i[0] +
                    (lhs_i[0] * rel_i[1] + lhs_i[1] * rel_i[0]) * rhs_i[1],
                    1, keepdim=True
                )  # ComplEx的score function
                # [batch , 1]
                return_value.append(score_img)
            if self.fusion_dscp:
                dscp_embeddings = self.dscp_vec.mm(self.dscp_post_mats)
                str_dscp_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * dscp_embeddings
                str_dscp_embeddings /= self.temp
                lhs_i = str_dscp_embeddings[(x[:, 0])]
                rel_i = self.dscp_rel_embeddings(x[:, 1])
                rhs_i = str_dscp_embeddings[(x[:, 2])]

                lhs_i = lhs_i[:, :self.rank], lhs_i[:, self.rank:]
                rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]
                rhs_i = rhs_i[:, :self.rank], rhs_i[:, self.rank:]
                score_dscp = torch.sum(
                    (lhs_i[0] * rel_i[0] - lhs_i[1] * rel_i[1]) * rhs_i[0] +
                    (lhs_i[0] * rel_i[1] + lhs_i[1] * rel_i[0]) * rhs_i[1],
                    1, keepdim=True
                )  # ComplEx的score function
                # [batch , 1]
                return_value.append(score_dscp)
        return tuple(return_value)

    def forward(self, x):
        lhs = self.r_embeddings[0](x[:, 0])
        rel = self.r_embeddings[1](x[:, 1])
        rhs = self.r_embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.r_embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]

        score_str = (
                (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        )
        factors_str = (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

        return_value = []
        return_value.append(score_str)
        return_value.append(factors_str)

        if self.modality_split:
            if self.fusion_img:
                img_embeddings = self.img_vec.mm(self.img_post_mats)
                str_img_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
                str_img_embeddings /= self.temp

                lhs_i = str_img_embeddings[(x[:, 0])]
                rel_i = self.img_rel_embeddings(x[:, 1])
                rhs_i = str_img_embeddings[(x[:, 2])]
                lhs_i = lhs_i[:, :self.rank], lhs_i[:, self.rank:]
                rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]
                rhs_i = rhs_i[:, :self.rank], rhs_i[:, self.rank:]

                to_score_i = str_img_embeddings
                to_score_i = to_score_i[:, :self.rank], to_score_i[:, self.rank:]

                score_img = (
                        (lhs_i[0] * rel_i[0] - lhs_i[1] * rel_i[1]) @ to_score_i[0].transpose(0, 1) +
                        (lhs_i[0] * rel_i[1] + lhs_i[1] * rel_i[0]) @ to_score_i[1].transpose(0, 1)
                )
                # torch.Size([1000, 14951])
                factors_img = (
                    torch.sqrt(lhs_i[0] ** 2 + lhs_i[1] ** 2),
                    torch.sqrt(rel_i[0] ** 2 + rel_i[1] ** 2),
                    torch.sqrt(rhs_i[0] ** 2 + rhs_i[1] ** 2)
                )
                return_value.append(score_img)
                return_value.append(factors_img)
            if self.fusion_dscp:
                dscp_embeddings = self.dscp_vec.mm(self.dscp_post_mats)
                str_dscp_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * dscp_embeddings
                str_dscp_embeddings /= self.temp
                lhs_d = str_dscp_embeddings[(x[:, 0])]
                rel_d = self.dscp_rel_embeddings(x[:, 1])
                rhs_d = str_dscp_embeddings[(x[:, 2])]
                lhs_d = lhs_d[:, :self.rank], lhs_d[:, self.rank:]
                rel_d = rel_d[:, :self.rank], rel_d[:, self.rank:]
                rhs_d = rhs_d[:, :self.rank], rhs_d[:, self.rank:]

                to_score_d = str_dscp_embeddings
                to_score_d = to_score_d[:, :self.rank], to_score_d[:, self.rank:]

                score_dscp = (
                        (lhs_d[0] * rel_d[0] - lhs_d[1] * rel_d[1]) @ to_score_d[0].transpose(0, 1) +
                        (lhs_d[0] * rel_d[1] + lhs_d[1] * rel_d[0]) @ to_score_d[1].transpose(0, 1)
                )
                factors_dscp = (
                    torch.sqrt(lhs_d[0] ** 2 + lhs_d[1] ** 2),
                    torch.sqrt(rel_d[0] ** 2 + rel_d[1] ** 2),
                    torch.sqrt(rhs_d[0] ** 2 + rhs_d[1] ** 2)
                )
                return_value.append(score_dscp)
                return_value.append(factors_dscp)
        return tuple(return_value)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        rhs_str = self.r_embeddings[0].weight[
                  chunk_begin:chunk_begin + chunk_size
                  ].transpose(0, 1)
        return_value = []
        return_value.append(rhs_str)
        if self.modality_split:
            if self.fusion_img:
                img_embeddings = self.img_vec.mm(self.img_post_mats)
                str_img_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
                str_img_embeddings /= self.temp
                rhs_img = str_img_embeddings[
                          chunk_begin:chunk_begin + chunk_size
                          ].transpose(0, 1)
                return_value.append(rhs_img)
            if self.fusion_dscp:
                dscp_embeddings = self.dscp_vec.mm(self.dscp_post_mats)
                str_dscp_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * dscp_embeddings
                str_dscp_embeddings /= self.temp
                rhs_dscp = str_dscp_embeddings[
                           chunk_begin:chunk_begin + chunk_size
                           ].transpose(0, 1)
                return_value.append(rhs_dscp)

        return tuple(return_value)

    def get_queries(self, queries: torch.Tensor):
        embedding = self.r_embeddings[0].weight
        lhs = embedding[(queries[:, 0])]
        rel = self.r_embeddings[1](queries[:, 1])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        queries_str = torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)
        return_value = []
        return_value.append(queries_str)

        if self.modality_split:
            if self.fusion_img:
                img_embeddings = self.img_vec.mm(self.img_post_mats)
                str_img_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
                str_img_embeddings /= self.temp

                lhs_i = str_img_embeddings[(queries[:, 0])]
                rel_i = self.img_rel_embeddings(queries[:, 1])

                lhs_i = lhs_i[:, :self.rank], lhs_i[:, self.rank:]
                rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]

                queries_img = torch.cat([
                    lhs_i[0] * rel_i[0] - lhs_i[1] * rel_i[1],
                    lhs_i[0] * rel_i[1] + lhs_i[1] * rel_i[0]
                ], 1)
                return_value.append(queries_img)

            if self.fusion_dscp:
                dscp_embeddings = self.dscp_vec.mm(self.dscp_post_mats)
                str_dscp_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * dscp_embeddings
                str_dscp_embeddings /= self.temp
                lhs_d = str_dscp_embeddings[(queries[:, 0])]
                rel_d = self.dscp_rel_embeddings(queries[:, 1])

                lhs_d = lhs_d[:, :self.rank], lhs_d[:, self.rank:]
                rel_d = rel_d[:, :self.rank], rel_d[:, self.rank:]

                queries_dscp = torch.cat([
                    lhs_d[0] * rel_d[0] - lhs_d[1] * rel_d[1],
                    lhs_d[0] * rel_d[1] + lhs_d[1] * rel_d[0]
                ], 1)
                return_value.append(queries_dscp)
        return tuple(return_value)

    def get_ranking_score(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1, filt=False
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]  # eval时每次candidate都是所有实体 （chuck_size default = -1）
        scores_str_all = torch.zeros(size=(len(queries), self.sizes[2]))
        scores_img_all = torch.zeros(size=(len(queries), self.sizes[2]))
        scores_dscp_all = torch.zeros(size=(len(queries), self.sizes[2]))
        targets_str_all = torch.zeros(size=(len(queries), 1))
        targets_img_all = torch.zeros(size=(len(queries), 1))
        targets_dscp_all = torch.zeros(size=(len(queries), 1))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                pbar = tqdm(total=len(queries), ncols=80)
                while b_begin < len(queries):  # batch iter
                    these_queries = queries[b_begin:b_begin + batch_size]
                    rhs_str, rhs_img, rhs_dscp = self.get_rhs(c_begin, chunk_size)
                    # rhs: rank * ents
                    # q: batch * rank
                    q_str, q_img, q_dscp = self.get_queries(these_queries)
                    scores_str = q_str @ rhs_str  # batch, ents
                    scores_img = q_img @ rhs_img  # 500, 14951
                    scores_dscp = q_dscp @ rhs_dscp
                    targets_str, targets_img, targets_dscp = self.score(these_queries)  # batch,1
                    targets_str_all[b_begin:b_begin + batch_size] += targets_str.cpu()
                    targets_img_all[b_begin:b_begin + batch_size] += targets_img.cpu()
                    targets_dscp_all[b_begin:b_begin + batch_size] += targets_dscp.cpu()
                    if filt:
                        for i, query in enumerate(these_queries):
                            filter_out = filters[(query[0].item(), query[1].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                            if chunk_size < self.sizes[2]:  # if candidate is not all entity
                                filter_in_chunk = [
                                    int(x - c_begin) for x in filter_out
                                    if c_begin <= x < c_begin + chunk_size
                                ]
                                scores_str[i, torch.LongTensor(filter_in_chunk)] = -1e6
                                if self.fusion_img:
                                    scores_img[i, torch.LongTensor(filter_in_chunk)] = -1e6
                                if self.fusion_dscp:
                                    scores_dscp[i, torch.LongTensor(filter_in_chunk)] = -1e6
                            else:
                                scores_str[i, torch.LongTensor(filter_out)] = -1e6
                                if self.fusion_img:
                                    scores_img[i, torch.LongTensor(filter_out)] = -1e6
                                if self.fusion_dscp:
                                    scores_dscp[i, torch.LongTensor(filter_out)] = -1e6
                            pbar.update(1)
                    else:
                        pbar.update(batch_size)
                    scores_str_all[b_begin:b_begin + batch_size] += scores_str.cpu()
                    scores_img_all[b_begin:b_begin + batch_size] += scores_img.cpu()
                    scores_dscp_all[b_begin:b_begin + batch_size] += scores_dscp.cpu()
                    b_begin += batch_size
                c_begin += chunk_size
                pbar.close()
        return torch.stack([scores_str_all, scores_img_all, scores_dscp_all]), \
               torch.stack([targets_str_all, targets_img_all, targets_dscp_all])  # (M,T,E), (M,T)

    def get_ensemble_score_filtered_ranking(
            self, ensemble_score: torch.Tensor,
            mod_tri_weight: torch.Tensor,
            queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]  # eval时每次candidate都是所有实体 （chuck_size default = -1）
        ranks = torch.ones(len(queries))
        ranks_ent = torch.zeros(size=(len(queries), self.sizes[0]))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                pbar = tqdm(total=len(queries), ncols=80)
                while b_begin < len(queries):  # batch iter
                    these_queries = queries[b_begin:b_begin + batch_size]
                    these_scores = ensemble_score[b_begin:b_begin + batch_size].clone().cuda()
                    these_modalities = mod_tri_weight[:, b_begin:b_begin + batch_size, :].clone().cuda()  # 3, batch, 1
                    targets_str, targets_img, targets_dscp = self.score(these_queries)  # batch,1
                    targets_all = torch.stack([targets_str, targets_img, targets_dscp])  # 3, batch, 1
                    targets_ensemble = torch.sum(targets_all * these_modalities, dim=0)
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:  # if candidate is not all entity
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            these_scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            these_scores[i, torch.LongTensor(filter_out)] = -1e6
                        pbar.update(1)
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (these_scores >= targets_ensemble).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                c_begin += chunk_size
                pbar.close()
        return ranks

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """

        if chunk_size < 0:
            chunk_size = self.sizes[2]  # eval时每次candidate都是所有实体 （chuck_size default = -1）
        ranks = torch.ones(len(queries))
        ranks_str = torch.ones(len(queries))
        ranks_img = torch.ones(len(queries))
        ranks_dscp = torch.ones(len(queries))
        ranks_fusion = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                pbar = tqdm(total=len(queries), ncols=80)
                while b_begin < len(queries):  # batch iter
                    if not self.modality_split or (
                            self.modality_split and not self.fusion_img and not self.fusion_dscp):
                        these_queries = queries[b_begin:b_begin + batch_size]
                        rhs = self.get_rhs(c_begin, chunk_size)[0]
                        q = self.get_queries(these_queries)[0]
                        scores_str = q @ rhs
                        scores = scores_str  # torch.Size([500, 14951])
                        targets = self.score(these_queries)[0]
                        for i, query in enumerate(these_queries):
                            filter_out = filters[(query[0].item(), query[1].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                            if chunk_size < self.sizes[2]:  # if candidate is not all entity
                                filter_in_chunk = [
                                    int(x - c_begin) for x in filter_out
                                    if c_begin <= x < c_begin + chunk_size
                                ]
                                scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                            else:
                                scores[i, torch.LongTensor(filter_out)] = -1e6
                            pbar.update(1)
                        ranks[b_begin:b_begin + batch_size] += torch.sum(
                            (scores >= targets).float(), dim=1
                        ).cpu()
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size]
                        if self.fusion_img and self.fusion_dscp:
                            rhs_str, rhs_img, rhs_dscp = self.get_rhs(c_begin, chunk_size)
                            q_str, q_img, q_dscp = self.get_queries(these_queries)
                            scores_str = q_str @ rhs_str  # 500, 14951
                            scores_img = q_img @ rhs_img  # 500, 14951
                            scores_dscp = q_dscp @ rhs_dscp
                            targets_str, targets_img, targets_dscp = self.score(these_queries)  # 500,1
                        elif self.fusion_img:
                            rhs_str, rhs_img = self.get_rhs(c_begin, chunk_size)
                            q_str, q_img = self.get_queries(these_queries)
                            scores_str = q_str @ rhs_str  # 500, 14951
                            scores_img = q_img @ rhs_img  # 500, 14951
                            targets_str, targets_img = self.score(these_queries)  # 500,1
                        elif self.fusion_dscp:
                            rhs_str, rhs_dscp = self.get_rhs(c_begin, chunk_size)
                            q_str, q_dscp = self.get_queries(these_queries)
                            scores_str = q_str @ rhs_str  # 500, 14951
                            scores_dscp = q_dscp @ rhs_dscp  # 500, 14951
                            targets_str, targets_dscp = self.score(these_queries)  # 500,1
                        for i, query in enumerate(these_queries):
                            filter_out = filters[(query[0].item(), query[1].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                            if chunk_size < self.sizes[2]:  # if candidate is not all entity
                                filter_in_chunk = [
                                    int(x - c_begin) for x in filter_out
                                    if c_begin <= x < c_begin + chunk_size
                                ]
                                scores_str[i, torch.LongTensor(filter_in_chunk)] = -1e6
                                if self.fusion_img:
                                    scores_img[i, torch.LongTensor(filter_in_chunk)] = -1e6
                                if self.fusion_dscp:
                                    scores_dscp[i, torch.LongTensor(filter_in_chunk)] = -1e6
                            else:
                                scores_str[i, torch.LongTensor(filter_out)] = -1e6
                                if self.fusion_img:
                                    scores_img[i, torch.LongTensor(filter_out)] = -1e6
                                if self.fusion_dscp:
                                    scores_dscp[i, torch.LongTensor(filter_out)] = -1e6
                            pbar.update(1)

                        ranks_str[b_begin:b_begin + batch_size] += torch.sum(
                            (scores_str >= targets_str).float(), dim=1
                        ).cpu()
                        if self.fusion_img:
                            ranks_img[b_begin:b_begin + batch_size] += torch.sum(
                                (scores_img >= targets_img).float(), dim=1
                            ).cpu()
                        if self.fusion_dscp:
                            ranks_dscp[b_begin:b_begin + batch_size] += torch.sum(
                                (scores_dscp >= targets_dscp).float(), dim=1
                            ).cpu()
                    b_begin += batch_size

                c_begin += chunk_size
                pbar.close()

        if not self.modality_split or (self.modality_split and not self.fusion_img and not self.fusion_dscp):
            return ranks
        else:
            if self.fusion_img and self.fusion_dscp:
                ranks_fusion = torch.min(ranks_str, torch.min(ranks_img, ranks_dscp))
                print("ranks_str: {:.4f}, ranks_img: {:.4f}, ranks_dscp: {:.4f}".format(
                    sum(ranks_fusion == ranks_str) / ranks.shape[0],
                    sum(ranks_fusion == ranks_img) / ranks.shape[0],
                    sum(ranks_fusion == ranks_dscp) / ranks.shape[0]))
                # return ranks_str, ranks_img, ranks_dscp
            elif self.fusion_img:
                ranks_fusion = torch.min(ranks_str, ranks_img)
            elif self.fusion_dscp:
                ranks_fusion = torch.min(ranks_str, ranks_dscp)
            return ranks_fusion

    def get_meta_score_filtered_ranking(
            self, ensemble_score: torch.Tensor,
            ensemble_target: torch.Tensor,
            queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]  # eval时每次candidate都是所有实体 （chuck_size default = -1）
        ranks = torch.ones(len(queries))
        ranks_ent = torch.zeros(size=(len(queries), self.sizes[0]))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                pbar = tqdm(total=len(queries), ncols=80)
                while b_begin < len(queries):  # batch iter
                    these_queries = queries[b_begin:b_begin + batch_size]
                    these_scores = ensemble_score[b_begin:b_begin + batch_size].clone().cuda()
                    targets_ensemble = ensemble_target[b_begin:b_begin + batch_size].clone().cuda()  # batch
                    targets_ensemble = targets_ensemble.unsqueeze(-1)  # batch, 1
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:  # if candidate is not all entity
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            these_scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            these_scores[i, torch.LongTensor(filter_out)] = -1e6
                        pbar.update(1)
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (these_scores >= targets_ensemble).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                c_begin += chunk_size
                pbar.close()
        return ranks


# ------------ RSME model from https://github.com/wangmengsd/RSME ---------------------

def sc_wz_01(len, num_1):
    A = [1 for i in range(num_1)]
    B = [0 for i in range(len - num_1)]
    C = A + B
    C = A + B
    np.random.shuffle(C)
    return np.array(C, dtype=np.float)


class RSME(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,
            img_info='img_vec_id_fb15k_20_vit.pickle',
            sig_alpha='rel_MPR_SIG_vit_20.pickle',
            rel_pd='rel_MPR_PD_vit_20_mrp{}.pickle'
    ):
        super(RSME, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.r_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])

        self.r_embeddings[0].weight.data *= init_size
        self.r_embeddings[1].weight.data *= init_size
        if not constant:
            self.alpha = torch.from_numpy(pickle.load(open(sig_alpha, 'rb'))).cuda()
            self.alpha = torch.cat((self.alpha, self.alpha), dim=0)
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)  # [14951, 2000]
        self.img_dimension = 1000
        self.img_info = pickle.load(open(img_info, 'rb'))
        self.img_vec = torch.from_numpy(self.img_info).float().cuda()
        if not random_gate:
            self.rel_pd = torch.from_numpy(pickle.load(open(rel_pd.format(remember_rate), 'rb'))).cuda()
        else:
            tmp = pickle.load(open(rel_pd.format(remember_rate), 'rb'))
            self.rel_pd = torch.from_numpy(sc_wz_01(len(tmp), np.sum(tmp))).unsqueeze(1).cuda()

        self.rel_pd = torch.cat((self.rel_pd, self.rel_pd), dim=0)
        # self.alpha[self.img_info['missed'], :] = 1

        self.post_mats = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
        nn.init.xavier_uniform(self.post_mats)

    def score(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 2])]

            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1)

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            score_str = torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            )
            # beta = 0.95
            return beta * score_str + (1 - beta) * score_img


        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings

            lhs = embedding[(x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = embedding[(x[:, 2])]

            rel_pd = self.rel_pd[(x[:, 1])]
            lhs_img = self.img_vec[(x[:, 0])]
            rhs_img = self.img_vec[(x[:, 2])]

            # score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1)
            if forget_gate:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1) * rel_pd
            else:
                score_img = torch.cosine_similarity(lhs_img, rhs_img, 1).unsqueeze(1)

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            score_str = torch.sum(
                (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
                1, keepdim=True
            )
            # beta = 0.95
            return beta * score_str + (1 - beta) * score_img

    def forward(self, x):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 0]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = (1 - self.alpha[(x[:, 1])]) * self.r_embeddings[0](x[:, 2]) + self.alpha[(x[:, 1])] * img_embeddings[
                (x[:, 2])]

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]
            h_r = torch.cat((lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]), dim=-1)

            n = len(h_r)
            ans = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

            for i in range(n):
                i_alpha = self.alpha[(x[i, 1])]
                single_score = h_r[[i], :] @ (
                        (1 - i_alpha) * self.r_embeddings[0].weight + i_alpha * img_embeddings).transpose(0, 1)
                ans = torch.cat((ans, single_score.detach()), 0)

            return (ans), (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                           torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                           torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            lhs = embedding[(x[:, 0])]
            rel = self.r_embeddings[1](x[:, 1])
            rhs = embedding[(x[:, 2])]

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]
            rhs = rhs[:, :self.rank], rhs[:, self.rank:]

            to_score = embedding
            to_score = to_score[:, :self.rank], to_score[:, self.rank:]

            return (
                           (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                           (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
                   ), (
                       torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                       torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                       torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
                   )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            return self.r_embeddings[0].weight.data[
                   chunk_begin:chunk_begin + chunk_size
                   ], img_embeddings
        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            return embedding[
                   chunk_begin:chunk_begin + chunk_size
                   ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        img_embeddings = self.img_vec.mm(self.post_mats)
        if not constant:
            lhs = (1 - self.alpha[(queries[:, 1])]) * self.r_embeddings[0](queries[:, 0]) + self.alpha[
                (queries[:, 1])] * img_embeddings[
                      (queries[:, 0])]
            rel = self.r_embeddings[1](queries[:, 1])

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)



        else:
            embedding = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * img_embeddings
            lhs = embedding[(queries[:, 0])]
            rel = self.r_embeddings[1](queries[:, 1])

            lhs = lhs[:, :self.rank], lhs[:, self.rank:]
            rel = rel[:, :self.rank], rel[:, self.rank:]

            return torch.cat([
                lhs[0] * rel[0] - lhs[1] * rel[1],
                lhs[0] * rel[1] + lhs[1] * rel[0]
            ], 1)

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        计算score
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """

        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0

                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    if not constant:
                        r_embeddings, img_embeddings = self.get_rhs(c_begin, chunk_size)
                        h_r = self.get_queries(these_queries)
                        n = len(h_r)
                        scores_str = torch.ones(0, self.r_embeddings[0].weight.size(0)).cuda()

                        for i in range(n):
                            i_alpha = self.alpha[(these_queries[i, 1])]
                            single_score = h_r[[i], :] @ (
                                    (1 - i_alpha) * self.r_embeddings[0].weight + i_alpha * img_embeddings).transpose(0,
                                                                                                                      1)
                            scores_str = torch.cat((scores_str, single_score.detach()), 0)
                    else:
                        rhs = self.get_rhs(c_begin, chunk_size)
                        q = self.get_queries(these_queries)
                        scores_str = q @ rhs

                    lhs_img = F.normalize(self.img_vec[these_queries[:, 0]], p=2, dim=1)
                    rhs_img = F.normalize(self.img_vec, p=2, dim=1).transpose(0, 1)
                    score_img = lhs_img @ rhs_img  # equation 12
                    if forget_gate:
                        scores = beta * scores_str + (1 - beta) * score_img * self.rel_pd[these_queries[:, 1]]
                    else:
                        scores = beta * scores_str + (1 - beta) * score_img
                    targets = self.score(these_queries)
                    # print(scores)
                    # print(targets)
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks
