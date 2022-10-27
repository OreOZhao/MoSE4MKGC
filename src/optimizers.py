import torch
import tqdm
from torch import nn
from torch import optim

from models import KBCModel
from regularizers import Regularizer


# os.environ['CUDA_VISIBLE_DEVICES'] = device


class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256,
            modality_split=True, fusion_img=True, fusion_label=True, fusion_dscp=True,
            verbose: bool = True
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.modality_split = modality_split
        self.fusion_img = fusion_img
        self.fusion_label = fusion_label
        self.fusion_dscp = fusion_dscp

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]  # 随机数
        # examples shape: torch.Size([966284, 3])
        # actual_examples 是examples的乱序版
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose, ncols=80) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()  # [batch, 3]
                truth = input_batch[:, 2]
                # truth shape: 1000
                if self.modality_split:
                    if self.fusion_img and self.fusion_dscp:
                        preds_str, fac_str, \
                        preds_img, fac_img, \
                        preds_dscp, fac_dscp = self.model.forward(input_batch)
                        # preds shape: 1000 * 14951, N = 1000 batch size, C = 14951 class number
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l_img_fit = loss(preds_img, truth)
                        l_img_reg = self.regularizer.forward(fac_img)
                        l_dscp_fit = loss(preds_dscp, truth)
                        l_dscp_reg = self.regularizer.forward(fac_dscp)
                        l = l_str_fit + l_str_reg + l_img_fit + l_img_reg + l_dscp_fit + l_dscp_reg
                    elif self.fusion_img:
                        preds_str, fac_str, preds_img, fac_img = self.model.forward(input_batch)
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l_img_fit = loss(preds_img, truth)
                        l_img_reg = self.regularizer.forward(fac_img)
                        l = l_str_fit + l_str_reg + l_img_fit + l_img_reg
                    elif self.fusion_dscp:
                        preds_str, fac_str, preds_dscp, fac_dscp = self.model.forward(input_batch)
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l_dscp_fit = loss(preds_dscp, truth)
                        l_dscp_reg = self.regularizer.forward(fac_dscp)
                        l = l_str_fit + l_str_reg + l_dscp_fit + l_dscp_reg
                    else:
                        preds_str, fac_str = self.model.forward(input_batch)
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l = l_str_fit + l_str_reg
                else:
                    preds_str, fac_str = self.model.forward(input_batch)
                    l_str_fit = loss(preds_str, truth)
                    l_str_reg = self.regularizer.forward(fac_str)
                    l = l_str_fit + l_str_reg

                self.optimizer.zero_grad()
                l.backward()
                # try:
                #     for name, weight in self.model.named_parameters():
                #         if weight.requires_grad:
                #             print(weight.grad.mean(), weight.grad.min(), weight.grad.max())
                #             print(name)
                #             if torch.isnan(weight.grad.mean()):
                #                 print(input_batch)
                #
                # except Exception as e:
                #     pass

                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
        return l
