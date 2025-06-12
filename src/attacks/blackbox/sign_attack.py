"""
Implements SignHunter
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as ch

from attacks.blackbox.black_box_attack import BlackBoxAttack
from utils.compute_fcts import lp_step, sign


class SignAttack(BlackBoxAttack):
    """
    SignHunter
    """

    def __init__(self, max_loss_queries, epsilon, p, fd_eta, lb, ub):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_crit_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub)
        self.fd_eta = fd_eta
        self.best_est_deriv = None
        self.xo_t = None
        self.sgn_t = None
        self.h = 0
        self.i = 0
    #
    # def _suggest(self, xs_t, loss_fct, metric_fct):
    #     _shape = list(xs_t.shape)
    #     dim = np.prod(_shape[1:])
    #     # additional queries at the start
    #     add_queries = 0
    #     if self.is_new_batch:
    #         self.xo_t = xs_t.clone()
    #         self.h = 0
    #         self.i = 0
    #     if self.i == 0 and self.h == 0:
    #         self.sgn_t = sign(ch.ones(_shape[0], dim))
    #         fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
    #         bxs_t = self.xo_t
    #         est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / self.epsilon
    #         self.best_est_deriv = est_deriv
    #         add_queries = 3  # because of bxs_t and the 2 evaluations in the i=0, h=0, case.
    #     chunk_len = np.ceil(dim / (2 ** self.h)).astype(int)
    #     istart = self.i * chunk_len
    #     iend = min(dim, (self.i + 1) * chunk_len)
    #     self.sgn_t[:, istart:iend] *= - 1.
    #     fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
    #     bxs_t = self.xo_t
    #     est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / self.epsilon
    #     self.sgn_t[[i for i, val in enumerate(est_deriv < self.best_est_deriv) if val], istart: iend] *= -1.
    #     self.best_est_deriv = (est_deriv >= self.best_est_deriv) * est_deriv + (
    #             est_deriv < self.best_est_deriv) * self.best_est_deriv
    #     # compute the cosine similarity
    #     cos_sims, ham_sims = metric_fct(self.xo_t.cpu().numpy(), self.sgn_t.cpu().numpy())
    #     # perform the step
    #     new_xs = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
    #     # update i and h for next iteration
    #     self.i += 1
    #     if self.i == 2 ** self.h or iend == dim:
    #         self.h += 1
    #         self.i = 0
    #         # if h is exhausted, set xo_t to be xs_t
    #         if self.h == np.ceil(np.log2(dim)).astype(int) + 1:
    #             self.xo_t = xs_t.clone()
    #             self.h = 0
    #             print("new change")
    #     return new_xs, np.ones(_shape[0]) + add_queries, cos_sims, ham_sims

    def _suggest(self, xs_t, loss_fct, metric_fct):
        _shape = list(xs_t.shape)  # 获取输入张量xs_t的形状，转换为列表
        dim = np.prod(_shape[1:])  # 计算除第一个维度外，其余维度元素的乘积，得到样本特征维度

        # 额外查询数，初始化为0
        add_queries = 0

        # 如果是新一批数据，初始化相关状态变量
        if self.is_new_batch:
            self.xo_t = xs_t.clone()  # 复制输入样本，作为基准样本xo_t
            self.h = 0  # 初始化分块层级h为0
            self.i = 0  # 初始化当前分块索引i为0

        # 第一次迭代且分块层级为0时，初始化符号向量sgn_t并估计初始导数
        if self.i == 0 and self.h == 0:
            self.sgn_t = sign(ch.ones(_shape[0], dim))  # 初始化符号向量，全为正号，形状为（样本数，特征维度）
            fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)  # 计算正向扰动样本
            bxs_t = self.xo_t  # 原始基准样本
            # 估计导数：通过正向扰动样本和基准样本的损失差值除以扰动幅度epsilon
            est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / self.epsilon
            self.best_est_deriv = est_deriv  # 保存当前最佳估计导数
            add_queries = 3  # 统计查询次数：bxs_t和两次loss评估的查询数量

        # 计算当前分块长度，按2的h次幂划分
        chunk_len = np.ceil(dim / (2 ** self.h)).astype(int)
        istart = self.i * chunk_len  # 当前分块起始索引
        iend = min(dim, (self.i + 1) * chunk_len)  # 当前分块结束索引（不超过总维度）

        # 将当前分块的符号翻转（乘以-1）
        self.sgn_t[:, istart:iend] *= - 1.

        # 计算翻转符号后的扰动样本和基准样本的损失差估计导数
        fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
        bxs_t = self.xo_t
        est_deriv = (loss_fct(fxs_t.cpu().numpy()) - loss_fct(bxs_t.cpu().numpy())) / self.epsilon

        # 对于导数估计更差的样本，将对应分块符号再翻转回去，保持较优符号方向
        self.sgn_t[[i for i, val in enumerate(est_deriv < self.best_est_deriv) if val], istart: iend] *= -1.

        # 更新最佳估计导数，取较优者
        self.best_est_deriv = (est_deriv >= self.best_est_deriv) * est_deriv + (
                est_deriv < self.best_est_deriv) * self.best_est_deriv

        # 计算当前样本和符号向量的余弦相似度及汉明距离（通过传入的metric_fct函数）
        cos_sims, ham_sims = metric_fct(self.xo_t.cpu().numpy(), self.sgn_t.cpu().numpy())

        # 根据当前符号向量执行一步扰动，生成新的对抗样本
        new_xs = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)

        # 更新分块索引i，准备下次迭代
        self.i += 1

        # 若当前层级分块遍历完成或到达维度末尾，层级h加1，分块索引i归零
        if self.i == 2 ** self.h or iend == dim:
            self.h += 1
            self.i = 0
            # 如果层级h超过最大分块层数，重置基准样本为当前输入，并重置层级h
            if self.h == np.ceil(np.log2(dim)).astype(int) + 1:
                self.xo_t = xs_t.clone()
                self.h = 0
                print("new change")  # 标记开始新一轮搜索

        # 返回新生成的对抗样本，查询次数（包含额外查询），余弦相似度，汉明距离
        return new_xs, np.ones(_shape[0]) + add_queries, cos_sims, ham_sims

    def get_gs(self):
        """
        return the current estimated of the gradient sign
        :return:
        """
        return self.sgn_t

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_crit_queries": "inf" if np.isinf(self.max_crit_queries) else self.max_crit_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "fd_eta": self.fd_eta,
            "attack_name": self.__class__.__name__
        }
