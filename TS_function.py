# -*- coding: utf-8 -*-
"""
@Auth: 陈可铨
@File: TS_function.py
@IDE: PyCharm
"""
from General_function import *
from GA_function import *
import random
import copy
import numpy as np


class TabuSearch(JobShopProblem):
    def __init__(self, target, generation_total, generation_truncation, generation_print):
        super().__init__(target, generation_total, generation_truncation, generation_print)

        self.TabuList = ListFixed()  # 禁忌表
        self.L = 0  # 禁忌表长度

        self.T_local = []  # 当前个体Gantt图
        self.C_local = np.zeros((self.N, self.M))  # 当前个体完工时间矩阵
        self.C_local_max = math.inf  # 当前个体最大完工时间
        self.Ind_local = []  # 当前个体染色体
        self.CriticalPath_local = []  # 当前个体关键路径
        self.CriticalPath_op = []  # 最优个体关键路径

        self.InitiateInd()
        self.InitiateTabu()

    def UpdateOp(self, T, C, C_max, Ind, CriticalPath=None):
        """
        更新最优个体
        :param T:
        :param C:
        :param C_max:
        :param Ind:
        :param CriticalPath:
        :return:
        """
        if CriticalPath is None:
            CriticalPath = []
        # if C_max < self.C_op_max:
        self.T_op = T
        self.C_op = C
        self.C_op_max = C_max
        self.Ind_op = Ind
        self.CriticalPath_op = CriticalPath

    def UpdateLocal(self, T, C, C_max, Ind, CriticalPath=None):
        """
        更新当前个体
        :param T:
        :param C:
        :param C_max:
        :param Ind:
        :param CriticalPath:
        :return:
        """
        if CriticalPath is None:
            CriticalPath = []
        # if C_max < self.C_op_max:
        self.T_local = T
        self.C_local = C
        self.C_local_max = C_max
        self.Ind_local = Ind
        self.CriticalPath_local = CriticalPath

    def InitiateInd(self):
        """
        随机生成初始解
        :return:
        """
        self.Ind_local = self.CreateInd()
        self.T_local, self.C_local, self.C_local_max, self.Ind_local = self.Decode(self.Ind_local)
        self.CriticalPath_local = self.GetCriticalPath(self.T_local, self.C_local_max)

        # self.T_op, self.C_op, self.C_op_max, self.Ind_op, self.CriticalPath_op = \
        #     self.T_local, self.C_local, self.C_local_max, self.Ind_local, self.CriticalPath_local
        self.UpdateOp(self.T_local, self.C_local, self.C_local_max, self.Ind_local, self.CriticalPath_local)

    def InitiateTabu(self):
        L = 10 + self.N / self.M
        L_min = int(L)
        if self.N <= 2 * self.M:
            L_max = int(1.4 * L)
        else:
            L_max = int(1.5 * L)
        self.L = random.randint(L_min, L_max)
        self.TabuList = ListFixed(L=self.L)

    def IsOptimum(self):
        """
        根据关键路径判断是否最优
        :param:
        :return:
        """
        flag = True
        # 关键工序在同一台机器上
        if len(self.CriticalPath_op) == 1:
            pass
        else:
            # 关键工序属于同一个工件
            for block in self.CriticalPath_op:
                if len(block['block']) != 1:
                    flag = False
                    break
        if flag:
            print(f'第{self.Gen - 1}代达到理论最优')
        return flag

    def Neighbors(self, critical_path, T, C_max):
        neighbors = []
        for i in range(len(critical_path)):
            block = critical_path[i]['block']
            block_len = len(block)
            if block_len == 1:
                continue

            # 首工序插入工序内部，或内部工序移至首工序前（不包括首尾工序互相移动）
            if i != 0:
                for j in range(1, block_len - 1):
                    move_temp = self.Move(critical_path[i], 0, j, T, C_max, 'forward')
                    if move_temp:
                        neighbors.append(move_temp)
                    # 避免相邻互相重复插入
                    if j != 1:
                        move_temp = self.Move(critical_path[i], 0, j, T, C_max, 'backward')
                        if move_temp:
                            neighbors.append(move_temp)

            # 尾工序插入工序内部，或内部工序移至尾工序后（不包括首尾工序互相移动）
            if i != len(critical_path) - 1:
                for j in range(1, block_len - 1):
                    move_temp = self.Move(critical_path[i], j, block_len - 1, T, C_max, 'forward')
                    if move_temp:
                        neighbors.append(move_temp)
                    # 避免相邻互相重复插入
                    if j != block_len - 2:
                        move_temp = self.Move(critical_path[i], j, block_len - 1, T, C_max, 'backward')
                        if move_temp:
                            neighbors.append(move_temp)

            # 首尾互相插入（必须要满足首尾工序均非本工件的首尾）（若全在同一个机器上则不会进行邻域构造）
            # if i != 0 and i != len(critical_path) - 1:
            move_temp = self.Move(critical_path[i], 0, block_len - 1, T, C_max, 'forward')
            if move_temp:
                neighbors.append(move_temp)
            if block_len > 2:
                move_temp = self.Move(critical_path[i], 0, block_len - 1, T, C_max, 'backward')
                if move_temp:
                    neighbors.append(move_temp)

        neighbors.sort(key=lambda move: move['evaluation'])
        return neighbors

    def Move(self, block, u, v, T, C_max, direction):
        if self.Theorem(block, u, v, T, C_max, direction):
            block_interchange = copy.deepcopy(block)
            block_interchange['head_index'] += u
            block_temp = block_interchange['block'][u:v+1]
            if direction == 'forward':
                block_temp.append(block_temp.pop(0))
            elif direction == 'backward':
                block_temp.insert(0, block_temp.pop())
            else:
                raise ValueError('参数direction应为"forward"或"backward"')
            block_interchange['block'] = block_temp
            block_interchange['evaluation'] = self.Evaluate(block_interchange, T, C_max)

            return block_interchange
        else:
            return False

    def Evaluate(self, block, T, C_max):
        block_len = len(block['block'])
        Values = np.zeros((2, block_len))  # 第1行为L(0, w)，第2行为L(w, n)

        # Q = {l1, l2, ……, v, u} / {v, u, l1, ……, lk}
        # L(0, l1) / L(0, v)
        if block['head_index'] == 0:
            Values[0][0] = self.Value(block, 0, T, C_max, 'job', -1)
        else:
            Values[0][0] = max(self.Value(block, 0, T, C_max, 'job', -1), self.Value(block, 0, T, C_max, 'machine', -1))

        # L(0, w), w = l2, l3, ……, v, u / w = u, l1, l2, ……, lk
        for i in range(1, block_len):
            job_pre, job_process_pre = block['block'][i - 1]
            Values[0][i] = max(self.Value(block, i, T, C_max, 'job', -1), Values[0][i - 1] + self.P[job_pre][job_process_pre])

        # L(u, n) / L(lk, n)
        job_now, job_process_now = block['block'][block_len - 1]
        if block['head_index'] + block_len == len(T[block['machine']]):
            Values[1][-1] = \
                self.P[job_now][job_process_now] + self.Value(block, block_len - 1, T, C_max, 'job', 1)
        else:
            Values[1][-1] = \
                self.P[job_now][job_process_now] + \
                max(self.Value(block, block_len - 1, T, C_max, 'job', 1), self.Value(block, block_len - 1, T, C_max, 'machine', 1))

        # L(w, n), w = l1, l2, l3, ……, v / w = v, u, l1, ……,lk-1
        for i in range(2, block_len + 1):
            job_now, job_process_now = block['block'][block_len - i]
            Values[1][block_len - i] = \
                self.P[job_now][job_process_now] + \
                max(self.Value(block, block_len - i, T, C_max, 'job', 1), Values[1][block_len - i + 1])

        # 相加取最大
        return np.max(Values.sum(axis=0))

    def Value(self, block, block_index, T, C_max, adjacent, direction):
        process_now = block['block'][block_index]
        if adjacent == 'job':
            job_now, job_process_now = process_now
            try:
                machine_d = self.J[job_now][job_process_now + direction] - 1
            except IndexError:
                return 0
            for process in T[machine_d]:
                if process[1] == job_now:
                    if direction == -1:
                        return process[-1]
                    elif direction == 1:
                        return C_max - process[0]
        elif adjacent == 'machine':
            if direction == -1:
                return T[block['machine']][block['head_index'] - 1][-1]
            elif direction == 1:
                return C_max - T[block['machine']][block['head_index'] + len(block['block'])][0]

    def Perturb(self, block, T_pre):
        # 先得到机器上的工件加工顺序M
        M = []
        for each_machine in T_pre:
            M.append([])
            for each_job in each_machine:
                M[-1].append(each_job[1:3])

        block_temp = copy.deepcopy(block['block'])
        block_len = len(block_temp)
        M[block['machine']][block['head_index']:block['head_index']+block_len] = block_temp

        # 通过M得到临时chrom
        k_jobs = np.zeros(self.N, dtype=int)
        chrom = []
        while True:
            flag = 0
            for machine in range(self.M):
                if M[machine]:
                    job, job_process = M[machine][0]
                    if job_process == k_jobs[job]:
                        chrom.append(job)
                        k_jobs[job] += 1
                        M[machine].pop(0)
                else:
                    flag += 1
            if flag == self.M:
                break

        return self.Decode(chrom)

    def Theorem(self, block, u, v, T, C_max, direction):
        L1, L2 = 0, 0
        # Theorem 1
        if direction == 'forward':
            # L(v, n)
            L1 = C_max - T[block['machine']][block['head_index'] + v][0]
            # L(Js[u], n)
            L2 = self.Value(block, u, T, C_max, 'job', 1)
        # Theorem 2
        elif direction == 'backward':
            # L(0, u) + p_u
            L1 = T[block['machine']][block['head_index'] + u][-1]
            # L(0, Jp[v]) + p_Jp[v]
            L2 = self.Value(block, v, T, C_max, 'job', -1)

        return L1 >= L2

    def TerminationCriterion(self):
        """
        终止准则
        :return:
        """
        if self.C_op_max == self.Optimum:
            print(f'第{self.Gen - 1}代达到理论最优')
            return True
        elif self.GenerationTruncation == self.GenStuck:
            print(f'第{self.Gen - 1}代达到局部最优')
            print(f'--------------------------------------')
            self.InitiateInd()
            self.InitiateTabu()
            return False
        elif self.Gen == self.GenerationTotal:
            return True
        else:
            return False


class ListFixed(list):
    def __init__(self, L=10):
        super().__init__()

        self.L = L

    def append(self, __object) -> None:
        super().append(__object)

        if len(self) > self.L:
            self.pop(0)
