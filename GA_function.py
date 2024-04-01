# -*- coding: utf-8 -*-
"""
@Auth: 陈可铨
@File: GA_function.py
@IDE: PyCharm
"""
from General_function import *
import numpy as np
import copy
import math
import random
import itertools


class GeneticAlgorithm(JobShopProblem):
    def __init__(self, target, population_size=500, generation_total=50000, generation_truncation=15000,
                 generation_print=500, k_fit=100, b_fit=0, p_sel=0.8, p_cross=0.8, n_cross=2, p_mutation=0.01, l_mutation=4):
        super().__init__(target, generation_total, generation_truncation, generation_print)

        self.PopSize = population_size if population_size % 2 == 1 else population_size + 1  # 种群规模
        self.Pop = []  # 种群
        self.Fit = []  # 适应度列表

        # 微调参数
        self.K_Fit = k_fit
        self.B_Fit = b_fit
        self.P_SEL = p_sel
        self.P_Cross = p_cross
        self.N_Cross = n_cross
        self.P_Mutation = p_mutation
        self.L_Mutation = l_mutation

        self.InitiatePop()

    def InitiatePop(self):
        """
        种群初始化
        :return:
        """
        self.C_op_max = math.inf
        self.Pop = []
        self.Fit = []
        for _ in range(self.PopSize):
            self.Pop.append(self.CreateInd())

        for i in range(self.PopSize):
            T_temp, C_temp, C_temp_max, self.Pop[i] = self.Decode(self.Pop[i])
            self.UpdateOp(T_temp, C_temp, C_temp_max, self.Pop[i])
            self.Fit.append(self.CalculateFit(C_temp_max))

    def CalculateFit(self, p):
        """
        计算适应度
        :param p:
        :return:
        """
        return self.K_Fit / (p + self.B_Fit)

    def Cross(self, Parent1, Parent1_fit, Parent2, Parent2_fit):
        """
        POX交叉，从最优父代和全部子代中选择2个进入下一代
        :param Parent1:
        :param Parent1_fit:
        :param Parent2:
        :param Parent2_fit:
        :return:
        """
        population_temp = []
        fitness_temp = []

        # 最优父代加入临时种群
        if Parent1_fit >= Parent2_fit:
            population_temp.append(Parent1)
            fitness_temp.append(Parent1_fit)
        else:
            population_temp.append(Parent2)
            fitness_temp.append(Parent2_fit)

        for _ in range(self.N_Cross):
            # 1.随机划分工件集
            Jobs = list(range(self.N))
            random.shuffle(Jobs)
            len1 = random.randint(1, self.N - 1)  # 保证工件集J1、J2
            # Jobs1 = Jobs[:len1]
            Jobs2 = Jobs[len1:]

            # 2.P1在J1中的复制到C1，P2在J1中的复制到C2，保留位置；P1在J2中的复制到C2，P2在J2中的复制到C1，保留顺序
            Child1 = copy.copy(Parent1)
            Child2 = copy.copy(Parent2)
            Parent1_in_Jobs2 = list(filter(lambda x: x in Jobs2, Parent1))
            Parent2_in_Jobs2 = list(filter(lambda x: x in Jobs2, Parent2))
            for i in range(len(Child1)):
                if Child1[i] in Jobs2:
                    Child1[i] = Parent2_in_Jobs2.pop(0)
                if Child2[i] in Jobs2:
                    Child2[i] = Parent1_in_Jobs2.pop(0)

            # 3.子代加入临时种群
            T1, C1, C1_max, Child1 = self.Decode(Child1)
            population_temp.append(Child1)
            # self.UpdateOp(T1, C1, C1_max, Child1)
            fitness_temp.append(self.CalculateFit(C1_max))
            T2, C2, C2_max, Child2 = self.Decode(Child2)
            population_temp.append(Child2)
            # self.UpdateOp(T2, C2, C2_max, Child2)
            fitness_temp.append(self.CalculateFit(C2_max))

        # 选择临时种群中适应度最高的2个个体进入下一代种群
        index1 = np.array(fitness_temp).argmax()
        fitness1 = fitness_temp[index1]
        fitness_temp[index1] = 0
        index2 = np.array(fitness_temp).argmax()
        fitness2 = fitness_temp[index2]
        return population_temp[index1], fitness1, population_temp[index2], fitness2
        # index_new = np.array(fitness_temp).argsort()[-2:]
        # self.Pop[Parent1_index] = population_temp[index_new[0]]
        # self.Fit[Parent1_index] = fitness_temp[index_new[0]]
        # self.Pop[Parent2_index] = population_temp[index_new[1]]
        # self.Fit[Parent2_index] = fitness_temp[index_new[1]]

    def Mutation(self, Parent):
        """
        基于邻域搜索的变异
        :param Parent:
        :return:
        """
        index_mutation = []
        jobs_mutation = []
        population_temp = []
        fitness_temp = []

        # 1.选取Lambda个不同基因进行变异
        while len(index_mutation) < self.L_Mutation:
            index_temp = random.randint(0, self.N * self.M - 1)
            if Parent[index_temp] not in jobs_mutation:
                jobs_mutation.append(Parent[index_temp])
                index_mutation.append(index_temp)

        # 2.取变异基因的全排列作为邻域进行变异
        jobs_permutations = itertools.permutations(jobs_mutation)
        next(jobs_permutations)  # 不包括原基因
        for each_jobs in jobs_permutations:
            Child_temp = copy.copy(Parent)
            for i in range(len(each_jobs)):
                Child_temp[index_mutation[i]] = each_jobs[i]
            T_temp, C_temp, C_temp_max, Child_temp = self.Decode(Child_temp)
            population_temp.append(Child_temp)
            # self.UpdateOp(T_temp, C_temp, C_temp_max, Child_temp)
            fitness_temp.append(self.CalculateFit(C_temp_max))

        # 选择临时种群中适应度最高的个体进入下一代种群
        index_new = np.array(fitness_temp).argmax()
        return population_temp[index_new], fitness_temp[index_new]
        # self.Pop[Parent_index] = population_temp[index_new]
        # self.Fit[Parent_index] = fitness_temp[index_new]

    def ProportionalSelect(self):
        """
        比例选择（即轮盘赌选择）
        :return:
        """
        # 构造比例列表
        fitness_proportion = np.array(self.Fit)
        fitness_proportion = np.cumsum(fitness_proportion / np.sum(fitness_proportion))
        # 抽取随机数
        random_parameter = random.random()
        for index in range(self.PopSize):
            if random_parameter <= fitness_proportion[index]:
                return self.Pop[index], self.Fit[index]
        # while True
        #     random_list = []
        #     for _ in range(n_select - keep_best):
        #         random_list.append(random.random())
        #
        #     index_temp = 0
        #     index_list = []
        #     selection_list = np.sort(np.concatenate((fitness_proportion, np.array(random_list)), axis=0))
        #     for each_number in selection_list:
        #         if each_number < fitness_proportion[index_temp]:
        #             population_temp.append(self.Pop[index_temp])
        #             fitness_temp.append(self.Fit[index_temp])
        #             index_list.append(index_temp)
        #         elif each_number == fitness_proportion[index_temp]:
        #             index_temp += 1
        #
        #     if cover_flag:
        #         self.Pop = population_temp
        #         self.Fit = fitness_temp
        #         break
        #     elif len(index_list) == len(set(index_list)):
        #         break
        # return index_list

    def TournamentSelect(self):
        """
        锦标赛选择
        :return:
        """
        index1 = random.randint(0, self.PopSize - 1)
        index2 = random.randint(0, self.PopSize - 1)
        if random.random() <= self.P_SEL:
            if self.Fit[index1] >= self.Fit[index2]:
                return self.Pop[index1], self.Fit[index1]
            else:
                return self.Pop[index2], self.Fit[index2]
        else:
            if self.Fit[index2] >= self.Fit[index1]:
                return self.Pop[index1], self.Fit[index1]
            else:
                return self.Pop[index2], self.Fit[index2]

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
            self.InitiatePop()
            return False
        elif self.Gen == self.GenerationTotal:
            return True
        else:
            return False
