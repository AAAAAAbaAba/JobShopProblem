# -*- coding: utf-8 -*-
"""
@Auth: 陈可铨
@File: JSP_GA.py
@IDE: PyCharm
"""

from General_function import *
from GA_function import *
from TS_function import *
import random
import numpy as np
import yaml


with open(file='DATA.yaml', mode='r', encoding='UTF-8') as f:
    data = yaml.safe_load(f)
target = data['TARGET']
generation_total = data['GA']['GENERATION']['TOTAL']
generation_truncation = data['GA']['GENERATION']['TRUNCATION']
generation_print = data['GA']['GENERATION']['PRINT']
population_size = data['GA']['POPULATION_SIZE']
k_fit = data['GA']['K_FIT']
b_fit = data['GA']['B_FIT']
p_cross = data['GA']['CROSS']['P']
n_cross = data['GA']['CROSS']['N']
p_mutation = data['GA']['MUTATION']['P']
l_mutation = data['GA']['MUTATION']['L']

ga = GeneticAlgorithm(target, population_size, generation_total, generation_truncation, generation_print,
                      k_fit, b_fit, p_cross, n_cross, p_mutation, l_mutation)
C_history = []
print(f'数据集：{ga.Target}\n'
      f'【理论最优值：{ga.Optimum}】【n={ga.N};m={ga.M}】')
print(f'第0代种群，Cmax={ga.C_op_max}')
C_history.append(ga.C_op_max)

while True:
    ga.Gen += 1
    if ga.TerminationCriterion():
        break

    # # 选择
    # ga.ProportionalSelect(n_select=ga.PopSize, cover=True, keep_best=1)
    #
    # # 与最优个体进行交叉
    # cross_index1 = np.array(ga.Fit).argmax()
    # for _ in range(int(ga.PopSize * ga.P_Cross)):
    #     while True:
    #         cross_index2 = random.randint(0, ga.PopSize - 1)
    #         if cross_index1 != cross_index2:
    #             break
    #     ga.Cross(cross_index1, cross_index2)
    #
    # # 最优个体不进行突变
    # for _ in range(int(ga.PopSize * ga.P_Mutation)):
    #     while True:
    #         mutation_index = random.randint(0, ga.PopSize - 1)
    #         if cross_index1 != mutation_index:
    #             break
    #     ga.Mutation(mutation_index)

    # 轮盘赌选择后随机交叉
    Parent_index1, Parent_index2 = ga.ProportionalSelect(keep_best=0, n_select=2)

    if random.random() <= ga.P_Cross:
        ga.Cross(Parent_index2, Parent_index1)
    # 随机突变
    elif random.random() <= ga.P_Mutation:
        ga.Mutation(Parent_index1)
        ga.Mutation(Parent_index2)

    if ga.Gen % ga.GenerationPrint == 0:
        print(f'第{ga.Gen}代种群，Cmax={ga.C_op_max}')
    if ga.C_op_max == C_history[-1]:
        ga.GenStuck += 1
    else:
        ga.GenStuck = 0
    C_history.append(ga.C_op_max)
