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
p_sel = data['GA']['P_SEL']
p_cross = data['GA']['CROSS']['P']
n_cross = data['GA']['CROSS']['N']
p_mutation = data['GA']['MUTATION']['P']
l_mutation = data['GA']['MUTATION']['L']

ga = GeneticAlgorithm(target, population_size, generation_total, generation_truncation, generation_print,
                      k_fit, b_fit, p_sel, p_cross, n_cross, p_mutation, l_mutation)
C_history = []
print(f'数据集：{ga.Target}\n'
      f'【理论最优值：{ga.Optimum}】【n={ga.N};m={ga.M}】')
print(f'第0代种群，Cmax={ga.C_op_max}')
C_history.append(ga.C_op_max)

while True:
    ga.Gen += 1
    # 判断是否满足终止准则
    if ga.TerminationCriterion():
        break

    pop_temp = [ga.Ind_op]  # 最佳个体保存
    fit_temp = [ga.CalculateFit(ga.C_op_max)]
    while len(pop_temp) < ga.PopSize:
        # 选择操作
        Parent1, Parent1_fit = ga.ProportionalSelect()  # 比例选择（轮盘赌选择）
        while True:
            Parent2, Parent2_fit = ga.ProportionalSelect()
            if Parent1 != Parent2:
                break
        # 交叉
        if random.random() <= ga.P_Cross:
            Child1, Child1_fit, Child2, Child2_fit = ga.Cross(Parent1, Parent1_fit, Parent2, Parent2_fit)
        else:
            Child1, Child1_fit, Child2, Child2_fit = Parent1, Parent1_fit, Parent2, Parent2_fit
        # 突变
        if random.random() <= ga.P_Mutation:
            Child1, Child1_fit = ga.Mutation(Child1)
            Child2, Child2_fit = ga.Mutation(Child2)
        pop_temp.append(Child1)
        fit_temp.append(Child1_fit)
        pop_temp.append(Child2)
        fit_temp.append(Child2_fit)

    ga.T_op, ga.C_op, ga.C_op_max, ga.Ind_op = ga.Decode(pop_temp[np.array(fit_temp).argmax()])
    ga.Pop = pop_temp
    ga.Fit = fit_temp

    if ga.Gen % ga.GenerationPrint == 0:
        print(f'第{ga.Gen}代种群，Cmax={ga.C_op_max}')
    if ga.C_op_max == C_history[-1]:
        ga.GenStuck += 1
    else:
        ga.GenStuck = 0
    C_history.append(ga.C_op_max)
