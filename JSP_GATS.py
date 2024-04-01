# -*- coding: utf-8 -*-
"""
@Auth: 陈可铨
@File: JSP_GATS.py
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

generation_total_ts = data['TS']['GENERATION']['TOTAL']
generation_truncation_ts = data['TS']['GENERATION']['TRUNCATION']
generation_print_ts = data['TS']['GENERATION']['PRINT']

ga = GeneticAlgorithm(target, population_size, generation_total, generation_truncation, generation_print,
                      k_fit, b_fit, p_sel, p_cross, n_cross, p_mutation, l_mutation)
ts = TabuSearch(target, generation_total_ts, generation_truncation_ts, generation_print_ts)
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

    # 按选择策略选取下一代种群
    pop_temp = [ga.Ind_op]  # 最佳个体保存
    fit_temp = [ga.CalculateFit(ga.C_op_max)]
    while len(pop_temp) < ga.PopSize:
        # 选择操作
        Parent1, Parent1_fit = ga.TournamentSelect()  # 锦标赛选择
        while True:
            Parent2, Parent2_fit = ga.TournamentSelect()
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

    # 禁忌搜索优化下一代种群个体
    for index in range(ga.PopSize):
        # 重置禁忌搜索框架
        ts.InitiateTabu()
        ts.InitiateInd(pop_temp[index])
        # 禁忌搜索
        while True:
            ts.Gen += 1
            # 判断是否满足终止准则
            if ts.TerminationCriterion() or ts.IsOptimum():
                break

            # 由邻域结构产生当前解的候选解
            neighbors = ts.Neighbors(ts.CriticalPath_local, ts.T_local, ts.C_local_max)
            for i in range(len(neighbors)):
                move_temp = neighbors[i]
                T_temp, C_temp, C_temp_max, Ind_temp = ts.Perturb(move_temp, ts.T_local)
                # 判断是否满足特赦准则
                if C_temp_max < ts.C_op_max:
                    CriticalPath_temp = ts.GetCriticalPath(T_temp, C_temp_max)
                    # 若满足则设为当前解和最优解
                    ts.UpdateLocal(T_temp, C_temp, C_temp_max, Ind_temp, CriticalPath_temp)
                    ts.UpdateOp(T_temp, C_temp, C_temp_max, Ind_temp, CriticalPath_temp)
                    # 更新禁忌表
                    ts.TabuList.append(move_temp)
                    ts.GenStuck = 0
                    break
                # 候选解中选择非禁忌最优解
                elif move_temp not in ts.TabuList:
                    CriticalPath_temp = ts.GetCriticalPath(T_temp, C_temp_max)
                    # 设为当前解
                    ts.UpdateLocal(T_temp, C_temp, C_temp_max, Ind_temp, CriticalPath_temp)
                    # 更新禁忌表
                    ts.TabuList.append(move_temp)
                    ts.GenStuck += 1
                    break
                # 既不满足特赦准则，也无非禁忌解，则随机选择
                elif i == len(neighbors) - 1:
                    index_random = random.randint(0, len(neighbors) - 1)
                    move_temp = neighbors[index_random]

                    T_temp, C_temp, C_temp_max, Ind_temp = ts.Perturb(move_temp, ts.T_local)
                    CriticalPath_temp = ts.GetCriticalPath(T_temp, C_temp_max)
                    ts.UpdateLocal(T_temp, C_temp, C_temp_max, Ind_temp, CriticalPath_temp)
                    ts.GenStuck += 1

        pop_temp[index] = ts.Ind_op
        fit_temp[index] = ga.CalculateFit(ts.C_op_max)
        if ts.TerminationCriterion() == 1:
            break
    if ts.TerminationCriterion() == 1:
        ga.T_op, ga.C_op, ga.C_op_max, ga.Ind_op = ga.Decode(ts.Ind_op)
        ga.TerminationCriterion()
        break

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
