# -*- coding: utf-8 -*-
"""
@Auth: 陈可铨
@File: JSP_TS.py
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
generation_total = data['TS']['GENERATION']['TOTAL']
generation_truncation = data['TS']['GENERATION']['TRUNCATION']
generation_print = data['GA']['GENERATION']['PRINT']

ts = TabuSearch(target, generation_total, generation_truncation, generation_print)
C_history = []
print(f'数据集：{ts.Target}\n'
      f'【理论最优值：{ts.Optimum}】【n={ts.N};m={ts.M}】')
print(f'第0代种群，Cmax={ts.C_op_max}')
C_history.append(ts.C_op_max)
flag = [0, 0, 0]

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
            flag[0] += 1
            ts.GenStuck = 0
            break
        # 候选解中选择非禁忌最优解
        elif move_temp not in ts.TabuList:
            # Ind_temp = ts.GetChromFromT(T_temp, len(ts.Ind_local))
            CriticalPath_temp = ts.GetCriticalPath(T_temp, C_temp_max)
            # 设为当前解
            ts.UpdateLocal(T_temp, C_temp, C_temp_max, Ind_temp, CriticalPath_temp)
            # 更新禁忌表
            ts.TabuList.append(move_temp)
            flag[1] += 1
            ts.GenStuck += 1
            break
        # 既不满足特赦准则，也无非禁忌解，则随机选择
        elif i == len(neighbors) - 1:
            index_random = random.randint(0, len(neighbors) - 1)
            move_temp = neighbors[index_random]

            T_temp, C_temp, C_temp_max, Ind_temp = ts.Perturb(move_temp, ts.T_local)
            # C_temp_max = C_temp.max(initial=0)
            # Ind_temp = ts.GetChromFromT(T_temp, len(ts.Ind_local))
            CriticalPath_temp = ts.GetCriticalPath(T_temp, C_temp_max)
            ts.UpdateLocal(T_temp, C_temp, C_temp_max, Ind_temp, CriticalPath_temp)
            flag[2] += 1
            ts.GenStuck += 1

    if ts.Gen % ts.GenerationPrint == 0:
        print(f'第{ts.Gen}代种群，Cmax={ts.C_op_max}')
    C_history.append(ts.C_op_max)
