# -*- coding: utf-8 -*-
"""
@Auth: 陈可铨
@File: JSP.py
@IDE: PyCharm
"""

from General_function import *
from GA_function import *
import random
import numpy as np


if __name__ == '__main__':
    target = 'la02'
    C_history = []
    J, P, jobs_num, machines_num, optimum = load_data(target)
    ga = GeneticAlgorithm(J, P, jobs_num, machines_num, optimum)

    print(f'数据集{target}理论最优值：{ga.Optimum}')
    print(f'第0代种群，Cmax={ga.C_max}')
    C_history.append(ga.C_max)
    for gen in range(1, 501):
        cross_index1 = np.array(ga.Fit).argmax()
        for cross_index2 in range(ga.PopSize):
            if (random.random() <= ga.P_Cross) and (cross_index1 != cross_index2):
                ga.Cross(cross_index1, cross_index2)

        for mutation_index in range(ga.PopSize):
            if (random.random() <= ga.P_Mutation) and (cross_index1 != mutation_index):
                ga.Mutation(mutation_index)

        if gen % 10 == 0:
            print(f'第{gen}代种群，Cmax={ga.C_max}')
        C_history.append(ga.C_max)

        if ga.C_max == ga.Optimum:
            break

    draw_Gantt(ga.T_op)

    plt.plot(C_history)
    plt.show()
