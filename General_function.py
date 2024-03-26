# -*- coding: utf-8 -*-
"""
@Auth: 陈可铨
@File: General_function.py
@IDE: PyCharm
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import math


class JobShopProblem:
    def __init__(self, target):
        self.Target = target
        self.J = np.array([])  # 加工机器矩阵
        self.P = np.array([])  # 加工时间矩阵
        self.N = 0  # 机器数
        self.M = 0  # 理论最优目标值
        self.Optimum = 0

        self.load_data()
        self.J_R = np.flip(self.J, axis=1)
        self.P_R = np.flip(self.P, axis=1)

    def load_data(self):
        with open(os.path.join(os.path.abspath('.'), '../JSPLIB/instances.json'), 'r') as f:
            data = json.load(f)

        instance = [inst for inst in data if inst['name'] == self.Target]
        if len(instance) == 0:
            raise Exception(f'There is no instance named {self.Target}')

        instance = instance[0]
        path = os.path.abspath(os.path.join(os.path.abspath('.'), f'../JSPLIB/{instance["path"]}'))
        self.Optimum = instance['optimum']

        if self.Optimum is None:
            if instance['bounds'] is None:
                self.Optimum = "nan"
            else:
                self.Optimum = instance['bounds']['lower']

        # 读取Target文件
        with open(path, 'r') as file:
            lines = file.readlines()
        while lines[0][0] == '#':
            lines.pop(0)

        # 读取工件数、机器数
        self.N, self.M = map(int, lines[0].split())

        # 读取J、P
        self.J = np.zeros((self.N, len(lines[1].split()) // 2), dtype=int)
        self.P = np.zeros((self.N, len(lines[1].split()) // 2), dtype=int)
        for i in range(1, len(lines)):
            data = list(map(int, lines[i].split()))
            for j in range(len(data)):
                if j % 2 == 0:
                    self.J[i - 1][j // 2] = data[j] + 1
                else:
                    self.P[i - 1][j // 2] = data[j]

    def Decode(self, chrom):
        """
        反转+贪婪插入式解码获得全活动调度
        :param chrom: 待解码的染色体序列
        :return T: Gantt图矩阵([起始时间, 工件号, 工序号, 结束时间])
        :return C: 完工时间矩阵
        """
        chrom_len = len(chrom)

        C = np.zeros((self.N, self.M))
        T = [[[0]] for _ in range(self.M)]
        k_jobs = np.zeros(self.N, dtype=int)
        # 反转染色体与J、P求解
        for job in reversed(chrom):
            machine = self.J_R[job, k_jobs[job]] - 1
            process_time = self.P_R[job, k_jobs[job]]
            finish_time_last_job = C[job, k_jobs[job] - 1]

            # 寻找空闲时段插入
            start_time = max(finish_time_last_job, T[machine][-1][-1])
            insert_index = len(T[machine])
            for i in range(1, len(T[machine])):
                gap_start = max(finish_time_last_job, T[machine][i - 1][-1])
                gap_end = T[machine][i][0]
                if gap_end - gap_start >= process_time:
                    start_time = gap_start
                    insert_index = i
                    break
            end_time = start_time + process_time
            C[job, k_jobs[job]] = end_time
            T[machine].insert(insert_index, [start_time, job, k_jobs[job], end_time])
            k_jobs[job] += 1
        for i in range(len(T)):
            T[i].pop(0)

        # Gantt图反推全活动调度染色体
        chrom = []
        k_machines = np.zeros(self.M, dtype=int)
        start_time_list = np.array([T[machine][0][0] for machine in range(len(T))])
        while len(chrom) < chrom_len:
            machine_early = np.argmin(start_time_list)
            chrom.append(T[machine_early][k_machines[machine_early]][1])
            k_machines[machine_early] += 1
            try:
                start_time_list[machine_early] = T[machine_early][k_machines[machine_early]][0]
            except IndexError:
                start_time_list[machine_early] = math.inf

        C = np.zeros((self.N, self.M))
        T = [[[0]] for _ in range(self.M)]
        k_jobs = np.zeros(self.N, dtype=int)
        # 全活动调度染色体与J、P求解
        chrom.reverse()
        for job in chrom:
            machine = self.J[job, k_jobs[job]] - 1
            process_time = self.P[job, k_jobs[job]]
            finish_time_last_job = C[job, k_jobs[job] - 1]

            # 寻找空闲时段插入
            start_time = max(finish_time_last_job, T[machine][-1][-1])
            insert_index = len(T[machine])
            for i in range(1, len(T[machine])):
                gap_start = max(finish_time_last_job, T[machine][i - 1][-1])
                gap_end = T[machine][i][0]
                if gap_end - gap_start >= process_time:
                    start_time = gap_start
                    insert_index = i
                    break
            end_time = start_time + process_time
            C[job, k_jobs[job]] = end_time
            T[machine].insert(insert_index, [start_time, job, k_jobs[job], end_time])
            k_jobs[job] += 1
        for i in range(len(T)):
            T[i].pop(0)

        return T, C, chrom


def draw_Gantt(timelist):
    """
    绘制
    :param timelist:
    :return:
    """
    T = timelist.copy()

    # 创建基础画布
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots(figsize=(20, 12))

    # 每个工件一个颜色
    color_map = {}
    for machine in T:
        for task_data in machine[:]:
            job = task_data[1]
            if job not in color_map:
                color_map[job] = (random.random(), random.random(), random.random())

    # 遍历机器
    for machine_idx, machine_schedule in enumerate(T):
        for task_data in machine_schedule[:]:
            start_time, job, operation, end_time = task_data
            color = color_map[job]
            # 绘制Gantt图
            ax.barh(machine_idx, end_time - start_time, left=start_time, height=.5, color=color)
            #
            label = f'{job}-{operation}'
            ax.text((start_time + end_time) / 2, machine_idx, label, ha='center', va='center', color='black', fontsize=12)

    ax.set_yticks(range(len(T)))
    ax.set_yticklabels(f'MACHINE{i + 1}' for i in range(len(T)))
    plt.xlabel('Time')
    plt.title('JSP Gantt')
    legend_list = []
    for job, color in dict(sorted(color_map.items(), key=lambda x: x[0], reverse=False)).items():
        legend_list.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f'JOB{job}'))
    plt.legend(handles=legend_list, title='job')

    plt.show()
