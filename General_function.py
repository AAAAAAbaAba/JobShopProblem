# -*- coding: utf-8 -*-
"""
@Auth: 陈可铨
@File: General_function.py
@IDE: PyCharm
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import math


class JobShopProblem:
    def __init__(self, target, generation_total, generation_truncation, generation_print):
        self.Target = target
        self.J = np.array([])  # 加工机器矩阵
        self.P = np.array([])  # 加工时间矩阵
        self.N = 0  # 机器数
        self.M = 0  # 理论最优目标值
        self.Optimum = 0

        if self.Target != 'CUSTOM':
            self.load_data()
        self.J_R = np.flip(self.J, axis=1)
        self.P_R = np.flip(self.P, axis=1)

        self.T_op = []  # 最优个体Gantt图
        self.C_op = np.zeros((self.N, self.M))  # 最优个体完工时间矩阵
        self.C_op_max = math.inf  # 最优个体最大完工时间
        self.Ind_op = []  # 最优个体染色体

        self.GenerationTotal = generation_total  # 终止迭代代数
        self.GenerationTruncation = generation_truncation  # 局部截止迭代代数
        self.GenerationPrint = generation_print  # 打印结果迭代代数
        self.Gen = 0  # 迭代代数
        self.GenStuck = 0  # 局部迭代代数

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
        :return C_max: 最大完工时间
        :return chrom: 全活动调度染色体序列
        """
        # 反转染色体求解反转Gantt图
        T_reverse, _ = self.GetTCFromChrom(chrom, reverse=True)
        # 反转Gantt图反推全活动调度染色体
        chrom = self.GetChromFromT(T_reverse, len(chrom))
        chrom.reverse()
        # 全活动调度染色体求解Gantt图
        T, C = self.GetTCFromChrom(chrom, reverse=False)
        C_max = C.max(initial=0)

        return T, C, C_max, chrom

    def GetTCFromChrom(self, chrom, reverse):
        """
        通过染色体求解T、C
        :param chrom:
        :param reverse:
        :return:
        """
        T = [[[0]] for _ in range(self.M)]
        C = np.zeros((self.N, self.M))
        k_jobs = np.zeros(self.N, dtype=int)
        if reverse:
            J = self.J_R
            P = self.P_R
            chrom = list(reversed(chrom))
        else:
            J = self.J
            P = self.P

        for job in chrom:
            machine = J[job, k_jobs[job]] - 1
            process_time = P[job, k_jobs[job]]
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

        return T, C

    def GetChromFromT(self, T, chrom_len):
        """
        通过T求解染色体chrom
        :param T:
        :param chrom_len:
        :return:
        """
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

        return chrom

    def GetCriticalPath(self, T, C_max):
        """
        求解关键路径（外层），考虑到多个机器可能同时达到最大完成时间
        :param T:
        :param C_max:
        :return critical_path: 关键路径 [{'machine': 机器, 'head_index': 块首位置, 'block'： 关键块}]
        """
        critical_path = []
        for machine in range(len(T)):
            if T[machine][-1][-1] == C_max:
                flag_critical_path, critical_path = self.FindCriticalPath(T, machine, len(T[machine]) - 1, [])
                if flag_critical_path == 1:
                    break
        return critical_path

    def FindCriticalPath(self, T, machine_now, index_now, critical_path):
        """
        求解关键路径（内层）
        :param T:
        :param machine_now:
        :param index_now:
        :param critical_path:
        :return:
        """
        flag = 0
        process_now = T[machine_now][index_now]
        if not critical_path:
            critical_path.append({'machine': machine_now, 'head_index': index_now, 'block': [process_now[1:3]]})
        elif machine_now == critical_path[0]['machine']:
            critical_path[0]['block'].insert(0, process_now[1:3])
            critical_path[0]['head_index'] = index_now
        else:
            critical_path.insert(0, {'machine': machine_now, 'head_index': index_now, 'block': [process_now[1:3]]})
        start_time_now, job_now, job_process_now, _ = process_now
        # 若当前工序为起始工序，则确定已找到关键路径
        if start_time_now == 0:
            flag = 1

        # 机器紧前工序
        if flag == 0 and index_now != 0:
            index_pre_machine = index_now - 1
            process_pre_machine = T[machine_now][index_pre_machine]
            end_time_pre_machine = process_pre_machine[-1]
            if end_time_pre_machine == start_time_now:
                flag, critical_path = self.FindCriticalPath(T, machine_now, index_pre_machine, critical_path)

        # 工件紧前工序
        if flag == 0 and job_process_now != 0:
            job_process_pre_job = job_process_now - 1
            machine_pre_job = self.J[job_now][job_process_pre_job] - 1
            index_pre_job = 0
            process_pre_job = []
            for index in range(len(T[machine_pre_job])):
                if T[machine_pre_job][index][1] == job_now:
                    index_pre_job = index
                    process_pre_job = T[machine_pre_job][index_pre_job]
                    break
            end_time_pre_job = process_pre_job[-1]
            if end_time_pre_job == start_time_now:
                flag, critical_path = self.FindCriticalPath(T, machine_pre_job, index_pre_job, critical_path)

        return flag, critical_path

    def CreateInd(self):
        """
        个体染色体初始化
        :return chromosome: 个体染色体
        """
        jobs_list = list(range(self.N))
        J_temp = copy.deepcopy(self.J)
        chromosome = []
        while not np.all(J_temp == 0):
            job = np.random.choice(jobs_list)
            machine = J_temp[job, 0]
            if machine != 0:
                chromosome.append(job)
                J_temp[job, :] = np.roll(J_temp[job, :], -1)
                J_temp[job, :][-1] = 0
            else:
                jobs_list.remove(job)

        return chromosome

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
            pass
        if C_max <= self.C_op_max:
            self.T_op = T
            self.C_op = C
            self.C_op_max = C_max
            self.Ind_op = Ind

    def TerminationCriterion(self):
        """
        终止准则
        :return:
        """
        pass


def DrawGantt(timelist):
    """
    绘制Gantt图
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
