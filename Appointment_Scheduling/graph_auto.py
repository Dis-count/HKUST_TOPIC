import matplotlib.pyplot as plt
import numpy as np


class drawGraph:
    def __init__(self, total_time, people_num):
        self.total_time = total_time
        self.people_number = people_num
        self.time_slot = int(total_time/people_num)

    def graph(self, x):
        #  总的时间段为5小时，即300分钟
        people_number = self.people_number

        #  一个预约时长为15到45分钟之间的均匀分布
        # start_time = [i for i in range(0, total_time_slots, time_slot)]

        start_time = x

        # for i in range(people_number):
        #     start_time[i] += np.random.randint(-10, 10)

        # min_service_time = 25
        # max_service_time = 45


        # 初始化一个空的时间段列表，用于记录每个时间段内的服务占用时间
        service_time = [0] * people_number

        # 指数分布
        np.random.seed(120)
        # 正态分布
        service_time[0] = np.random.normal(30,10)

        # 模拟10个预约
        # service_time[0] = np.random.randint(min_service_time, max_service_time)

        for customer_id in range(1, people_number):
        # 随机选择一个时间段进行预约
        # appointment_time = random.randint(0, total_time_slots - 1)

        # 随机生成服务时长
        # 更新时间段内的服务占用时间
            service_time[customer_id] = np.random.normal(30, 10) + max(service_time[customer_id - 1] - self.time_slot, 0)

        # 生成每个顾客占用时间长度的横向条形图
        plt.figure(figsize = (12, 6))
        for i in range(people_number):
            # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
            # Start Time
            int_label = [str(i) for i in range(people_number)]

            plt.barh(int_label, service_time, height = 0.5, left = start_time, label = f'Customer {i+1}', color = plt.cm.viridis(i/10))

        plt.xlabel('Time Slots')
        plt.ylabel('Customers')
        plt.title('Customer Service Time v.s. Time Slots')
        # plt.yticks(range(10), [f'Customers {i}' for i in range(10)])
        plt.legend()
        plt.grid(axis='x')
        plt.show()

if __name__ == "__main__":
    total_time = 300
    people_num = 10

    a = drawGraph(total_time, people_num)
    x = [i for i in range(0, total_time, a.time_slot)]
    # print(x)
    # a.graph(x)
    
    # y = [0, 0, 0.59840703, 0.81710032, 0.85243918, 0.8463719, 0.83447579, 0.86968908, 0.74821749, 0.61779605]
    # y = [10* i for i in y]
    
    y = [0, -1.03, -0.0369, -0.0253, -0.0328, -0.059, -2.5e-4, -0.116, -0.155, -0.3436]
    y = [10 * i for i in y]
    y = np.cumsum(y)
    x1 = [x[i]+y[i] for i in range(people_num)]
    print(x1)
    a.graph(x1)
