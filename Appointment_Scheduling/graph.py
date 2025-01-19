import matplotlib.pyplot as plt
import numpy as np


#  总的时间段为5小时，即300分钟
total_time_slots = 300

people_number = 10
time_slot = int(total_time_slots/people_number)

#  一个预约时长为15到45分钟之间的均匀分布
start_time = [i for i in range(0, total_time_slots, time_slot)] 

for i in range(people_number):
    start_time[i] += np.random.randint(-10, 10)

min_service_time = 20
max_service_time = 40

# 初始化一个空的时间段列表，用于记录每个时间段内的服务占用时间
service_time = [0] * people_number

# 指数分布

# 正态分布
#  random.gauss(5,1)

# 模拟10个预约
service_time[0] = np.random.randint(min_service_time, max_service_time)

for customer_id in range(1, people_number):
# 随机选择一个时间段进行预约
# appointment_time = random.randint(0, total_time_slots - 1)

# 随机生成服务时长
# 更新时间段内的服务占用时间
    service_time[customer_id] = np.random.randint(
        min_service_time, max_service_time) + max(service_time[customer_id - 1] - time_slot, 0)

# 生成每个顾客占用时间长度的横向条形图
plt.figure(figsize = (12, 6))
for i in range(people_number):
    # Start Time
    plt.barh(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], service_time, height = 0.5, left = start_time, label = f'Customer {i+1}', color = plt.cm.viridis(i/10))

plt.xlabel('Time Slots')
plt.ylabel('Customers')
plt.title('Customer Service Time v.s. Time Slots')
# plt.yticks(range(10), [f'Customers {i}' for i in range(10)])
plt.legend()
plt.grid(axis='x')
plt.show()