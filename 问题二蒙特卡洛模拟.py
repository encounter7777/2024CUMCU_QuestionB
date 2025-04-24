import numpy as np
"""
该文件利用蒙特卡洛算法模拟一定拆解时的生产情况，
"""
"""=======================设定随机种子，保证结果可重复性==========================="""
np.random.seed(42)
"""=======================#定义蒙特卡洛模拟函数==========================="""
def monte_carlo_simulation(data, decision,num_simulations=10000):
    """
    读取data中的参数，包括零件1的成本，检测成本，次品率，零件二成本.....等等
    共计12个参数
    """
    total_costs = []
    r1 = data[0]
    c1 = data[1]
    t1 = data[2]
    r2 = data[3]
    c2 = data[4]
    t2 = data[5]
    r3 = data[6]
    c3 = data[7]
    t3 = data[8]
    sales = data[9]
    loss = data[10]
    decay_loss = data[11]
    """=======================进行蒙特卡洛随机实验10000次==========================="""
    for _ in range(num_simulations):
        """=======================模拟购买的零配件1和零配件2==========================="""
        part1_def = np.random.rand() < r1
        part1_test = decision[0]
        part1_cost = c1 + (t1 if  part1_test else 0)
        part2_def = np.random.rand() < r2
        part2_test = decision[1]
        part2_cost = c2 + (t2 if  part2_test else 0)
        """=======================模拟零配件1和零配件2的测试==========================="""
        if part1_test and part1_def:
            final_cost = part1_cost
            total_costs.append(final_cost)
            continue
        if part2_test and part2_def:
            if part1_test:
                final_cost = t1+part2_cost
            else:
                final_cost = part2_cost
            total_costs.append(final_cost)
            continue
        """=======================模拟成品装配==========================="""
        product_def = part1_def or part2_def or (np.random.rand() < r3)
        assembly_cost = c3
        product_revenue = sales
        product_test = decision[2]
        """=======================模拟成品检测和拆解==========================="""
        if product_test:
            product_test_cost = t3
            if product_def:
                # 拆解处理
                disassembly_cost = decay_loss
                # 拆解后的零配件重新检测
                if part1_test:
                    part1_good = np.random.rand() > r1
                    part1_total_cost = part1_cost + t1- (c1 if  part1_good else 0)
                else:
                    part1_total_cost = part1_cost
                if part2_test:
                    part2_good = np.random.rand() > r2
                    part2_total_cost = part2_cost + t2 - (c2 if  part2_good else 0)
                else:
                    part2_total_cost = part2_cost
                final_cost = part1_total_cost + part2_total_cost + assembly_cost + product_test_cost+disassembly_cost
            else:
                # 成品合格
                final_cost = part1_cost + part2_cost + assembly_cost + product_test_cost - product_revenue
            """=======================模拟成品不检测时拆解==========================="""
        else:
            if product_def:
                # 拆解处理
                disassembly_cost = decay_loss
                # 拆解后的零配件重新检测
                if part1_test:
                    part1_good = np.random.rand() > r1
                    part1_total_cost = part1_cost + t1 - (c1 if  part1_good else 0)
                else:
                    part1_total_cost = part1_cost
                if part2_test:
                    part2_good = np.random.rand() > r2
                    part2_total_cost = part2_cost + t2 - (c2 if  part2_good else 0)
                else:
                    part2_total_cost = part2_cost
                final_cost = part1_total_cost + part2_total_cost + assembly_cost + disassembly_cost+loss
                """=======================模拟成品售卖==========================="""
            else:
                # 成品合格
                final_cost = part1_cost + part2_cost + assembly_cost - product_revenue
        #返回损失
        total_costs.append(final_cost)
    total_costs = np.array(total_costs)
    # 返回模拟后的期望成本
    return np.mean(total_costs)
"""=======================读入数据==========================="""
import pandas as pd
df = pd.read_excel('表1数据.xlsx',header=0)
df_numpy = df.values
data = df_numpy[:,1:]
decisions = [(0,0,0),(0,0,1), (0,1,0), (0,1,1),(1,0,0),(1,0,1), (1,1,0), (1,1,1)]
"""=======================对每个场景每种情况进行模拟==========================="""
for idx in range(data.shape[0]):
    expect_costs = np.inf
    flag = 0
    for x in decisions:
        temp = monte_carlo_simulation(data[idx],x)
        print(f"Scenario {idx + 1}",'decision',x,'cost',temp)
        if temp<expect_costs:
            expect_costs = temp
            flag = x
    print(f"Scenario {idx + 1}: Expected Least Cost = {expect_costs},Expected Most Income = {-expect_costs},decision = ",flag)





