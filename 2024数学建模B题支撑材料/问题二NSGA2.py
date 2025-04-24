from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np
import pandas as pd
'''
该文件使用非支配排序遗传算法解决不拆解时生产情况的决策问题
'''
"""=======================定义优化函数==========================="""
class ProductionProblem(Problem):
    def __init__(self, data):
        # 定义决策变量的数量
        super().__init__(n_var=3, n_obj=1, n_constr=0, xl=0, xu=1,variable_type="I")
        self.data = data
    def _evaluate(self, x, out, *args, **kwargs):
        # 计算每个样本的目标函数值
        f = np.array([(self.objective(i)) for i in x])
        out["F"] = f.reshape(-1, 1)
    def objective(self, x):
        r1, c1, t1, r2, c2, t2, r3, c3, t3, sales, loss, decay_loss = self.data
        cost = c1 + c2 + c3 + t1 * x[0] + t2 * x[1] + t3 * x[2]
        good_r = (1 - r1) * (1 - r2) * (1 - r3)
        loss_dh = (1 - x[2]) * (1 - good_r) * loss
        income = sales * good_r - loss_dh
        return cost - income
"""=======================读入数据==========================="""
df = pd.read_excel('表1数据.xlsx', header=0)
df_numpy = df.values
data = df_numpy[:, 1:]
"""=======================执行优化函数，得到目标结果==========================="""
for i, row_data in enumerate(data):
    problem = ProductionProblem(row_data)
    algorithm = NSGA2(pop_size=100)
    result = minimize(problem, algorithm, ('n_gen', 100), verbose=False)
    best_x = result.X[0]
    best_y = result.F[0][0]
    print(f"Scenario {i + 1}: Expected Most Income = {-best_y}, decision = ", best_x)