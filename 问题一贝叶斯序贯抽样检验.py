import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
'''
该文件利用贝叶斯序贯检验找出最小检验样本数
'''
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 设置随机种子以确保结果可复现
np.random.seed(60)
"""=======================定义β分布函数==========================="""
def beta_pdf(x, alpha, beta):
    """计算贝塔分布的概率密度函数"""
    return stats.beta.pdf(x, alpha, beta)
"""=======================定义计算检测最小样本数临界值函数==========================="""
def calculate_critical_values(n, p0, alpha, beta, a, b):
    """计算临界值 L_n 和 U_n """
    L_n = 0
    U_n = np.inf
    for x in range(n+1):
        posterior_prob_H0 = stats.beta.cdf(p0, x + a, n - x + b) - stats.beta.cdf(0, x + a, n - x + b)
        if posterior_prob_H0 < alpha:
            U_n = x
            break

    for x in range(n, 0, -1):
        posterior_prob_H1 =  stats.beta.cdf(1, x+a, n-x+b) - stats.beta.cdf(p0, x+a, n-x+b)
        if posterior_prob_H1 < beta:
            L_n = x
            break
    return L_n, U_n
"""=======================贝叶斯序贯检验==========================="""
def bayesian_sequential_test(N, p0,p, alpha, beta, a, b,iteration = 10):
    """执行贝叶斯序贯检验"""
    result = 0
    num = 0
    for i in range(iteration):
        n = 10
        x = np.random.binomial(1, p,10).sum()
        while n < N:
            L_n, U_n = calculate_critical_values(n, p0, alpha, beta, a, b)
            # 假设每次实验成功或失败的概率是随机的，这里简化为均匀分布
            sample = np.random.binomial(1, p)  # 假设按p生成的样本
            if sample == 1:
                x += 1
            n += 1
            if x <= L_n:
                #print(f"Accept H0 at n={n}, x={x}")
                result += 1
                num += n
                break
            elif x >= U_n:
                #print(f"Accept H1 at n={n}, x={x}")
                result += 0
                num += n
                break
            """=======================设置最大检验样本数为1000==========================="""
        else:
            print("Reach maximum number of trials without decision.")
    num = num / iteration
    if result >= 5:
        print(f"Accept H0 at n={num},")
    else:
        print(f"Accept H1 at n={num} ")
    return num
"""=======================贝叶斯序贯检验的参数==========================="""
# 参数示例
N = 1000  # 最大实验次数
p0 = 0.1  # 次品概率阈值
p1 = 0.9  #良品概率阈值
p_list = np.linspace(0.01, 0.4, 40) #真实次品概率
ran = 0.04
alpha0 = 0.05  # 第一类错误概率
beta0 = 0.05  # 第二类错误概率
alpha1 = 0.10  # 第一类错误概率
beta1 = 0.10  # 第二类错误概率
a = 1  # 贝塔分布的alpha参数
b = 1  # 贝塔分布的beta参数
"""=======================执行贝叶斯序贯检验==========================="""
# num1 = []
# for p in p_list:
#     if np.abs(p-0.1)>ran:
#         result_num1 = bayesian_sequential_test(N, p0,p, alpha0, beta0, a, b,1000)
#         num1.append(result_num1)
#num1 = np.array(num1)
#np.save("num1.npy", num1)
"""=======================可视化和保存结果==========================="""
num1 = np.load('num1.npy')
plt.figure()
plt.scatter(p_list[np.where(np.abs(p_list-0.1)>ran)], num1,label='95%信度拒收')
#plt.scatter(p_list[np.where(np.abs(p_list-0.1)>ran)], num2,label='95%信度接收')
plt.xlabel('次品率')
plt.ylabel('最小检验样本数')
plt.legend()
plt.show()
# plt.savefig('./picture/结果1.png')
"""=======================执行贝叶斯序贯检验==========================="""
# num2 = []
# for p in p_list:
#     if np.abs(p-0.1)>ran:
#         result_num2 = bayesian_sequential_test(N, p0,p, alpha1, beta1, a, b,1000)
#         num2.append(result_num2)
"""=======================可视化和保存结果==========================="""
# num2 = np.array(num2)
# np.save("num2.npy", num2)
num2 = np.load('num2.npy')
plt.figure()
plt.scatter(p_list[np.where(np.abs(p_list-0.1)>ran)], num2,label='90%信度接收')
plt.xlabel('次品率')
plt.ylabel('最小检验样本数')
plt.legend()
plt.show()
#plt.savefig('./picture/结果2.png')
