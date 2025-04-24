import numpy as np
import random
from collections import defaultdict
from complicate_factory import Factory
import complicate_factory
'''
此文件基于贪心思想,构造新的蒙特卡洛树
用于解决问题三复杂生产情况时的多阶段决策问题
'''
np.random.seed(42)  # 设置随机种子以确保结果可复现
"""=======================创建一个按顺序生成的二进制矩阵==========================="""
def create_sequential_binary_matrix(rows, cols):
    matrix = np.array([list(map(int, format(i, '08b'))) for i in range(2 ** 8)])
    decision = []
    for x in matrix:
        if (x[2] * x[3]) == 0:
            x[7] = 0
        if (x[1] * x[3]) == 0:
            x[4] = 0
        if (x[2] * x[4]) == 0:
            x[5] = 0

        decision.append(x)
    decision = np.array(decision)
    decision = np.unique(decision, axis=0)
    decisions =[]
    for x in decision:
        x = [x[0:2], x[2:4], x[4], x[5:7], x[7]]
        decisions.append(x)
    return decisions
"""======================= 定义MCTS节点类==========================="""
class MCTSNode:
    def __init__(self, state, parent=None):
        """
        初始化蒙特卡洛树节点
        :param state: 当前节点的状态
        :param parent: 父节点
        """
        self.state = state  # 节点的状态
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.value = 0  # 节点值，用于记录模拟过程中的累积结果

    def is_fully_expanded(self):
        """
        检查当前节点是否已完全扩展（即是否所有可能的行动都已生成子节点）
        :return: 如果所有可能的行动都有子节点，则返回 True；否则返回 False
        """
        return len(self.children) == len(self.get_possible_actions())

    def get_possible_actions(self):
        """
        获取所有可能的决策方案
        :return: 所有可能的二进制决策方案
        """
        return create_sequential_binary_matrix(32, 8)

    def expand(self):
        """
        扩展当前节点，生成所有可能的子节点
        """
        actions = self.get_possible_actions()
        for action in actions:
            if not any(np.array_equal(action, child.state) for child in self.children):
                child_node = MCTSNode(action, self)
                self.children.append(child_node)
    def simulate(self):
        """
        从当前节点开始进行模拟，计算该状态的损失
        :return: 损失
        """
        loss = 0
        for i in range(100):
            factory = Factory()
            loss += factory.si(*self.state)  # 调用工厂类的方法，传入当前状态
        return loss  # 返回模拟过程中的损失
    def backpropagate(self, value):
        """
        回传模拟结果，更新节点的访问次数和价值
        :param value: 模拟结果的价值
        """
        node = self
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
"""======================= 定义 MCTS 类==========================="""
class MCTS:
    def __init__(self, root):
        """
        初始化蒙特卡洛树搜索
        :param root: 树的根节点
        """
        self.root = root  # 根节点
    def select(self, node):
        """
        使用 UCB1 算法选择一个子节点用于扩展
        :param node: 当前节点
        :return: 选择的子节点
        """
        while node.is_fully_expanded():
            values = [
                (child.value / (child.visits + 1e-10)) + np.sqrt(2 * np.log(node.visits + 1) / (child.visits + 1e-10))
                for child in node.children]
            node = node.children[np.argmax(values)]
        return node
    def run(self, iterations):
        """
        运行蒙特卡洛树搜索
        :param iterations: 迭代次数
        """
        for _ in range(iterations):
            node = self.root
            while node.is_fully_expanded() and node.children:
                node = self.select(node)
            if not node.is_fully_expanded():
                node.expand()
            value = node.simulate()
            node.backpropagate(value)
    def best_action(self):
        """
        获取当前树的前5个最佳行动（即最优决策方案）
        :return: 最佳决策状态列表
        """
        # 对所有子节点按值排序，并取前5个
        top_actions = sorted(self.root.children, key=lambda x: x.value)[:5]
        # 返回排序后的前5个最佳决策状态
        return [action.state for action in top_actions]
    # def best_action(self):
    #     """
    #     获取当前树的最佳行动（即最优决策方案）
    #     :return: 最佳决策状态
    #     """
    #     return min(self.root.children, key=lambda x: x.value).state
if __name__ == '__main__':
    # 创建根节点并执行 MCTS
    """=======================创建根节点并执行 MCTS==========================="""
    #root = MCTSNode(state=(0, 0, 0, 0, 0, 0, 0, 0))  # 初始状态为全都不检验
    root = MCTSNode(state=([1, 1], [1, 1], 1, [1, 1], 1))  # 初始状态为全检验
    mcts = MCTS(root)
    mcts.run(1000)  # 运行 1000 次迭代
    # 获取最佳决策方案
    best_decision = mcts.best_action()
    print('最优决策方案:', best_decision[0])
    """=======================MCTS灵敏性检验==========================="""
    # for _ in range(10):
    #     complicate_factory.p = complicate_factory.p*(1-random.choice([-1, 1])*0.05)
    #     root = MCTSNode(state=([1, 1], [1, 1], 1, [1, 1], 1))  # 初始状态为全检验
    #     mcts = MCTS(root)
    #     mcts.run(1000)  # 运行 1000 次迭代
    #     # 获取最佳决策方案
    #     best_decision = mcts.best_action()
    #     print('最优决策方案:', best_decision[0])
