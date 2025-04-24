import numpy as np
np.random.seed(22)
'''
此文件利用python类编程，模拟了问题二工厂生产时的情况，零件->成品->市场
通过此文件,可以检验问题二模型的合理性
并且同时也一定程度的验证了问题三中模拟工厂模型的可行性与合理性
'''
"""=======================创建零件类==========================="""
class Component:
    total_loss = 0
    def __init__(self, cost,p):
        self.value = cost
        self.cost = cost
        self.is_defective = np.random.random() < p  # 假设次品率为10%
        Component.total_loss += cost
    """=======================观测==========================="""
    def inspect(self, fee):
        self.cost += fee
        Component.total_loss += fee
        return not self.is_defective
    @classmethod
    def get_total_loss(cls):
        """获取所有零件的损失之和"""
        return cls.total_loss
    @classmethod
    def reset_total_loss(cls):
        cls.total_loss = 0
"""=======================创建半成品类==========================="""
class Product:

    def __init__(self, components,p,component_fee):
        self.components = components
        self.is_defective = any(comp.is_defective for comp in components) or np.random.random() < p
        self.cost = sum(comp.cost for comp in components)+component_fee

    """=======================观测==========================="""
    def inspect(self, fee):
        self.cost += fee
        return not self.is_defective
    """=======================拆解==========================="""
    def disassemble(self,inspectcomponents,decay_fee):
        if inspectcomponents[0]:
            if not self.components[0].is_defective:
                self.cost -= self.components[0].cost
        if inspectcomponents[1]:
            if not self.components[1].is_defective:
                self.cost -= self.components[1].cost
        self.cost+=decay_fee
    @classmethod
    def get_total_loss(cls):
        """获取所有零件的损失之和"""
        return cls.total_loss
    @classmethod
    def reset_total_loss(cls):
        cls.total_loss = 0
        cls.income = 0
class Factory:
    def __init__(self):
        self.total_cost = 0
        self.total_loss = 0
        self.total_revenue = 0
    """=======================购买零配件==========================="""
    def produce_components(self,components_cost,component_p):
        component1 = Component(components_cost[0],component_p[0])
        component2 = Component(components_cost[1],component_p[1])
        components = [component1, component2]
        return components
    """=======================组装成品==========================="""
    def assemble_product(self, components,Product_p,component_fee):
        product = Product(components,Product_p,component_fee)
        return product
    """=======================处理零配件==========================="""
    def process_components(self, components, inspect,inspectfee):

        if inspect[0]:
            temp = components[0].inspect(inspectfee[0])
            if not temp:
                result1 = 0
            else:
                result1 = 1
        else:
            result1 = 1
        if inspect[1]:
            temp = components[1].inspect(inspectfee[1])
            if not temp:
                result2 = 0
            else:
                result2 = 1
        else:
            result2 = 1
        result = result1 and result2
        if not result:
            if inspect[0]:
                if not  components[0].is_defective:
                    self.total_loss -= components[0].value
            if inspect[1]:
                if not  components[1].is_defective:
                    self.total_loss -= components[1].value

        return result,components

    """=======================处理成品==========================="""
    def process_product(self, Product1,inspect_components,inspect,decay,inspect_fee,sales,loss,decay_fee):
        if inspect:
            inspect_result = Product1.inspect(inspect_fee)
            if inspect_result:
                Product1.cost = Product1.cost - sales
            else:
                if decay:
                    if any(inspect_components):
                        Product1.disassemble(inspect_components,decay_fee)

        else:
            if Product1.is_defective:
                Product1.cost += loss
                if decay:
                    if any(inspect_components):
                        Product1.disassemble(inspect_components,decay_fee)
            else:
                Product1.cost -= sales
        self.total_loss = Product1.cost
    def simulate(self,state,data):
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

        Component.reset_total_loss()
        components = self.produce_components([c1,c2],[r1,r2])
        self.total_loss = Component.total_loss
        result,components = self.process_components(components,state[0:2],[t1,t2])
        if result == True:
            product = self.assemble_product(components,r3,c3)
            self.total_loss = product.cost
            self.process_product(product,state[0:2],state[2],state[3],t3,sales,loss,decay_loss)
        return self.total_loss

def create_sequential_binary_matrix():
    state = np.array([list(map(int, format(i, '04b'))) for i in range(2**4)])
    state = np.delete(state,[1,3],axis=0)
    return state



if __name__ == '__main__':
    """=======================读入数据==========================="""
    import pandas as pd
    df = pd.read_excel('表1数据.xlsx', header=0)
    df_numpy = df.values
    data = df_numpy[:, 1:]
    """======================求出最佳决策和此时的收益==========================="""
    state = np.array([list(map(int, format(i, '04b'))) for i in range(2**4)])
    state = np.delete(state,[1,3],axis=0)

    def fun_simu(num):
        loss_list = []
        for i in range(state.shape[0]):
            x = state[i]
            loss = 0
            for _ in range(10000):
                factory = Factory()
                loss += factory.simulate(x,data[num])
            loss_list.append(loss)
        loss_list = np.array(loss_list)
        arg = np.argsort(loss_list)
        loss_list = loss_list[arg]
        st = state[arg]
        return st[0],-loss_list[0]
    for i in range(6):
        test,income = fun_simu(i)
        print(test,income)
