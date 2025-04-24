import numpy as np
np.random.seed(32)
'''
此文件利用python类编程，模拟了工厂生产时的情况，零件->半成品->成品->市场
'''
"""=======================创建零件类==========================="""
p = 0.1
class Component:
    total_loss = 0
    def __init__(self, cost):
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
class semiProduct:
    total_loss = 0
    income = 0
    def __init__(self, components):
        self.components = components
        self.is_defective = any(comp.is_defective for comp in components) or np.random.random() < p
        self.cost = sum(comp.cost for comp in components)+8
        semiProduct.total_loss += self.cost
    """=======================观测==========================="""
    def inspect(self, fee):
        self.cost += fee
        semiProduct.total_loss += fee
        return not self.is_defective
    """=======================拆解==========================="""
    def disassemble(self):
        good_components = [comp for comp in self.components if not comp.is_defective]
        semiProduct.income = semiProduct.income+sum(comp.value for comp in good_components)-10
        self.cost = self.cost - semiProduct.income
        semiProduct.total_loss = semiProduct.total_loss-semiProduct.income
    @classmethod
    def get_total_loss(cls):
        """获取所有零件的损失之和"""
        return cls.total_loss
    @classmethod
    def reset_total_loss(cls):
        cls.total_loss = 0
        cls.income = 0
"""=======================创建成品类==========================="""
class Product:
    def __init__(self, semiProducts):
        self.semiProducts = semiProducts
        self.is_defective = any(sp.is_defective for sp in semiProducts) or np.random.random() < p
        self.cost = sum(sp.cost for sp in semiProducts)+8
    """=======================观测==========================="""
    def inspect(self, fee):
        self.cost += fee
        return not self.is_defective
    """=======================拆解==========================="""
    def disassemble(self,inspect_components):
        bad_components = [sp for sp in self.semiProducts if sp.is_defective]
        good_components = [sp for sp in self.semiProducts if not sp.is_defective]
        for sp in good_components:
            self.cost = self.cost-sp.cost
        if inspect_components:
            for sp in bad_components:
                sp.disassemble()
            for sp in good_components:
                self.cost = self.cost - sp.cost/3
            self.cost = self.cost-semiProduct.income+4*3
        self.cost = self.cost+10
"""=======================创建工厂类，模拟工厂==========================="""
class Factory:
    def __init__(self):
        self.total_cost = 0
        self.total_loss = 0
        self.total_revenue = 0
    """=======================购买零配件==========================="""
    def produce_components(self):
        component1 = Component(2)
        component2 = Component(8)
        component3 = Component(12)
        component4 = Component(2)
        component5 = Component(8)
        component6 = Component(12)
        component7 = Component(8)
        component8 = Component(12)
        components = [component1, component2, component3, component4, component5, component6, component7, component8]
        return components
    """=======================组装半成品==========================="""
    def assemble_semiproduct(self, components):
        semiproduct1 = semiProduct(components[0:3])
        semiproduct2 = semiProduct(components[3:6])
        semiproduct3 = semiProduct(components[6:8])
        semiproducts = [semiproduct1, semiproduct2, semiproduct3]
        return semiproducts
    """=======================组装成品==========================="""
    def assemble_product(self, components):
        product = Product(components)
        return product
    """=======================处理零配件==========================="""
    def process_components(self, components, inspect):
        if inspect:
            self.comp_good = []
            components_test_fee = [1, 1, 2, 1, 1, 2, 1, 2]
            for i, comp in enumerate(components):
                temp = comp.inspect(components_test_fee[i])
                self.comp_good.append(temp)
            self.total_loss = Component.get_total_loss()
            if all(self.comp_good):
                return 1, components
            else:
                for i, x in enumerate(self.comp_good):
                    if x:
                        self.total_loss -= components[i].cost
                return 0, components
        else:
            return 1, components
    """=======================处理半成品==========================="""
    def process_semiproducts(self, semiproducts,inspect_components,inspect,decay):
        if inspect:
            self.semiproduct_good = []
            for i, semiproduct in enumerate(semiproducts):
                temp = semiproduct.inspect(4)
                self.semiproduct_good.append(temp)
            if all(self.semiproduct_good):
                return 1, semiproducts
            else:
                for i, x in enumerate(self.semiproduct_good):
                    if not x:
                        if decay and inspect_components:
                            semiproducts[i].disassemble()
                self.total_loss = semiProduct.get_total_loss()
                return 0,semiproducts
        else:
            return 1, semiproducts
    """=======================处理成品==========================="""
    def process_product(self, Product1,inspect_components,inspect_semiproducts,inspect,decay):
        if inspect:
            inspect_result = Product1.inspect(6)
            if inspect_result:
                Product1.cost = Product1.cost - 200
            else:
                if decay:
                    if inspect_semiproducts:
                        Product1.disassemble(inspect_components)

        else:
            if Product1.is_defective:
                Product1.cost += 40
                if decay:
                    if inspect_semiproducts:
                        Product1.disassemble(inspect_components)

            else:
                Product1.cost -= 200
        self.total_loss = Product1.cost
    """=======================模拟仿真生产流程==========================="""
    def si(self,inspect_components, inspect_semiproducts, inspect_products,disassemble_semiproducts,disassemble_products):
        Component.reset_total_loss()
        semiProduct.reset_total_loss()
        components = self.produce_components()
        self.total_loss = Component.get_total_loss()
        result1,components = self.process_components(components,inspect_components)
        if result1:
            semiproducts = self.assemble_semiproduct(components)
            self.total_loss = semiProduct.get_total_loss()
            result2,semiproducts = self.process_semiproducts(semiproducts,inspect_components,inspect_semiproducts,disassemble_semiproducts)
            if result2:
                Product1 = self.assemble_product(semiproducts)
                self.total_loss = Product1.cost
                self.process_product(Product1,inspect_components,inspect_semiproducts,inspect_products,disassemble_products)
        return self.total_loss

if __name__ == '__main__':
    """=======================创建出决策数组==========================="""
    def create_sequential_binary_matrix(rows, cols):
        # Ensure we have enough rows
        if rows > 2 ** cols:
            raise ValueError("Number of rows exceeds the number of unique binary vectors possible.")
        # Generate binary numbers sequentially
        binary_matrix = np.array([list(map(int, format(i, f'0{cols}b'))) for i in range(rows)])
        return binary_matrix
    # Create a 32x5 matrix
    matrix = create_sequential_binary_matrix(32, 5)
    decision = []
    for x in matrix:
        if (x[3]+x[4]==0):
            decision.append(x)
        elif(x[1] * x[4]):
            decision.append(x)
        elif(x[0]*x[1]*x[3]):
            decision.append(x)
    decision = np.array(decision)
    """=======================创建工厂模型，并仿真模拟生产流程==========================="""
#Run simulation
    loss_list = []
    for j in range(len(decision)):
        loss = 0
        arr = decision[j]
        for i in range(10000):
            factory = Factory()
            loss += factory.si(arr[0], arr[1], arr[2], arr[3], arr[4])
        loss_list.append(loss)
    loss_list = np.array(loss_list)

    arg = np.argsort(loss_list)
    decision = decision[arg]
    loss_list = loss_list[arg]
    import matplotlib.pyplot as plt
    for i in range(len(loss_list)):
        print('决策方案:',decision[i],'期望收益',-loss_list[i])
    plt.plot(-loss_list)
    plt.show()
    print('最好的决策方案:',decision[np.argmin(loss_list)],'此时期望收益',-np.min(loss_list))