import copy
import random
import numpy as np
from utils import get_pro

def random_architecture(startLength):
    """Returns a random architecture (bit-string) represented as an int."""
    return list(np.random.randint(0, 2, startLength))

def random_architecture_new(startLength):
    """Returns a random architecture (bit-string) represented as an int."""
    # 0代表T 1/2/3代表多种P
    return list(np.random.randint(0, 4, startLength))

def random_architecture_addition(startLength):
    """Returns a random architecture (bit-string) represented as an int."""
    # 0代表T 1/2/3/4/5代表多种P
    return list(np.random.randint(0, 6, startLength))


def mutate_arch(parent_arch, mutate_type):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    #随机将arch中一个操作变化,从T-P和P-T相互变化
    if mutate_type == 2:
        position = random.randint(0, length-1)
        child_arch[position] ^= 1
    #随机插入一个P操作
    elif mutate_type == 1:
        position = random.randint(0, length)
        child_arch.insert(position, 1)
    #随机插入一个T操作
    elif mutate_type == 0:
        position = random.randint(0, length)
        child_arch.insert(position, 0)
    else:
        print('mutate type error')
    return child_arch

def mutate_arch_new(parent_arch, mutate_type):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    #判断arch长度大于2，这样保证删除后arch还是会进行P/T处理
    if mutate_type == 3:
        if length > 2:
            position = random.randint(0,length-1)
            del child_arch[position]
            return child_arch
        else:
            mutate_type = random.randint(0,2)
    #随机将arch中一个操作变化,从T-P和P-T相互变化
    if mutate_type == 2:
        position = random.randint(0, length-1)
        child_arch[position] ^= 1
    #随机插入一个P操作
    elif mutate_type == 1:
        position = random.randint(0, length)
        child_arch.insert(position, 1)
    #随机插入一个T操作
    elif mutate_type == 0:
        position = random.randint(0, length)
        child_arch.insert(position, 0)
    return child_arch

def mutate_arch_multi(parent_arch, mutate_type):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    #判断arch长度大于2，这样保证删除后arch还是会进行P/T处理
    if mutate_type == 3:
        if length > 2:
            position = random.randint(0,length-1)
            del child_arch[position]
            return child_arch
        else:
            mutate_type = random.randint(0,2)
    #随机将arch中一个操作变化,从T-P和P-T相互变化
    if mutate_type == 2:
        position = random.randint(0, length-1)
        if child_arch[position] == 0:
            new_value = random.randint(2,4)
            child_arch[position] = new_value
        else:
            child_arch[position] = 0
    #随机插入一个P操作
    elif mutate_type == 1:
        position = random.randint(0, length-1)
        new_value = random.randint(2, 4)
        child_arch.insert(position, new_value)
    #随机插入一个T操作
    elif mutate_type == 0:
        position = random.randint(0, length-1)
        child_arch.insert(position, 0)
    return child_arch


def mutate_arch_addition(parent_arch, mutate_type):
    """不是以随机的方式生成0/1，而是随机生成1,2,3,4,5五种P操作"""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    #判断arch长度大于2，这样保证删除后arch还是会进行P/T处理
    if mutate_type == 3:
        if length > 2:
            position = random.randint(0,length-1)
            del child_arch[position]
            return child_arch
        else:
            mutate_type = random.randint(0,2)
    #随机将arch中一个操作变化,从T-P和P-T相互变化
    if mutate_type == 2:
        position = random.randint(0, length-1)
        if child_arch[position] == 0:
            idx = random.randint(1,5)
            child_arch[position] = idx
        else:
            child_arch[position] = 0
    #随机插入一个P操作
    elif mutate_type == 1:
        idx = random.randint(1, 5)
        position = random.randint(0, length)
        child_arch.insert(position, idx)
    #随机插入一个T操作
    elif mutate_type == 0:
        position = random.randint(0, length)
        child_arch.insert(position, 0)
    return child_arch

def mutate_arch_xiaorong(parent_arch):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    #随机将arch中一个操作变化,从T-P和P-T相互变化
    position = random.randint(0, length-1)
    child_arch[position] ^= 1
    return child_arch

def mutate_arch_constrain(parent_arch, mutate_type, s_dict, a_dict):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    s_list = get_pro(s_dict)
    a_list = get_pro(a_dict)
    def normalization(a):
        total = 0
        length = len(a)
        for i in a:
            total += i
        if total != 0:
            for i in range(length):
                a[i] = a[i]/total
        else:
            for i in range(length):
                a[i] = 1 / length
        return a
    # 判断arch长度大于2，这样保证删除后arch还是会进行P/T处理
    if mutate_type == 3:
        if length > 2:
            position = random.randint(0,length-1)
            del child_arch[position]
            return child_arch
        else:
            mutate_type = random.randint(1,2)
    # 随机将arch中一个操作变化,从T-P和P-T相互变化
    if mutate_type == 2:
        position = random.randint(0, length-1)
        # 如果选择的位置是T，转换为3种P操作
        if child_arch[position] == 0:
            p_list = []
            p_list.extend(s_list[1:])
            normalization(p_list)
            a = [1, 2, 3]
            idx = np.random.choice(a, p=p_list)
            child_arch[position] = idx
        else:
            child_arch[position] = 0
    # 如果是addition的话则将其在长度为2的子序列之中按照概率进行选择
    else:
        position = random.randint(0, length-1)
        # 选择到当前变异的位置的值temp，在a_list/dict_1中是[temp*4,temp*4+3]
        temp = int(child_arch[position])
        p_list = []
        p_list.extend(a_list[temp*4:temp*4+4])
        # 将该算子开头的子序列进行归一化
        normalization(p_list)
        # 插入到position之后即可
        a = [0,1,2,3]
        idx = np.random.choice(a, p=p_list)
        child_arch.insert(position, idx)
    return child_arch


def mutate_arch_combine(parent_arch, mutate_type, s_dict, a_dict):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    s_list = get_pro(s_dict)
    a_list = get_pro(a_dict)
    def normalization(a):
        total = 0
        length = len(a)
        for i in a:
            total += i
        if total != 0:
            for i in range(length):
                a[i] = a[i] / total
        else:
            for i in range(length):
                a[i] = 1 / length
        return a
    # 判断arch长度大于2，这样保证删除后arch还是会进行P/T处理
    if mutate_type == 3:
        if length > 2:
            position = random.randint(0,length-1)
            del child_arch[position]
            return child_arch
        else:
            mutate_type = random.randint(1,2)
    # 随机将arch中一个操作变化,从T-P和P-T相互变化
    if mutate_type == 2:
        position = random.randint(0, length-1)
        # 如果选择的位置是T，转换为3种P操作
        if child_arch[position] == 0:
            p_list = []
            p_list.extend(s_list[1:])
            normalization(p_list)
            a = [1, 2, 3, 4, 5]
            idx = np.random.choice(a, p=p_list)
            child_arch[position] = idx
        else:
            child_arch[position] = 0
    # 如果是addition的话则将其在长度为2的子序列之中按照概率进行选择
    else:
        position = random.randint(0, length-1)
        # 选择到当前变异的位置的值temp，在a_list/dict_1中是[temp*4,temp*4+5]
        temp = int(child_arch[position])
        p_list = []
        p_list.extend(a_list[temp*4:temp*4+6])
        # 将该算子开头的子序列进行归一化
        normalization(p_list)
        # 插入到position之后即可
        a = [0, 1, 2, 3, 4, 5]
        idx = np.random.choice(a, p=p_list)
        child_arch.insert(position, idx)
    return child_arch


#提供一个跟随迭代轮数变化的概率p，来决定使用哪种方式变异
def mutate_arch_combine_new(parent_arch, mutate_type, s_dict, a_dict, p):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    def normalization(a):
        total = 0
        length = len(a)
        for i in a:
            total += i
        if total != 0:
            for i in range(length):
                a[i] = a[i] / total
        else:
            for i in range(length):
                a[i] = 1 / length
        return a
    # 当随机数小于p则进行约束，迭代轮数越后可能性越大
    if random.random() < p:
        s_list = get_pro(s_dict)
        a_list = get_pro(a_dict)
        # 判断arch长度大于2，这样保证删除后arch还是会进行P/T处理
        if mutate_type == 3:
            if length > 2:
                position = random.randint(0,length-1)
                del child_arch[position]
                return child_arch
            else:
                mutate_type = random.randint(1,2)
        # 随机将arch中一个操作变化,从T-P和P-T相互变化
        if mutate_type == 2:
            position = random.randint(0, length-1)
            # 如果选择的位置是T，转换为5种P操作
            if child_arch[position] == 0:
                p_list = []
                p_list.extend(s_list[1:])
                normalization(p_list)
                a = [1, 2, 3, 4, 5]
                idx = np.random.choice(a, p=p_list)
                child_arch[position] = idx
            else:
                child_arch[position] = 0
        # 如果是addition的话则将其在长度为2的子序列之中按照概率进行选择
        else:
            position = random.randint(0, length-1)
            # 选择到当前变异的位置的值temp，在a_list/dict_1中是[temp*4,temp*4+5]
            temp = int(child_arch[position])
            p_list = []
            p_list.extend(a_list[temp*4:temp*4+6])
            # 将该算子开头的子序列进行归一化
            normalization(p_list)
            # 插入到position之后即可
            a = [0, 1, 2, 3, 4, 5]
            idx = np.random.choice(a, p=p_list)
            child_arch.insert(position, idx)
    # 进行随机变异
    else:
        if mutate_type == 3:
            if length > 2:
                position = random.randint(0, length - 1)
                del child_arch[position]
                return child_arch
            else:
                mutate_type = random.randint(0, 2)
        # 随机将arch中一个操作变化,从T-P和P-T相互变化
        if mutate_type == 2:
            position = random.randint(0, length - 1)
            if child_arch[position] == 0:
                idx = random.randint(1, 5)
                child_arch[position] = idx
            else:
                child_arch[position] = 0
        # 随机插入一个P操作
        elif mutate_type == 1:
            idx = random.randint(1, 5)
            position = random.randint(0, length)
            child_arch.insert(position, idx)
        # 随机插入一个T操作
        elif mutate_type == 0:
            position = random.randint(0, length)
            child_arch.insert(position, 0)
    return child_arch



#提供一个跟随迭代轮数变化的概率p，来决定使用哪种方式变异
def mutate_arch_constrain_new(parent_arch, mutate_type, s_dict, a_dict, p):
    """Computes the architecture for a child of the given parent architecture."""
    length = len(parent_arch)
    child_arch = parent_arch.copy()
    def normalization(a):
        total = 0
        length = len(a)
        for i in a:
            total += i
        if total != 0:
            for i in range(length):
                a[i] = a[i] / total
        else:
            for i in range(length):
                a[i] = 1 / length
        return a

    # 当随机数小于p则进行约束，迭代轮数越后可能性越大
    if random.random() < p:
        s_list = get_pro(s_dict)
        a_list = get_pro(a_dict)
        # 判断arch长度大于2，这样保证删除后arch还是会进行P/T处理
        if mutate_type == 3:
            if length > 2:
                position = random.randint(0,length-1)
                del child_arch[position]
                return child_arch
            else:
                mutate_type = random.randint(1,2)
        # 随机将arch中一个操作变化,从T-P和P-T相互变化
        if mutate_type == 2:
            position = random.randint(0, length-1)
            # 如果选择的位置是T，转换为P操作
            if child_arch[position] == 0:
                child_arch[position] = 1
            else:
                child_arch[position] = 0
        # 如果是addition的话则将其在长度为2的子序列之中按照概率进行选择
        else:
            position = random.randint(0, length-1)
            # 选择到当前变异的位置的值temp，在a_list/dict_1中是[temp*2,temp*2+1]
            temp = int(child_arch[position])
            p_list = []
            p_list.extend(a_list[temp*2:temp*2+2])
            # 将该算子开头的子序列进行归一化
            normalization(p_list)
            # 插入到position之后即可
            a = [0, 1]
            idx = np.random.choice(a, p=p_list)
            child_arch.insert(position, idx)
    # 进行随机变异
    else:
        if mutate_type == 3:
            if length > 2:
                position = random.randint(0, length - 1)
                del child_arch[position]
                return child_arch
            else:
                mutate_type = random.randint(0, 2)
        # 随机将arch中一个操作变化,从T-P和P-T相互变化
        if mutate_type == 2:
            position = random.randint(0, length - 1)
            if child_arch[position] == 0:
                child_arch[position] = 1
            else:
                child_arch[position] = 0
        # 随机插入一个P操作
        elif mutate_type == 1:
            position = random.randint(0, length)
            child_arch.insert(position, 1)
        # 随机插入一个T操作
        elif mutate_type == 0:
            position = random.randint(0, length)
            child_arch.insert(position, 0)
    return child_arch


