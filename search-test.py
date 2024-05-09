import sys
import time
import random
import argparse
import collections
import numpy as np

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from utils import *
from train import *
from operation import *
from mutation import *
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser("new-data")
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--data', default='wisconsin',help='six new datasets')
parser.add_argument('--hiddim', type=int, default=256, help='hidden dims')
parser.add_argument('--fdrop', type=float, default=0.5, help='drop for pubmed feature')
parser.add_argument('--drop', type=float, default=0.8, help='drop for pubmed layers')
parser.add_argument('--learning_rate', type=float, default=0.03, help='init pubmed learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--flag', type=int, default=0, help='determine which kind of dataset')
# 该参数是用于在验证集上取最优的evals个体求平均
parser.add_argument('--evals', type=int, default=10, help='num of evals')
parser.add_argument('--startLength', type=int, default=4, help='num of startArch')
args = parser.parse_args()

# #取出对应数据集名字和数据集划分数据
datastr=args.data

if args.flag == 0:
    splitstr=splitstr = '../splits/'+args.data+'_split_0.6_0.2_'+str(1)+'.npz'
    adj, features, labels, idx_train, idx_val, idx_test = load_new_data(datastr, splitstr)
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_big_data(args.data)
# adj, features, labels, idx_train, idx_val, idx_test = load_data(path="../data", dataset='pubmed')



#将P操作增加为三个无参聚合
adj_nor = aug_normalized_adjacency(adj)
adj_com = aug_compare_adjacency(adj)
adj_sing = adj_com + sp.eye(adj_com.shape[0])

#关于行列标准化的无参聚集算子增加
adj_row = aug_row_normalizaed_adjacency(adj)
adj_col = aug_col_normalizaed_adjacency(adj)
adj_row = sparse_mx_to_torch_sparse_tensor(adj_row).float().cuda()
adj_col = sparse_mx_to_torch_sparse_tensor(adj_col).float().cuda()

adj_nor = sparse_mx_to_torch_sparse_tensor(adj_nor).float().cuda()
adj_com = sparse_mx_to_torch_sparse_tensor(adj_com).float().cuda()
adj_sing = sparse_mx_to_torch_sparse_tensor(adj_sing).float().cuda()
features = features.cuda()
labels = labels.cuda()
data = adj_nor, adj_com, adj_sing, adj_row, adj_col, features, labels

idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
index = idx_train, idx_val, idx_test

# 对于addition设置一个字典进行转换
dict_1 = {
    0:"00",
    1:"01",
    2:"02",
    3:"03",
    4:"04",
    5:"05",
    6:"10",
    7:"11",
    8:"12",
    9:"13",
    10:"14",
    11:"15",
    12:"20",
    13:"21",
    14:"22",
    15:"23",
    16:"24",
    17:"25",
    18:"30",
    19:"31",
    20:"32",
    21:"33",
    22:"34",
    23:"35",
    24:"40",
    25:"41",
    26:"42",
    27:"43",
    28:"44",
    29:"45",
    30:"50",
    31:"51",
    32:"52",
    33:"53",
    34:"54",
    35:"55"
}



class Model(object):
    """A class representing a model."""

    def __init__(self):
        self.arch = None
        self.val_acc = None
        self.test_acc = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return self.arch


def main(cycles, population_size, sample_size):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)



    # 根据fitness来淘汰个体，需要一个列表来保存当前的population
    population = []
    history = []  # Not used by the algorithm, only used to report results.

    val_compare = []

    temp_num = population_size*5
    # temp_num = 10
    init_num = 0
    len_arch=args.startLength*3
    init_record = []
    # 记录conversion中的变异概率
    s_record = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0}
    # 记录addition的变异概率
    a_record = {
        '00':0,
        '01':0,
        '02':0,
        '03':0,
        '04':0,
        '05':0,
        '10':0,
        '11':0,
        '12':0,
        '13':0,
        '14':0,
        '15':0,
        '20':0,
        '21':0,
        '22':0,
        '23':0,
        '24':0,
        '25':0,
        '30':0,
        '31':0,
        '32':0,
        '33':0,
        '34':0,
        '35':0,
        '40': 0,
        '41': 0,
        '42': 0,
        '43': 0,
        '44': 0,
        '45': 0,
        '50': 0,
        '51': 0,
        '52': 0,
        '53': 0,
        '54': 0,
        '55': 0,
    }
    # 从5倍的init_set中选择出population进行频率统计
    while init_num < temp_num:
        model = Model()
        if args.flag == 1:
            tlen = random.randint(args.startLength, len_arch)
            model.arch = random_architecture_addition(tlen)
        else:
            tlen = random.randint(1, 4)
            model.arch = random_architecture_addition(tlen)
        model.val_acc, model.test_acc, nnew_arch = train_and_eval_constrain(args, model.arch, data, index)
        init_record.append(model)
        init_num += 1
        print("the %d initialization res{val_acc:%f   test_acc:%f}" % (init_num , model.val_acc, model.test_acc))
    sorted(init_record,key=lambda i:i.val_acc,reverse=True)
    history.extend(init_record[:population_size])
    population.extend(init_record[:population_size])



    # 根据populaion赋予两个频率表初始值
    for i in population:
        temp_arch = i.arch
        s_record['0'] += find_substr(temp_arch, '0')
        s_record['1'] += find_substr(temp_arch, '1')
        s_record['2'] += find_substr(temp_arch, '2')
        s_record['3'] += find_substr(temp_arch, '3')
        s_record['4'] += find_substr(temp_arch, '4')
        s_record['5'] += find_substr(temp_arch, '5')
        a_record['00'] += find_substr(temp_arch, "00")
        a_record['01'] += find_substr(temp_arch, "01")
        a_record['02'] += find_substr(temp_arch, "02")
        a_record['03'] += find_substr(temp_arch, "03")
        a_record['04'] += find_substr(temp_arch, "04")
        a_record['05'] += find_substr(temp_arch, "05")
        a_record['10'] += find_substr(temp_arch, "10")
        a_record['11'] += find_substr(temp_arch, "11")
        a_record['12'] += find_substr(temp_arch, "12")
        a_record['13'] += find_substr(temp_arch, "13")
        a_record['14'] += find_substr(temp_arch, "14")
        a_record['15'] += find_substr(temp_arch, "15")
        a_record['20'] += find_substr(temp_arch, "20")
        a_record['21'] += find_substr(temp_arch, "21")
        a_record['22'] += find_substr(temp_arch, "02")
        a_record['23'] += find_substr(temp_arch, "23")
        a_record['24'] += find_substr(temp_arch, "24")
        a_record['25'] += find_substr(temp_arch, "25")
        a_record['30'] += find_substr(temp_arch, "30")
        a_record['31'] += find_substr(temp_arch, "31")
        a_record['32'] += find_substr(temp_arch, "32")
        a_record['33'] += find_substr(temp_arch, "33")
        a_record['34'] += find_substr(temp_arch, "34")
        a_record['35'] += find_substr(temp_arch, "35")
        a_record['40'] += find_substr(temp_arch, "40")
        a_record['41'] += find_substr(temp_arch, "41")
        a_record['42'] += find_substr(temp_arch, "42")
        a_record['43'] += find_substr(temp_arch, "43")
        a_record['44'] += find_substr(temp_arch, "44")
        a_record['45'] += find_substr(temp_arch, "45")
        a_record['50'] += find_substr(temp_arch, "50")
        a_record['51'] += find_substr(temp_arch, "51")
        a_record['52'] += find_substr(temp_arch, "52")
        a_record['53'] += find_substr(temp_arch, "53")
        a_record['54'] += find_substr(temp_arch, "54")
        a_record['55'] += find_substr(temp_arch, "55")
    print("------------ initalization finish ------------")



    # 每轮进行迭代更新
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        # 随机采样作为父本群
        while len(sample) < sample_size:
            candidate = random.choice(population)
            sample.append(candidate)

        #设置更新一个自适应参数
        p = len(history) / cycles

        # 选择父本群中验证集acc最高的个体作为变异
        parent = max(sample, key=lambda i: i.val_acc)

        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch_combine_new(parent.arch, np.random.randint(1, 4), s_record, a_record, p)
        child.val_acc, child.test_acc, cnew_arch = train_and_eval_constrain(args, child.arch, data, index)

        # 移除最不适应的个体,如果发现变异过后的个体效果更差则进行局部搜索
        lowest = min(population, key=lambda i: i.val_acc)
        lowest_idx = population.index(min(population, key=lambda i: i.val_acc))


        # 如果发现新产生的变异val-acc更差则略过
        if lowest.val_acc > child.val_acc:
            history.append(child)
            print("this child in %d iteration is wrost than the lowest one in current population, abandon it" %(len(history) - population_size))

        else:
            history.append(child)
            population.pop(lowest_idx)
            population.append(child)
            update_dict_add(child.arch, s_record, a_record, dict_1)
            update_dict_del(lowest.arch, s_record, a_record, dict_1)

            print(cnew_arch)
            print("the %d iteration's res{val_acc:%f   test_acc:%f}" % (
                len(history) - population_size, child.val_acc, child.test_acc))

        temp_val = sorted(history, key=lambda i: i.val_acc, reverse=True)
        temp_val = temp_val[:5]
        temp_val = [x.val_acc for x in temp_val]
        temp_val = np.mean(temp_val)
        temp_var = np.var(temp_val)
        i = len(history)-len(population)
        val_compare[i] = [temp_val,temp_var]

    path = "./val-compare/"
    if not os.path.exists(path):
        os.mkdir(path)
    name = 'DGNAS-PE-res-'+args.data+'.npy'
    name = os.path.join(path,name)
    np.save(name,val_compare)

    return history



# store the search history
# print("-----------round begin-----------")
h = main(500, 20, 3)

acc = {}
d = {}
res = 0
for i in range(len(h)):
    hstr = l2s(h[i].arch)
    acc[hstr] = h[i].val_acc * 100
#根据val-acc将最优的十个结构进行选择
accs = dict(sorted(acc.items(), key= lambda x:x[0],reverse=True))
accs = dict_slice(accs,0,10)
#重复进行5轮实验来验证这十个结构哪个为最优
iteration=5
res =np.zeros(10)
for it in range(iteration):
    num = 0
    for sarch in accs.keys():
        arch = s2l(sarch)
        val_acc, test_acc, cnew_arch = train_and_eval_combine(args, arch, data, index)
        print("this arch:")
        print(cnew_arch)
        print("the val_acc is %f" %(val_acc))
        res[num] += val_acc
        num += 1
res = np.array([i/5 for i in res])
idx = np.argmax(res)

#找到最优结构
best_arch = s2l(list((accs.keys()))[idx])
print("最优结构是:")
print(best_arch)

