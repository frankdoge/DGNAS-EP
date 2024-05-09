import random

import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

hidden_units = [32, 64, 128, 256]

#即正则化后的GCN的聚集操作层
class Graph(nn.Module):
    def __init__(self, adj):
        super(Graph, self).__init__()
        self.adj = adj
    #即利用正则化矩阵与H(l-1)相乘得到H(l)，T操作只改变H(l)维度
    def forward(self, x):
        x = self.adj.matmul(x)
        return x



class MLP(nn.Module):
    def __init__(self, nfeat, nclass, dropout, last=False):
        super(MLP, self).__init__()
        self.lr1 = nn.Linear(nfeat, nclass)
        self.dropout = dropout
        self.last = last

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lr1(x)
        if not self.last:
            x = F.relu(x)
        return x

#只由P-GCN(聚集)和T(MLP)操作构成
class ModelOp(nn.Module):
    def __init__(self, arch, adj, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp, self).__init__()
        self._ops = nn.ModuleList()
        #默认第一个T预处理也为一个P层
        self._numP = 1
        self._arch = arch
        for element in arch:
            #P为1,T为0
            if element == 1:
                op = Graph(adj)
                self._numP += 1
            elif element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
            else:
                print("arch element error")
            self._ops.append(op)
        #将gate按P层的层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5*torch.randn(self._numP), requires_grad=True)
        #一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)
    
    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT  = []
        for i in range(len(self._arch)):
            if i == 0:
                #所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1
                
                res = self._ops[i](res)
                if self._arch[i] == 1:
                    #如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    #如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            #针对第二层起，要考虑其上一层是什么操作
            else:
                #当上一层为P时，进入门控机制
                if self._arch[i - 1] == 1:
                    #如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    #如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        #记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                #如果上一层是T
                else:
                    #当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    #如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            #这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits


class ModelOp_change(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_change, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        self.new_arch=[]
        for element in arch:
            # P为1,T为0
            if element == 1:
                #这里对三种无参传播矩阵进行随机选择
                idx = random.randint(0,3)
                if idx == 0:
                    op = Graph(adj_nor)
                    self.new_arch.append(2)
                elif idx == 1:
                    op = Graph(adj_com)
                    self.new_arch.append(3)
                else:
                    op = Graph(adj_sing)
                    self.new_arch.append(4)
                self._numP += 1
            elif element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
                self.new_arch.append(0)
            else:
                print("arch element error")

            # if element == 0:
            #     op = MLP(hid_dim, hid_dim, dropout)
            # else:
            #     #代表nor
            #     if element == 2:
            #         op = Graph(adj_nor)
            #     #代表sing
            #     elif element == 3:
            #         op = Graph(adj_sing)
            #     #代表com
            #     else:
            #         op = Graph(adj_com)
            #     self._numP += 1
            self._ops.append(op)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] == 1:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] == 1:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits


class ModelOp_change_new(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_change_new, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        # self._arch = arch
        self._arch = []
        for i in arch:
            if i == 0:
                self._arch.append(0)
            else:
                self._arch.append(1)
        for element in arch:
            # P为1,T为0
            idx = element
            if element != 0:
                #这里对三种无参传播矩阵进行随机选择
                if idx == 2:
                    op = Graph(adj_nor)
                elif idx == 3:
                    op = Graph(adj_com)
                else:
                    op = Graph(adj_sing)
                self._numP += 1
            else:
                op = MLP(hid_dim, hid_dim, dropout)

            self._ops.append(op)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] == 1:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] == 1:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits


class ModelOp_padd(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing, adj_row, adj_col, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_padd, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        self.new_arch=[]
        for element in arch:
            # P为1,T为0
            if element == 1:
                #这里对三种无参传播矩阵进行随机选择
                idx = random.randint(0,4)
                if idx == 0:
                    op = Graph(adj_nor)
                    self.new_arch.append(2)
                elif idx == 1:
                    op = Graph(adj_com)
                    self.new_arch.append(3)
                elif idx == 2:
                    op = Graph(adj_sing)
                    self.new_arch.append(4)
                elif idx == 3:
                    op = Graph(adj_row)
                    self.new_arch.append(5)
                else:
                    op = Graph(adj_col)
                    self.new_arch.append(6)
                self._numP += 1
            elif element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
                self.new_arch.append(0)
            else:
                print("arch element error")

            # if element == 0:
            #     op = MLP(hid_dim, hid_dim, dropout)
            # else:
            #     #代表nor
            #     if element == 2:
            #         op = Graph(adj_nor)
            #     #代表sing
            #     elif element == 3:
            #         op = Graph(adj_sing)
            #     #代表com
            #     else:
            #         op = Graph(adj_com)
            #     self._numP += 1
            self._ops.append(op)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] == 1:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] == 1:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits


class ModelOp_Tattetion(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_Tattetion, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._numT = 0
        self._arch = arch
        self.new_arch=[]
        for element in arch:
            # P为1,T为0
            if element == 1:
                #这里对三种无参传播矩阵进行随机选择
                idx = random.randint(0,3)
                if idx == 0:
                    op = Graph(adj_nor)
                    self.new_arch.append(2)
                elif idx == 1:
                    op = Graph(adj_com)
                    self.new_arch.append(3)
                else:
                    op = Graph(adj_sing)
                    self.new_arch.append(4)
                self._numP += 1
            elif element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
                self.new_arch.append(0)
                self._numT += 1
            else:
                print("arch element error")
            self._ops.append(op)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        # self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 增加一个对于T层的attention机制
        self.gateT = torch.nn.Parameter(1e-5 * torch.randn(self._numT), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        numT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] == 1:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numT.append(i)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] == 1:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        # res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = sum(tempP)
                        res = self._ops[i](res)
                        tempT.append(res)
                        numT.append(i)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum([torch.mul(F.sigmoid(self.gateT[i]), tempT[i]) for i in range(len(numT))])
                        res = self._ops[i](res)
                        numT.append(i)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            # res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
            res = sum(tempP)
        else:
            res = sum([torch.mul(F.sigmoid(self.gateT[i]), tempT[i]) for i in range(len(numT))])
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits

class ModelOp_Tchange(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_Tchange, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        #记录第i个T层的输出维度以便concat
        self.new_arch=[]
        self.input_dim = hid_dim
        self.dropout=dropout
        for element in arch:
            # P为1,T为0
            if element != 0:
                # #这里对三种无参传播矩阵进行随机选择
                # idx = random.randint(0,2)
                # if idx == 0:
                #     op = Graph(adj_nor)
                #     self.new_arch.append(2)
                # elif idx == 1:
                #     op = Graph(adj_com)
                #     self.new_arch.append(3)
                # else:
                #     op = Graph(adj_sing)
                #     self.new_arch.append(4)
                idx = element
                if idx == 1:
                    op = Graph(adj_nor)
                elif idx == 2:
                    op = Graph(adj_com)
                else:
                    op = Graph(adj_sing)
                self.new_arch.append(idx)
                self._numP += 1
            elif element == 0:
                output_idx = random.randint(0, 3)
                output_dim = hidden_units[output_idx]
                op = MLP(self.input_dim, output_dim, dropout)
                self.input_dim = output_dim
                self.new_arch.append(self.input_dim)
            self._ops.append(op)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(self.input_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] != 0:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] != 0:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] != 0:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        temp_dim = 0
                        for j in tempT:
                            temp_dim += j.size(1)
                        res = torch.concat(tempT, 1)
                        tMLP = MLP(temp_dim, self.new_arch[i-1], self.dropout).cuda()
                        res = tMLP(res)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            temp_dim = 0
            for j in tempT:
                temp_dim += j.size(1)
            res = torch.concat(tempT,1)
            tMLP=MLP(temp_dim,self.new_arch[i],self.dropout).cuda()
            res=tMLP(res)

        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits

class ModelOp_constrain(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing, adj_row, adj_col, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_constrain, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        self.new_arch = []
        for element in arch:
            # P为1,T为0
            if element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
            else:
                # 这里对三种无参传播矩阵进行频率选择
                # p_list = []
                # p_list.extend(s_list[1:])
                # a = [1,2,3]
                # idx = np.random.choice(a,p=p_list)
                idx = element
                if idx == 1:
                    op = Graph(adj_nor)
                elif idx == 2:
                    op = Graph(adj_com)
                else:
                    op = Graph(adj_sing)
                self._numP += 1
            self._ops.append(op)
            self.new_arch.append(element)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] != 0:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] != 0:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] != 0:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits



class ModelOp_combine(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing,  adj_row, adj_col, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_combine, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        # 记录第i个T层的输出维度以便concat
        self.new_arch = []
        self.input_dim = hid_dim
        self.dropout = dropout
        for element in arch:
            # P为1,T为0
            if element == 0:
                output_idx = random.randint(0, 3)
                output_dim = hidden_units[output_idx]
                op = MLP(self.input_dim, output_dim, dropout)
                self.input_dim = output_dim
                self.new_arch.append(self.input_dim)
            else:
                idx = element
                if idx == 1:
                    op = Graph(adj_nor)
                elif idx == 2:
                    op = Graph(adj_com)
                elif idx == 3:
                    op = Graph(adj_sing)
                elif idx == 4:
                    op = Graph(adj_row)
                else:
                    op = Graph(adj_col)
                self._numP += 1
                self.new_arch.append(idx)
            self._ops.append(op)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(self.input_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] != 0:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] != 0:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] != 0:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        temp_dim = 0
                        for j in tempT:
                            temp_dim += j.size(1)
                        res = torch.concat(tempT, 1)
                        tMLP = MLP(temp_dim, self.new_arch[i - 1], self.dropout).cuda()
                        res = tMLP(res)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            temp_dim = 0
            for j in tempT:
                temp_dim += j.size(1)
            res = torch.concat(tempT, 1)
            tMLP = MLP(temp_dim, self.new_arch[i], self.dropout).cuda()
            res = tMLP(res)

        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits

class ModelOp_combine_withoutT(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing,  adj_row, adj_col, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_combine_withoutT, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        # 记录第i个T层的输出维度以便concat
        self.new_arch = []
        self.dropout = dropout
        for element in arch:
            # P为1,T为0
            if element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
            else:
                idx = element
                if idx == 1:
                    op = Graph(adj_nor)
                elif idx == 2:
                    op = Graph(adj_com)
                elif idx == 3:
                    op = Graph(adj_sing)
                elif idx == 4:
                    op = Graph(adj_row)
                else:
                    op = Graph(adj_col)
                self._numP += 1
            self._ops.append(op)
            self.new_arch.append(element)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] != 0:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] != 0:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        # res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = sum([torch.mul(self.gate[totalP - len(numP) + i], tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] != 0:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            # res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
            res = sum([torch.mul(self.gate[totalP - len(numP) + i], tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        last_feature = res
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits,last_feature

class ModelOp_combine_contrast(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing,  adj_row, adj_col, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_combine_contrast, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        # 记录第i个T层的输出维度以便concat
        self.new_arch = []
        self.dropout = dropout
        for element in arch:
            # P为1,T为0
            if element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
            else:
                idx = element
                if idx == 1:
                    op = Graph(adj_nor)
                elif idx == 2:
                    op = Graph(adj_com)
                elif idx == 3:
                    op = Graph(adj_sing)
                elif idx == 4:
                    op = Graph(adj_row)
                else:
                    op = Graph(adj_col)
                self._numP += 1
            self._ops.append(op)
            self.new_arch.append(element)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] != 0:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] != 0:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        # res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = sum([torch.mul(self.gate[totalP - len(numP) + i], tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] != 0:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            # res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
            res = sum([torch.mul(self.gate[totalP - len(numP) + i], tempP[i]) for i in numP])
        else:
            res = sum(tempT)

        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits


class ModelOp_combine_withoutPadd(nn.Module):
    def __init__(self, arch, adj_nor, adj_com, adj_sing, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp_combine_withoutPadd, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        # 记录第i个T层的输出维度以便concat
        self.new_arch = []
        self.input_dim = hid_dim
        self.dropout = dropout
        for element in arch:
            # P为1,T为0
            if element == 0:
                output_idx = random.randint(0, 3)
                output_dim = hidden_units[output_idx]
                op = MLP(self.input_dim, output_dim, dropout)
                self.input_dim = output_dim
                self.new_arch.append(self.input_dim)
            else:
                idx = element
                if idx == 1:
                    op = Graph(adj_nor)
                elif idx == 2:
                    op = Graph(adj_com)
                elif idx == 3:
                    op = Graph(adj_sing)
                self._numP += 1
                self.new_arch.append(idx)
            self._ops.append(op)
        # 将gate按层数分配一个概率并且加入model的参数中跟随更新
        self.gate = torch.nn.Parameter(1e-5 * torch.randn(self._numP), requires_grad=True)
        # 一开始先使用一个T操作(MLP)来对特征进行处理
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(self.input_dim, num_classes, dropout, True)

    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT = []
        for i in range(len(self._arch)):
            if i == 0:
                # 所有结构都要先进行一次针对特征的预处理
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1

                res = self._ops[i](res)
                if self._arch[i] != 0:
                    # 如果第一个的操作为P，记录下来用于之后的T操作的之前P层的输入
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    # 如果第一个操作为T，重置预处理层的numP和tempP，因为前面没有P层
                    tempT.append(res)
                    numP = []
                    tempP = []
            # 针对第二层起，要考虑其上一层是什么操作
            else:
                # 当上一层为P时，进入门控机制
                if self._arch[i - 1] != 0:
                    # 如果当前层为T，论文中写的将之前的P层输入累加,其中sigmoid部分就是论文里的小s
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    # 如果当前层为P，直接对上一层输出进行P操作即可
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        # 记录上一次T-P结构后第一个P的层数索引与当前P层索引的差值
                        numP.append(i - point)
                        totalP += 1
                # 如果上一层是T
                else:
                    # 当前层是P，直接进行P操作并从当前层开始重新记录numP和tempP
                    if self._arch[i] != 0:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    # 如果当前是T，则直接把之前的T层之和当作输入
                    else:
                        temp_dim = 0
                        for j in tempT:
                            temp_dim += j.size(1)
                        res = torch.concat(tempT, 1)
                        tMLP = MLP(temp_dim, self.new_arch[i - 1], self.dropout).cuda()
                        res = tMLP(res)
                        res = self._ops[i](res)
                        tempT.append(res)

        if len(numP) > 0 or len(tempP) > 0:
            # 这儿的i要么是跟着之前的numP一起，要么就是遇到T-P操作后重置为0开始，能保证在总层数内
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            temp_dim = 0
            for j in tempT:
                temp_dim += j.size(1)
            res = torch.concat(tempT, 1)
            tMLP = MLP(temp_dim, self.new_arch[i], self.dropout).cuda()
            res = tMLP(res)

        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits





