import sys
import time
import torch
import argparse
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
import os
import pickle
import re

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_normalized_adjacency(adj):
    """
    Normalize adjacency matrix.
    即得到GCN传播操作中那个正则化的A^矩阵
    """
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def aug_row_normalizaed_adjacency(adj):
    """
        即得到行正则化矩阵
    """
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()

def aug_col_normalizaed_adjacency(adj):
    """
        即得到列正则化矩阵
    """
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).tocoo()

def aug_compare_adjacency(adj):
    """
    changing adjacency matrix.
    即得到传播操作中那个未加入单位矩阵的A矩阵
    """
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def aug_random_walk(adj):
    """Random Walk algorithm."""
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def laplacian(mx, norm):
    """Laplacian-normalize sparse matrix"""
    assert (all(len(row) == len(mx) for row in mx)), "Input should be a square matrix"

    return csgraph.laplacian(adj, normed=norm)

def accuracy(output, labels):
    #根据最后的值找argmax来作为预测的标签，然后判断预测正确的样本量
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

#序列和字符串转换
def l2s(arch):
    a_new = [str(x) for x in arch]
    a = "".join(a_new)
    return a

def s2l(str1):
    a = []
    for ch in str1:
        a.append(int(ch))
    return a

# 找到子串次数(不可重叠)
def find_substr(arch,substr):
    a_new = [str(x) for x in arch]
    a = "".join(a_new)
    def is_in(fullstr, str2):
        rlist = re.findall(str2, fullstr)
        return len(rlist)
    n = is_in(a, substr)
    return n

# 找到可重叠匹配的子串
def find_substr_all(arch,substr):
    a_new = [str(x) for x in arch]
    a = "".join(a_new)
    count = 0
    start = 0
    # 通过不断缩短主串来重叠匹配
    while start < len(a):
        flag = a.find(substr, start)
        # 把找到的idx下一个当做新的start
        if flag != -1:
            start = flag + 1
            count += 1
        else:
            return count
    return count

# 根据字典找到频率转换为列表
def get_pro(dict1):
    total = 0
    for i in dict1.values():
        total += i
    res = []

    for i in dict1.keys():
        res.append(float(dict1[i]/total))
    return res

# 根据输入的arch考虑删去后两个字典的变化
def update_dict_del(arch, s_dict, a_dict, dict_1):
    # 更新s_dict
    for i in range(len(s_dict)):
        j = find_substr(arch, str(i))
        # 约束为正值
        if s_dict[str(i)] > j:
            s_dict[str(i)] -= j
        else:
            s_dict[str(i)] = 0

    # 更新a_dict
    for i in dict_1.keys():
        # 找到当前的子序列
        temp = dict_1[i]
        j = find_substr(arch, temp)
        if a_dict[temp] > j:
            a_dict[temp] -= j
        else:
            a_dict[temp] = 0


def update_dict_add(arch, s_dict, a_dict, dict_1):
    # 更新s_dict
    for i in range(len(s_dict)):
        j = find_substr(arch, str(i))
        s_dict[str(i)] += j

    # 更新a_dict
    for i in dict_1.keys():
        # 找到当前的子序列
        temp = dict_1[i]
        j = find_substr(arch, temp)
        a_dict[temp] += j

#对字典进行切片，切片范围为[start,end)
def dict_slice(dict1, start, end):
    new_key = dict1.keys()
    new_dict = {}
    for k in list(new_key)[start:end]:
        new_dict[k] = dict1[k]
    return new_dict


def load_big_data(dataset_name):
    path_adj=os.path.join('./graphs/'+dataset_name+'_adj.pkl')
    path_feature=os.path.join('./graphs/'+dataset_name+'_features.pkl')
    path_label=os.path.join('./graphs/'+dataset_name+'_labels.pkl')
    path_split=os.path.join('./graphs/'+dataset_name+'_tvt_nids.pkl')
    f1 = open(path_adj, 'rb')
    f2 = open(path_feature, 'rb')
    f3 = open(path_label, 'rb')
    f4 = open(path_split, 'rb')
    adj = pickle.load(f1)
    features = pickle.load(f2)

    labels = pickle.load(f3)
    split = pickle.load(f4)
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum()))

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    print("| # of features : {}".format(num_features))
    print("| # of clases   : {}".format(num_labels))

    if dataset_name != 'airport':
        features = features.A
        features = normalize(features)
        features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    idx_train = split[0]
    idx_val = split[1]
    idx_test = split[2]
    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))
    return adj, features, labels, idx_train, idx_val, idx_test


def load_new_data(dataset_name, splits_file_path=None):
    #根据给出的两个txt文件来载入对应的邻接表和特征及标签
    graph_adjacency_list_file_path = os.path.join('../new_data', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('../new_data', dataset_name,
                                                            'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_name == 'film':
        #对于film数据集而言，其特征是标记出为1的下标索引
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                #通过置位符把line划分为三个部分，分别为[id,非零特征列表,标签号]
                line = line.rstrip().split('\t')
                #判断是否符合输入格式
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                #将所有置零，然后把为1的特征下标置1
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])

    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                #不同之处在于直接给出特征的列表，导入即可
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])
    #再把(u,v)边信息导入有向图G中
    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    #打印图的信息
    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    #用于特征行正则化
    features = normalize(features)

    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    print("| # of features : {}".format(num_features))
    print("| # of clases   : {}".format(num_labels))

    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))
    #将train_mask等转化为idx_train
    idx_train = [i for i,e in enumerate(train_mask) if e != 0]
    idx_val = [i for i, e in enumerate(val_mask) if e != 0]
    idx_test = [i for i, e in enumerate(test_mask) if e != 0]
    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)


    return adj, features, labels, idx_train, idx_val, idx_test

# copy from paper--Demystifying Graph Neural Network Via Graph Filter Assessment
def get_division_score(H, Y_onehot, nb_each_class_inv_mat, lmd = 100, norm_or_not = True):
    '''
    H is representation matrix of dim n * d
    Y_onehot is a one-hot matrix of dim n * c, c is number of class
    nb_each_class_inv_mat is a diagonal matrix of dim c * c
    this loss encourage node in different class to be as linear seperable as possible
    '''

    # def norm_expand_tensor(feature):
    #     assert len(feature.shape) == 2
    #     mean = feature.mean(dim=0, keepdim=True)
    #     var = feature.std(dim=0, keepdim=True)
    #     return (feature - mean) / (var + 1e-6)

    # if norm_or_not:
    #     H = norm_expand_tensor(H)

    #     step1: get shape
    nb_nodes = Y_onehot.shape[0]
    nb_class = Y_onehot.shape[1]

    #计算归一化系数
    nb_combine = nb_class*(nb_class-1)/2

    #     step2: get mean_mat, each column is a mean vector for a class
    H_T = torch.transpose(H, 0, 1)  # transpose of matrix H
    sum_mat = torch.mm(H_T, Y_onehot)  # d * c
    mean_mat = torch.mm(sum_mat, nb_each_class_inv_mat)  # d * c

    mean_mat = mean_mat.cpu().detach().numpy()
    result = 0
    #     step3: for each pair, get score and sum as result
    for i in range(nb_class):
        for j in range(i + 1, nb_class):
            score = lmd*(np.linalg.norm(mean_mat[:,i] - mean_mat[:,j],ord=2))/(np.exp(np.linalg.norm(mean_mat[:,i],ord=2)+np.linalg.norm(mean_mat[:,j],ord=2)))
            score= score/nb_combine
            result = result + score
    
    return result
def load_data(path="../data", dataset="cora"):
    """
    三大数据集的各项文件的含义
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    print("\n[STEP 1]: Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            #判断版本号是否大于3.0
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Citeseer dataset contains some isolated nodes in the graph
        #针对citeseer孤立节点，将其加入测试集，共有15个孤立节点
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum() / 2))

    features = normalize(features)
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #保存每个节点的groundtruth，比如citeseer是多分类(6类)
    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)

    return adj, features, labels, idx_train, idx_val, idx_test

def graph_decompose(adj,graph_name,k,metis_p,strategy="edge"):
    '''
    Input:
        adj:the adjacency matrix of original graph
        graph_name:"cora","citeseer","pubmed"
        k:decompose into k subgraphs
        metis_p:"no_skeleton","all_skeleton","number" (depending on metis preprocessing) 
        strategy:"edge" (for edge_decomposition),"node" (for node_decomposition)
    Output:
        the decomposed subgraphs
    '''
    print("Skeleton:",metis_p)
    print("Strategy:",strategy)
    g,g_rest,edges_rest,gs=get_graph_skeleton(adj,graph_name,k,metis_p)
    gs=allocate_edges(g_rest,edges_rest, gs, strategy)
       
    re=[]       
   
    #print the info of nodes and edges of subgraphs 
    edge_num_avg=0
    compo_num_avg=0
    print("Subgraph information:")
    for i in range(k):
        nodes_num=gs[i].number_of_nodes()
        edge_num=gs[i].number_of_edges()
        compo_num=nx.number_connected_components(gs[i])
        print("\t",nodes_num,edge_num,compo_num)
        edge_num_avg+=edge_num
        compo_num_avg+=compo_num
        re.append(nx.to_scipy_sparse_matrix(gs[i])) 
        
    #check the shared edge number in all subgrqphs
    edge_share=set(sort_edge(gs[0].edges()))
    for i in range(k):        
        edge_share&=set(sort_edge(gs[i].edges()))
        
    print("\tShared edge number is: %d"%len(edge_share))
    print("\tAverage edge number:",edge_num_avg/k) 
    print("\tAverage connected component number:",compo_num_avg/k)
    print("\n"+"-"*70+"\n")
    return re

def sort_edge(edges):
    edges=list(edges)
    for i in range(len(edges)):
        u=edges[i][0]
        v=edges[i][1]
        if u > v:
            edges[i]=(v,u)
    return edges

def get_graph_skeleton(adj,graph_name,k,metis_p): 
    '''
    Input:
        adj:the adjacency matrix of original graph
        graph_name:"cora","citeseer","pubmed"
        k:decompose into k subgraphs
        metis_p:"no_skeleton","all_skeleton","k" 
    Output:
        g:the original graph
        g_rest:the rest graph
        edges_rest:the rest edges
        gs:the skeleton of the graph for every subgraph
    '''
    g=nx.from_numpy_matrix(adj.todense())
    num_nodes=g.number_of_nodes()
    print("Original nodes number:",num_nodes)
    num_edges=g.number_of_edges()
    print("Original edges number:",num_edges)  
    print("Original connected components number:",nx.number_connected_components(g),"\n")    
    
    g_dic=dict()
    
    for v,nb in g.adjacency():
        g_dic[v]=[u[0] for u in nb.items()] 
            
    #initialize all the subgrapgs, add the nodes
    gs=[nx.Graph() for i in range(k)]
    for i in range(k):
        gs[i].add_nodes_from([i for i in range(num_nodes)])
    
    if metis_p=="no_skeleton":
        #no skeleton
        g_rest=g
        edges_rest=list(g_rest.edges())
    else:    
        if metis_p=="all_skeleton":
            #doesn't use metis to cut any edge
            graph_cut=g
        else:
            #read the cluster info from file
            f=open("metis_file/"+graph_name+".graph.part.%s"%metis_p,'r')
            cluster=dict()  
            i=0
            for lines in f:
                cluster[i]=eval(lines.strip("\n"))
                i+=1
           
            #get the graph cut by Metis    
            graph_cut=nx.Graph()
            graph_cut.add_nodes_from([i for i in range(num_nodes)])  
            
            for v in range(num_nodes):
                v_class=cluster[v]
                for u in g_dic[v]:
                    if cluster[u]==v_class:
                        graph_cut.add_edge(v,u)
            
        subgs=list(nx.connected_component_subgraphs(graph_cut))
        print("After Metis,connected component number:",len(subgs))
        
                
        #add the edges of spanning tree, get the skeleton
        for i in range(k):
            for subg in subgs:
                T=get_spanning_tree(subg)
                gs[i].add_edges_from(T)
        
        #get the rest graph including all the edges except the shared egdes of spanning trees
        edge_set_share=set(sort_edge(gs[0].edges()))
        for i in range(k):
            edge_set_share&=set(sort_edge(gs[i].edges()))
        edge_set_total=set(sort_edge(g.edges()))
        edge_set_rest=edge_set_total-edge_set_share   
        edges_rest=list(edge_set_rest)
        g_rest=nx.Graph()
        g_rest.add_nodes_from([i for i in range(num_nodes)])
        g_rest.add_edges_from(edges_rest)
       
          
    #print the info of nodes and edges of subgraphs
    print("Skeleton information:")
    for i in range(k):
        print("\t",gs[i].number_of_nodes(),gs[i].number_of_edges(),nx.number_connected_components(gs[i])) 
        
    edge_set_share=set(sort_edge(gs[0].edges()))
    for i in range(k):
        edge_set_share&=set(sort_edge(gs[i].edges()))
    print("\tShared edge number is: %d\n"%len(edge_set_share))
    
    return g,g_rest,edges_rest,gs

def get_spanning_tree(g):
    '''
    Input:Graph
    Output:list of the edges in spanning tree
    '''
    g_dic=dict()
    for v,nb in g.adjacency():
        g_dic[v]=[u[0] for u in nb.items()]
        np.random.shuffle(g_dic[v])
    flag_dic=dict()
    if g.number_of_nodes() ==1:
        return []
    gnodes=np.array(g.nodes)
    np.random.shuffle(gnodes)
    
    for v in gnodes:
        flag_dic[v]=0
    
    current_path=[]
    
    def dfs(u):
        stack=[u]
        current_node=u
        flag_dic[u]=1
        while len(current_path)!=(len(gnodes)-1):
            pop_flag=1
            for v in g_dic[current_node]:
                if flag_dic[v]==0:
                    flag_dic[v]=1
                    current_path.append((current_node,v))  
                    stack.append(v)
                    current_node=v
                    pop_flag=0
                    break
            if pop_flag:
                stack.pop()
                current_node=stack[-1]     
    dfs(gnodes[0])        
    return current_path

def allocate_edges(g_rest,edges_rest, gs, strategy):
    '''
    Input:
        g_rest:the rest graph
        edges_rest:the rest edges
        gs:the skeleton of the graph for every subgraph
        strategy:"edge" (for edge_decomposition),"node" (for node_decomposition)
    Output:
        the decomposed graphs after allocating rest edges
    '''
    k=len(gs)
    if strategy=="edge":  
        print("Allocate the rest edges randomly and averagely.")
        np.random.shuffle(edges_rest)
        t=int(len(edges_rest)/k)
        
        #add edges
        for i in range(k):       
            if i == k-1:
                gs[i].add_edges_from(edges_rest[t*i:])
            else:
                gs[i].add_edges_from(edges_rest[t*i:t*(i+1)])        
        return gs
    
    elif strategy=="node":
        print("Allocate the edges of each nodes randomly and averagely.")
        g_dic=dict()    
        for v,nb in g_rest.adjacency():
            g_dic[v]=[u[0] for u in nb.items()]
            np.random.shuffle(g_dic[v])
        
        def sample_neighbors(nb_ls,k):
            np.random.shuffle(nb_ls)
            ans=[]
            for i in range(k):
                ans.append([])
            if len(nb_ls) == 0:
                return ans
            if len(nb_ls) > k:
                t=int(len(nb_ls)/k)
                for i in range(k):
                    ans[i]+=nb_ls[i*t:(i+1)*t]
                nb_ls=nb_ls[k*t:]
            '''
            if len(nb_ls)>0:
                for i in range(k):
                    ans[i].append(nb_ls[i%len(nb_ls)])
            '''
            
            
            if len(nb_ls)>0:
                for i in range(len(nb_ls)):
                    ans[i].append(nb_ls[i])
            
            np.random.shuffle(ans)
            return ans
        
        #add edges
        for v,nb in g_dic.items():
            ls=np.array(sample_neighbors(nb,k))
            for i in range(k):
                gs[i].add_edges_from([(v,j) for j in ls[i]])
        
        return gs
