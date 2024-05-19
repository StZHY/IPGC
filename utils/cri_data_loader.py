import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import pickle

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd

def build_sparse_item2entity_graph(relation_dict):

    def counting_interact(adj_mat_list):
        count_mat = adj_mat_list[0].tocsr()
        for mat in adj_mat_list[1:]:
            mat = mat.tocsr()
            count_mat = count_mat + mat

        return count_mat.tocoo()
    
    def counting_nhop_interact(kg_mat):
        if args.count_nhop == 1:
            nhop_mat = kg_mat.tocsr()[:n_items, :].tocoo()
        else:
            nhop = args.count_nhop - 1
            nhop_mat = kg_mat.tocsr()
            mat = kg_mat.tocsr()
            for i in range(nhop):
                nhop_mat = nhop_mat.dot(mat)
            nhop_mat = nhop_mat[:n_items, :].tocoo()

        return nhop_mat
    
    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        vals = [1.] * len(np_mat)
        adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_entities, n_entities))
        adj_mat_list.append(adj)
    
    kg_mat = counting_interact(adj_mat_list)
    kg_item2entity_mat = counting_nhop_interact(kg_mat)

    return kg_item2entity_mat

def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict

def load_kg(model_args):
    """entity2item & item2entity"""
    entity2item = defaultdict(list)
    item2entity = defaultdict(list)

    global args 
    args = model_args
    directory = args.data_path + args.dataset + '/'

    kg_triplets = read_triplets(directory + 'kg_final.txt')

    kg_graph, rd = build_graph(kg_triplets)

    item_entity_mat = build_sparse_item2entity_graph(rd)

    for h_id, r_id, t_id in tqdm(kg_triplets, ascii=True):
        if h_id >= n_items and t_id < n_items:
            entity2item[h_id].append(t_id)
            item2entity[t_id].append(h_id)
        if h_id < n_items and t_id >= n_items:
            entity2item[t_id].append(h_id)
            item2entity[h_id].append(t_id)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }

    return kg_graph, entity2item, item2entity, item_entity_mat, n_params

    
