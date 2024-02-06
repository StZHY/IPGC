from utils.metrics import *
from utils.parser import parse_args
from tqdm import tqdm

import torch
import numpy as np
import multiprocessing
import heapq
from time import time

cores = multiprocessing.cpu_count() // 2

args = parse_args()
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag


def get_cri_entities(user_pos_set, need_ranking_items, rating):

    need_cri_items = get_highest_item(user_pos_set, need_ranking_items, rating)

    need_cri_key = get_cri_keyphrase(user_pos_set, need_cri_items)

    return need_cri_key

def get_highest_item(user_pos_set, need_ranking_items, rating):
    item_score = {}
    for i in need_ranking_items:
        item_score[i] = rating[i]
    
    top_100_items = heapq.nlargest(100, item_score, key = item_score.get)

    cri_items = []

    top_rank_items = top_100_items
    for i in top_rank_items:
        if i not in user_pos_set:
            cri_items.append(i)
    
    return cri_items[:args.item_rank_num]

def get_cri_keyphrase(user_pos_set, need_cri_items):

    pos_keyphrase = get_item_keyphrase(user_pos_set)
    cri_items_keyphrase = get_item_keyphrase(need_cri_items)

    keyphrase_diff = pos_keyphrase - cri_items_keyphrase
    keyphrase_cri_index = np.where(keyphrase_diff < 0)[0]
    only_key_cri_index = keyphrase_cri_index[keyphrase_cri_index >= n_items]
    only_key_cri_index_re = only_key_cri_index.tolist()

    if len(only_key_cri_index) > args.cri_key_rank_num:
        only_key_diff = keyphrase_diff[only_key_cri_index]
        keyphrase_index = heapq.nsmallest(args.cri_key_rank_num, range(len(only_key_diff)), only_key_diff.take)
        only_key_cri_index_re = only_key_cri_index[keyphrase_index].tolist()

    if len(keyphrase_cri_index) == 0:
        only_key_cri_index_re = []

    return only_key_cri_index_re

def get_item_keyphrase(items):
    keyphrases_set = np.zeros(n_entities)
    for item in items:
        key = kg[item]
        keyphrases_set = keyphrases_set + key
    
    return keyphrases_set

def test_one_user(x):
    rating = x[0]
    u = x[1]
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    train_pos_set = train_user_set[u]
    test_pos_items = test_user_set[u]

    all_items = set(range(0, n_items))

    need_ranking_items = list(all_items - set(train_pos_set))

    need_cri_entities = get_cri_entities(test_pos_items, need_ranking_items, rating)

    return {u: need_cri_entities}


def generate_cri_key(model, user_dict, n_params, kg_matrix, args_):
    user_cri_entities = dict()

    global n_users, n_items, n_entities
    n_items = n_params['n_items']
    n_users = n_params['n_users']
    n_entities = n_params['n_entities']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    global kg
    kg = kg_matrix.toarray()

    global args
    args = args_

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    user_gcn_emb, entity_gcn_emb = model.generate()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = min((u_batch_id + 1) * u_batch_size, n_test_users)

        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb(user_batch)

        if batch_test_flag:

            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                i_g_embddings = entity_gcn_emb(item_batch)

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:

            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = entity_gcn_emb(item_batch)
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()
        
        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            user_cri_entities.update(re)
        
    assert count == n_test_users
    pool.close()
    return user_cri_entities