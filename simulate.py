import random
import os
import sys
import torch
import numpy as np
import networkx as nx

from time import time
from datetime import datetime
from prettytable import PrettyTable
from tqdm import tqdm

from generate_cri import generate_cri_key
from utils.parser import parse_args
from utils.cri_data_loader import load_data, load_kg
from utils.cri_evaluate import test
from utils.helper import early_stopping

from modules.IPGC import Critique

from optimizer_utils.omega_utils import init_reg_params
from optimizer_utils.optimizer_lib import Local_Adam, omega_update


class Logger(object):
    def __init__(self, logFile = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

""" feed_dict for generating omega"""
def generate_omega_feed(train_cf_pairs, start, end):

    feed_dict = {}
    entity_pairs = train_cf_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]

    return feed_dict

def rand_walk(cri_key, entity2item, item2entity, walk_steps):

    """ random cri cf """
    cri_user_cf = []

    for user, keys in cri_key.items():
        for key in keys:
            
            if key not in entity2item:
                continue

            around_items = entity2item[key]
            if len(around_items) > args.rand_item_num:
                random_items = random.sample(around_items, args.rand_item_num)
            else: 
                random_items = around_items
            
            probability = args.r_probability

            for random_item in random_items:
                is_muti_walk = np.random.choice([False, True], p=[1-probability, probability])
                if walk_steps > 1 and is_muti_walk:
                    if random_item in item2entity:
                        rand_entity = random.choice(item2entity[random_item])
                        if rand_entity in entity2item:
                            random_item = random.choice(entity2item[rand_entity])

                cri_user_cf.append([user, random_item])
    
    return cri_user_cf

def get_cri_feed_dict(cri_user_cf, start, end, train_user_set):

    def postive_sampling(user_item, train_user_set):
        pos_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            user_pos_items = train_user_set[user]
            pos_item = random.choice(user_pos_items)
            pos_items.append(pos_item)
        return pos_items

    feed_dict = {}
    entity_pairs = cri_user_cf[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['neg'] = entity_pairs[:, 1]
    feed_dict['pos'] = torch.LongTensor(postive_sampling(entity_pairs, train_user_set)).to(device)
    return feed_dict


if __name__ == "__main__":

    """fix the random seed"""
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """log"""
    if args.training_log:
        now = datetime.now().strftime("%Y%m%d_%H%M")
        dataset_name = args.dataset
        sys.stdout = Logger('training_log/'+ dataset_name + ' ' + now + '.log')

    """print args important info"""
    print(os.path.basename(__file__))
    print("cri_lr: " + str(args.cri_lr) + " cri_key_rank_num: " + str(args.cri_key_rank_num) + " rand_item_num: " + str(args.rand_item_num) 
          + " walk_steps: " + str(args.walk_steps) + " reg_lambda: "+str(args.reg_lambda))

    """load train_data & test data & user_dict & params"""
    train_cf, test_cf, user_dict = load_data(args)

    """ cf to tensor"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))
    
    """load knowledge graph & key-item & item-key dict"""
    kg_graph, entity2item, item2entity, items_entity_mat, n_params = load_kg(args)
    
    """load base model"""
    kgin_save_path = 'weights/model_last-fm.pkl'
    kgin_model = torch.load(kgin_save_path, map_location=device)

    entity_embedding, user_embedding = kgin_model.generate()

    """critiquing model"""
    cri_model = Critique(args, user_embedding, entity_embedding).to(device)

    """init omega if using"""
    if args.using_omega:
        cri_model = init_reg_params(cri_model, args)

    """show the scores of based kgin model"""
    base_s_t = time()
    ret = test(cri_model, user_dict, n_params)
    base_e_t = time()

    base_res = PrettyTable()
    base_res.field_names = ["evaluating time", "recall", "ndcg", "precision", "hit_ratio"]
    base_res.add_row(
        [base_e_t - base_s_t, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
    )
    print(base_res)

    """ generate omega """
    if args.using_omega:
        cri_optimizer_ft = omega_update(cri_model.reg_params)
        cri_model.eval()
        Omega_s = 0
        while Omega_s + args.batch_size <= len(train_cf):
            batch = generate_omega_feed(train_cf_pairs, Omega_s, Omega_s + args.batch_size)
            cri_optimizer_ft.zero_grad()
            output = cri_model.create_output(batch)
            squared_output = torch.pow(output, 2)
            sum_norm = torch.sum(squared_output)
            sum_norm.backward()

            if Omega_s + args.batch_size > len(train_cf):
                cri_optimizer_ft.step(cri_model.reg_params, Omega_s, len(train_cf) - Omega_s, args)
            else:
                cri_optimizer_ft.step(cri_model.reg_params, Omega_s, args.batch_size, args)
            Omega_s += args.batch_size
            
        """ define critique optimizer """
        cri_model.train()
        cri_optimizer = Local_Adam(cri_model.reg_params, args.reg_lambda, lr=args.cri_lr)
    else:
        """not using omega"""
        cri_model.train()
        cri_optimizer = torch.optim.Adam(cri_model.parameters(), lr=args.cri_lr)

    for cri_epoch in range(args.cri_epoch):
        """ start critiquing """
        simulate_critiquing_s_t = time()

        """ simulate generating users' critique keyphrases """
        critiquing_key = generate_cri_key(cri_model, user_dict, n_params, items_entity_mat, args)

        """ random walk """
        randwalk_cri_cf = rand_walk(critiquing_key, entity2item, item2entity, args.walk_steps)
        cri_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in randwalk_cri_cf], np.int32))

        cri_loss, cri_s = 0, 0
        while cri_s + args.cri_batch_size <= len(cri_cf_pairs):
                    
            cri_batch = get_cri_feed_dict(cri_cf_pairs, cri_s, cri_s + args.cri_batch_size, user_dict['train_user_set'])
            bpr_loss = cri_model(cri_batch)

            cri_optimizer.zero_grad()
            bpr_loss.backward()
            if args.using_omega:
                cri_optimizer.step(cri_model.reg_params, args)
            else:
                cri_optimizer.step()

            cri_loss += bpr_loss
            cri_s += args.cri_batch_size

        """ending time """
        simulate_critiquing_e_t = time()

        simulate_score_s_t = time()
        simulate_ret = test(cri_model, user_dict, n_params)
        simulate_score_e_t = time()

        print('using Omega: ' + str(args.using_omega))
        simulate_res = PrettyTable()
        simulate_res.field_names = ["Cri_Epoch", "critiquing time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
        simulate_res.add_row(
            [cri_epoch, simulate_critiquing_e_t - simulate_critiquing_s_t, simulate_score_e_t - simulate_score_s_t, cri_loss.item(),\
             simulate_ret['recall'], simulate_ret['ndcg'], simulate_ret['precision'], simulate_ret['hit_ratio']]
            )
        print(simulate_res)