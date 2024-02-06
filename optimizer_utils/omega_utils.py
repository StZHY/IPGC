import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


def init_reg_params(model, args):
	device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

	reg_params = {}

	for name, param in model.named_parameters():
		if name == 'user_emb.weight':
			
			print ("Initializing omega values for layer", name)
			omega = torch.zeros(param.size())
			omega = omega.to(device)

			init_val = param.data.clone()
			param_dict = {}

			param_dict['omega'] = omega
			param_dict['init_val'] = init_val

			reg_params[param] = param_dict

	model.reg_params = reg_params

	return model 
