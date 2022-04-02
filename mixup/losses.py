import numpy as np
import torch
import logging
# import losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math
import random
import itertools
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

def margin(x, y):
	return x - y

def loss_pos_mixup(sim_mix, a1, pos):
	
	pos_pair_dist = []
	if len(a1) > 0:
		pos_pair_dist = torch.diagonal(sim_mix, 0)

	pos_pairs = (a1, pos)

	if len(pos_pair_dist) > 0:
		pos_loss = torch.nn.functional.relu(margin(pos_pair_dist, 0.0))

	loss_pos_dict =  {
		"pos_loss": {
			"losses": pos_loss,
			"indices": pos_pairs,
			"reduction_type": "pos_pair",
		}
	}

	return loss_pos_dict

def loss_posneg_mixup(sim_mix, pos, a2, neg, lam):
		
	if len(a2) > 0:
		pair_dist = torch.diagonal(sim_mix, 0)

	indices_tuple = (a2, pos, a2, neg)
	pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
	neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)
	
	if len(pair_dist) > 0:
		pos_loss = lam*torch.nn.functional.relu(margin(pair_dist, 0.0))
	if len(pair_dist) > 0:
		neg_loss = (1-lam)*torch.nn.functional.relu(margin(0.5, pair_dist))


	loss_pos_dict =  {
		"pos_loss": {
			"losses": pos_loss,
			"indices": pos_pairs,
			"reduction_type": "pos_pair",
		}
	}

	loss_neg_dict =  {
		"neg_loss": {
			"losses": neg_loss,
			"indices": neg_pairs,
			"reduction_type": "neg_pair",
		},
	}

	return loss_pos_dict, loss_neg_dict

