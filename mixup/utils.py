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


def mixup_data(p, n, alpha=0.4):    
	lam = np.random.beta(alpha, alpha)
	mixed_x = lam * p + (1 - lam) * n

	return mixed_x, lam


def input_pos_pair_mixup(a1, pos, images, target, sim_mat, network):
	anchor_images = images[a1]
	pos_images = images[pos]
	target_anchor = target[a1]

	mixed_pos_images, lam = mixup_data(anchor_images, pos_images, alpha=2.0)

	embedding_anchor_clean = network(anchor_images)
	embedding_pos_mixed = network(mixed_pos_images)

	similarity_mixed = sim_mat(embedding_anchor_clean, embedding_pos_mixed)

	return similarity_mixed, target_anchor


def input_posneg_pair_mixup_for_pos_anchor(a1, pos, a2, neg, clean_embedding, images, target, sim_mat, network, top_k):
	clean_sim = sim_mat(clean_embedding)
	anch_neg, hard_neg = [],[]
	for anchor_pos in a1:
		idxs = torch.where(anchor_pos == a2)[0]
		dist = clean_sim[a2[idxs], neg[idxs]]
		_,indices = torch.sort(dist,descending=False)

		hard_neg.append(neg[indices][:top_k].cpu().numpy()) 
		anch_neg.append(a2[idxs][:top_k].cpu().numpy())

	a2 = list(itertools.chain.from_iterable(anch_neg))
	neg = list(itertools.chain.from_iterable(hard_neg))
	pos = [ele for ele in pos.cpu().numpy() for i in range(top_k)]

	anchor_images = images[a2]
	pos_images = images[pos]
	neg_images = images[neg]

	target_pos = target[pos]
	target_neg = target[neg]

	mixed_images, lam = mixup_data(pos_images, neg_images, alpha=2.0)

	embedding_anchor_clean = network(anchor_images)
	embedding_mixed = network(mixed_images)

	similarity_mixed = sim_mat(embedding_anchor_clean, embedding_mixed)

	return similarity_mixed, target_pos, target_neg, torch.from_numpy(np.array(a2)), torch.from_numpy(np.array(pos)), torch.from_numpy(np.array(neg)), lam


def input_neg_pair_mixup_without_posanchor(a1, a2, neg, clean_embedding, images, target, sim_mat, network, top_k):
	clean_sim = sim_mat(clean_embedding)
	for anchor_pos in np.unique(a1.cpu().numpy()):
		idxs = torch.where(anchor_pos == a2)[0]
		a2 = torch.cat([a2[0:idxs[0]], a2[idxs[-1]+1:]])
		neg = torch.cat([neg[0:idxs[0]], neg[idxs[-1]+1:]])
	
	try:
		del anchor_pos
	except UnboundLocalError:
		pass
	
	anch_neg, hard_neg = [],[]
	for anchor in np.unique(a2.cpu().numpy()):
		idxs = torch.where(anchor == a2)[0]
		dist = clean_sim[a2[idxs], neg[idxs]]
		_,indices = torch.sort(dist,descending=False)

		hard_neg.append(neg[indices][:top_k].cpu().numpy()) 
		anch_neg.append(a2[idxs][:top_k].cpu().numpy())
	
	a2_new = list(itertools.chain.from_iterable(anch_neg))
	neg_new = list(itertools.chain.from_iterable(hard_neg))

	anchor_images = images[a2_new]
	neg_images = images[neg_new]

	target_pos = target[a2_new]
	target_neg = target[neg_new]

	mixed_images, lam = mixup_data(anchor_images, neg_images, alpha=2.0)

	embedding_anchor_clean = network(anchor_images)
	embedding_mixed = network(mixed_images)

	similarity_mixed = sim_mat(embedding_anchor_clean, embedding_mixed)

	return similarity_mixed, target_pos, target_neg, torch.from_numpy(np.array(a2_new)), torch.from_numpy(np.array(neg_new)), lam


def embed_posneg_pair_mixup_for_pos_anchor(a1, pos, a2, neg, embedding, target, sim_mat):
	new_anc, new_neg, new_pos = [],[],[]
	for anchor_pos in a1:

		idxs = torch.where(anchor_pos == a2)[0]
		new_neg.append(neg[idxs].cpu().numpy()) 
		new_anc.append(a2[idxs].cpu().numpy())

		index_pos = torch.where(anchor_pos == a1)[0]

		new_pos.append([pos[index_pos].cpu().numpy()[0]]*len(idxs))
		
	a2 = list(itertools.chain.from_iterable(new_anc))
	neg = list(itertools.chain.from_iterable(new_neg))
	pos = list(itertools.chain.from_iterable(new_pos))

	anchor_embedding = embedding[a2]
	pos_embedding = embedding[pos]
	neg_embedding = embedding[neg]

	target_pos = target[pos]
	target_neg = target[neg]

	mixed_embedding, lam = mixup_data(pos_embedding, neg_embedding, alpha=2.0)

	similarity_mixed = sim_mat(anchor_embedding, mixed_embedding)

	return similarity_mixed, target_pos, target_neg, torch.from_numpy(np.array(a2)), torch.from_numpy(np.array(pos)), torch.from_numpy(np.array(neg)), lam


def embed_neg_pair_mixup_without_posanchor(a1, a2, neg, embedding, target, sim_mat):

	for anchor_pos in np.unique(a1.cpu().numpy()):
		idxs = torch.where(anchor_pos == a2)[0]
		a2 = torch.cat([a2[0:idxs[0]], a2[idxs[-1]+1:]])
		neg = torch.cat([neg[0:idxs[0]], neg[idxs[-1]+1:]])
		
	try:
		del anchor_pos
	except UnboundLocalError:
		pass

	new_anc, new_neg = [],[]
	for anchor in np.unique(a2.cpu().numpy()):
		idxs = torch.where(anchor == a2)[0]

		new_neg.append(neg[idxs].cpu().numpy()) 
		new_anc.append(a2[idxs].cpu().numpy())
	
	a2 = list(itertools.chain.from_iterable(new_anc))
	neg = list(itertools.chain.from_iterable(new_neg))

	anchor_embedding = embedding[a2]
	neg_embedding = embedding[neg]
	
	target_pos = target[a2]
	target_neg = target[neg]

	mixed_embedding, lam = mixup_data(anchor_embedding, neg_embedding, alpha=2.0)

	similarity_mixed = sim_mat(anchor_embedding, mixed_embedding)

	return similarity_mixed, target_pos, target_neg, torch.from_numpy(np.array(a2)), torch.from_numpy(np.array(neg)),lam


def feature_posneg_pair_mixup_for_pos_anchor(a1, pos, a2, neg):
	
	new_anc, new_neg, new_pos = [],[],[]
	for anchor_pos in a1:
		idxs = torch.where(anchor_pos == a2)[0]
		new_neg.append(neg[idxs].cpu().numpy()) 
		new_anc.append(a2[idxs].cpu().numpy())

		index_pos = torch.where(anchor_pos == a1)[0]
		new_pos.append([pos[index_pos].cpu().numpy()[0]]*len(idxs))
		
	a2 = list(itertools.chain.from_iterable(new_anc))
	neg = list(itertools.chain.from_iterable(new_neg))
	pos = list(itertools.chain.from_iterable(new_pos))

	return torch.from_numpy(np.array(a2)), torch.from_numpy(np.array(pos)), torch.from_numpy(np.array(neg))


def feature_neg_pair_mixup_without_posanchor(a1, a2, neg):	
	for anchor_pos in np.unique(a1.cpu().numpy()):
		idxs = torch.where(anchor_pos == a2)[0]
		a2 = torch.cat([a2[0:idxs[0]], a2[idxs[-1]+1:]])
		neg = torch.cat([neg[0:idxs[0]], neg[idxs[-1]+1:]])
	
	try:
		del anchor_pos
	except UnboundLocalError:
		pass

	new_anc, new_neg = [],[]
	for anchor in np.unique(a2.cpu().numpy()):
		idxs = torch.where(anchor == a2)[0]

		new_neg.append(neg[idxs].cpu().numpy()) 
		new_anc.append(a2[idxs].cpu().numpy())
	
	a2 = list(itertools.chain.from_iterable(new_anc))
	neg = list(itertools.chain.from_iterable(new_neg))

	return torch.from_numpy(np.array(a2)), torch.from_numpy(np.array(neg))