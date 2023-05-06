import torch
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

import random
from general_utils import *

from mixup.utils import *
from mixup.losses import *

def baseline_contrastive(inputs, target, model, distance, reducer, opt, losses_per_epoch):
    
    out = model(inputs)
    embedding_similarity = distance(out)
    
    a1, p, a2, n = lmu.get_all_pairs_indices(target)

    pos_pair_dist, neg_pair_dist = [], []
    if len(a1) > 0:
        pos_pair_dist = embedding_similarity[a1, p]
    if len(a2) > 0:
        neg_pair_dist = embedding_similarity[a2, n]

    indices_tuple = (a1, p, a2, n)
    pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
    neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)

    if len(pos_pair_dist) > 0:
        pos_loss = torch.nn.functional.relu(margin(pos_pair_dist, 0.0))
    if len(neg_pair_dist) > 0:
        neg_loss = torch.nn.functional.relu(margin(0.5, neg_pair_dist))

    loss_dict =  {
        "pos_loss": {
            "losses": pos_loss,
            "indices": pos_pairs,
            "reduction_type": "pos_pair",
        },
        "neg_loss": {
            "losses": neg_loss,
            "indices": neg_pairs,
            "reduction_type": "neg_pair",
        },
    }

    loss = reducer(loss_dict, embedding_similarity, target)

    opt.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_value_(model.parameters(), 10)

    losses_per_epoch.append(loss.data.cpu().numpy())
    opt.step()
    
    return loss, losses_per_epoch 

def input_metrix_contrastive(inputs, target, model, distance, criterion, opt, losses_per_epoch, reducer_pos, reducer_neg):
    
    opt.zero_grad()

    out_clean = model(inputs)
    loss_clean = criterion(out_clean, target)
    layer_mix = random.randint(0,1)

    a1, pos, a2, neg = lmu.get_all_pairs_indices(target)
    
    if (layer_mix == 0):
        sim_mix, y_pos, y_neg, new_a2, pos, new_neg, lam_pn = input_posneg_pair_mixup_for_pos_anchor(a1, pos, a2, neg, out_clean, inputs, target, distance, model, top_k=1)
        loss_pos_dict, loss_neg_dict = loss_posneg_mixup(sim_mix, pos, new_a2, new_neg, lam_pn)
        loss_mixed_posneg_for_pos_anc = reducer_pos(loss_pos_dict, sim_mix, y_pos) + reducer_neg(loss_neg_dict, sim_mix, y_neg)
        loss = loss_clean + 0.4*loss_mixed_posneg_for_pos_anc 


    elif (layer_mix == 1):
        sim_mix_ancneg, y_pos_an, y_neg_an, new_a2_an, new_neg_an, lam_an = input_neg_pair_mixup_without_posanchor(a1, a2, neg, out_clean, inputs, target, distance, model, top_k=1)
        loss_dict_pos_ancneg, loss_dict_neg_ancneg = loss_posneg_mixup(sim_mix_ancneg, new_a2_an, new_a2_an, new_neg_an, lam_an)
        loss_mixed_ancneg_without_pos_anc = reducer_pos(loss_dict_pos_ancneg, sim_mix_ancneg, y_pos_an) + reducer_neg(loss_dict_neg_ancneg, sim_mix_ancneg, y_neg_an)
        loss = loss_clean + 0.4*loss_mixed_ancneg_without_pos_anc

    loss.backward()

    torch.nn.utils.clip_grad_value_(model.parameters(), 10)
    
    losses_per_epoch.append(loss.data.cpu().numpy())
    opt.step()

    return loss, losses_per_epoch

def embed_metrix_contrastive(inputs, target, model, distance, criterion, opt, losses_per_epoch, reducer_pos, reducer_neg):
    
    opt.zero_grad()

    out_clean = model(inputs)
    loss_clean = criterion(out_clean, target)
    layer_mix = random.randint(0,1)

    a1, pos, a2, neg = lmu.get_all_pairs_indices(target)
    
    if (layer_mix == 0):
        sim_mix, y_pos, y_neg, new_a2, new_pos, new_neg, lam_pn = embed_posneg_pair_mixup_for_pos_anchor(a1, pos, a2, neg, out_clean, target, distance)
        loss_pos_dict, loss_neg_dict = loss_posneg_mixup(sim_mix, new_pos, new_a2, new_neg, lam_pn)
        loss_mixed_posneg_for_pos_anc = reducer_pos(loss_pos_dict, sim_mix, y_pos) + reducer_neg(loss_neg_dict, sim_mix, y_neg)
        loss = loss_clean + 0.4*loss_mixed_posneg_for_pos_anc 

    elif (layer_mix == 1):
        sim_mix_ancneg, y_pos_an, y_neg_an, new_a2_an, new_neg_an, lam_an = embed_neg_pair_mixup_without_posanchor(a1, a2, neg, out_clean, target, distance)
        loss_dict_pos_ancneg, loss_dict_neg_ancneg = loss_posneg_mixup(sim_mix_ancneg, new_a2_an, new_a2_an, new_neg_an, lam_an)
        loss_mixed_ancneg_without_pos_anc = reducer_pos(loss_dict_pos_ancneg, sim_mix_ancneg, y_pos_an) + reducer_neg(loss_dict_neg_ancneg, sim_mix_ancneg, y_neg_an)
        loss = loss_clean + 0.3*loss_mixed_ancneg_without_pos_anc

    loss.backward()

    torch.nn.utils.clip_grad_value_(model.parameters(), 10)
    
    losses_per_epoch.append(loss.data.cpu().numpy())
    opt.step()

    return loss, losses_per_epoch

def feature_metrix_contrastive(inputs, target, model, distance, criterion, opt, losses_per_epoch, reducer_pos, reducer_neg, alpha):
    opt.zero_grad()

    out_clean = model(inputs, 0.0, 0.0, 0.0, 0.0, mode='clean', type='clean')
    loss_clean = criterion(out_clean, target)

    a1, pos, a2, neg = lmu.get_all_pairs_indices(target)
    layer_mix = random.randint(0,1)
    
    if (layer_mix == 0):
        lam_pn = np.random.beta(alpha, alpha)
        new_a2, new_pos, new_neg = feature_posneg_pair_mixup_for_pos_anchor(a1, pos, a2, neg)

        y_pos = target[new_pos.long()]
        y_neg = target[new_neg.long()]

        anchor_embedding = model(inputs, new_a2, new_pos, new_neg, lam_pn, mode='pos_neg_mixup', type='clean_anchor')
        mixed_embedding = model(inputs, new_a2, new_pos, new_neg, lam_pn, mode='pos_neg_mixup', type='mixed')
    
        sim_mix = distance(anchor_embedding, mixed_embedding)	
        loss_pos_dict, loss_neg_dict = loss_posneg_mixup(sim_mix, new_pos, new_a2, new_neg, lam_pn)
        loss_mixed_posneg_for_pos_anc = reducer_pos(loss_pos_dict, sim_mix, y_pos) + reducer_neg(loss_neg_dict, sim_mix, y_neg)
        loss = loss_clean + 0.4*loss_mixed_posneg_for_pos_anc 

    elif (layer_mix == 1):
        lam_an = np.random.beta(alpha, alpha)
        new_a2_an, new_neg_an = feature_neg_pair_mixup_without_posanchor(a1, a2, neg)
        y_pos_an = target[new_a2_an.long()]
        y_neg_an = target[new_neg_an.long()]

        anchor_embedding_an = model(inputs, new_a2_an, new_a2_an, new_neg_an, lam_an, mode='anc_neg_mixup', type='clean_anchor')
        mixed_embedding_an = model(inputs, new_a2_an, new_a2_an, new_neg_an, lam_an, mode='anc_neg_mixup', type='mixed')
    
        sim_mix_an = distance(anchor_embedding_an, mixed_embedding_an)	
        loss_pos_dict_an, loss_neg_dict_an = loss_posneg_mixup(sim_mix_an, new_a2_an, new_a2_an, new_neg_an, lam_an)
        loss_mixed_ancneg_without_pos_anc = reducer_pos(loss_pos_dict_an, sim_mix_an, y_pos_an) + reducer_neg(loss_neg_dict_an, sim_mix_an, y_neg_an)
        loss = loss_clean + 0.4*loss_mixed_ancneg_without_pos_anc 

    loss.backward()

    torch.nn.utils.clip_grad_value_(model.parameters(), 10)

    losses_per_epoch.append(loss.data.cpu().numpy())
    opt.step()

    return loss, losses_per_epoch

def baseline_multisimilarity(inputs, target, model, distance, miner, alpha, beta, base, opt, losses_per_epoch):
    
    out = model(inputs)
    mat = distance(out)

    a1, p, a2, n = miner(out, target)

    pos_exp = distance.margin(mat, base)
    neg_exp = distance.margin(base, mat)

    pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
    pos_mask[a1, p] = 1
    neg_mask[a2, n] = 1

    pos_loss = (1.0 / alpha) * lmu.logsumexp(
        alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
    )

    neg_loss = (1.0 / beta) * lmu.logsumexp(
        beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
    )

    loss = torch.mean(pos_loss + neg_loss)

    opt.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(model.parameters(), 10)

    losses_per_epoch.append(loss.data.cpu().numpy())
    opt.step()

    return loss, losses_per_epoch

def embed_metrix_multisimilarity(inputs, target, model, distance, miner, alpha, beta, base, opt, losses_per_epoch):
    
    out_clean = model(inputs)
    a1, p, a2, n = miner(out_clean, target)

    sim_mix, _, _, a2_pn, pos_pn, neg_pn, lam_pn = embed_posneg_pair_mixup_for_pos_anchor(a1, p, a2, n, out_clean, target, distance)

    pos_loss = multisimilarity_positive_loss(sim_mix, alpha, base, a1, p, distance, out_clean, a2_pn, pos_pn, lam_pn)

    neg_loss = multisimilarity_negative_loss(sim_mix, beta, base, a2, n, distance, out_clean, a2_pn, neg_pn, lam_pn)

    loss = torch.mean(pos_loss + neg_loss)

    opt.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(model.parameters(), 10)

    losses_per_epoch.append(loss.data.cpu().numpy())
    opt.step()

    return loss, losses_per_epoch

def baseline_proxyanchor(inputs, target, model, criterion, opt, losses_per_epoch):
    out = model(inputs)
    a1, p, a2, n = lmu.get_all_pairs_indices(target)

    loss = criterion(out, target, (a1, p, a2, n))

    opt.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
    torch.nn.utils.clip_grad_value_(model.parameters(), 10)

    losses_per_epoch.append(loss.data.cpu().numpy())
    opt.step()

    return loss, losses_per_epoch