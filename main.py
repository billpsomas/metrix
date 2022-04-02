import torch, argparse, os
import numpy as np

import dataset

from general_utils import *
from net.resnet import *
from dataset import sampler
from train import *

from torch import nn
from torch.utils.data.sampler import BatchSampler

from pytorch_metric_learning.losses.contrastive_loss import ContrastiveLoss
from pytorch_metric_learning.distances.lp_distance import LpDistance
from pytorch_metric_learning.reducers.avg_non_zero_reducer import AvgNonZeroReducer
from pytorch_metric_learning.reducers.multiple_reducers import MultipleReducers

import tqdm
from tqdm import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', type=int, default =1, help='Choose seed')
    parser.add_argument('--dataset', type=str, default='cub', choices=['cub', 'cars', 'sop', 'inshop'], help='Dataset to use for training')
    parser.add_argument('--data_root', type=str, default='/home/bill/datasets/', help='Root directory of your datasets')
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture to train')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use for training')
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
    parser.add_argument('--bn_freeze', type=int, default=1, help='Whether to freeze batch normalization or not')
    parser.add_argument('--l2_norm', type=int, default=1, help='Whether to use L2 Norm or not')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of epochs to train for')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--warm', type=int, default=5, help='Number of epochs to warm-up for')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='Learning rate decay step')
    parser.add_argument('--lr_decay_gamma', type=int, default=0.1, help='Learning rate decay gamma')
    parser.add_argument('--save_root', type=str, default='/home/bill/code/outputs/', help='Root directory of your datasets')
    parser.add_argument('--lr', type=int, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=int, default=1e-4, help='Weight decay')
    parser.add_argument('--loss', type=str, default='contrastive', choices=['contrastive', 'multisimilarity', 'proxyanchor'], help='Loss function')
    parser.add_argument('--images_per_class', type=int, default=5, help='Images per class for balanced sampling')
    parser.add_argument('--save_model', default=False, type=bool_flag, help="Whether to save model weights to file or not")
    parser.add_argument('--mode', type=str, default='feature', choices=['baseline', 'input', 'feature', 'embed'] , help="Choose between baseline or metrix")
    parser.add_argument('--alpha', type=float, default=2.0, help="Beta distribution alpha")
    args = parser.parse_args()

    # Set device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)
    
    # Set datasets and dataloaders
    train_dataset = dataset.load(
                name = args.dataset,
                root = args.data_root,
                mode = 'train',
                transform = dataset.utils.make_transform(
                    is_train = True, 
                    is_inception = (args.model == 'bn_inception')
                ))

    if args.dataset == 'cub' or args.dataset == 'cars':
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = args.num_workers,
            drop_last = True,
            pin_memory = True
        )
        print('Random Sampling')
    
    elif args.dataset == 'sop' or args.dataset == 'inshop':
        balanced_sampler = sampler.BalancedSampler(train_dataset, batch_size=args.batch_size, images_per_class = args.images_per_class)
        batch_sampler = BatchSampler(balanced_sampler, batch_size = args.batch_size, drop_last = True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers = args.num_workers,
            pin_memory = True,
            batch_sampler = batch_sampler
        )
        print('Balanced Sampling')
    else:
        print('Please specify correctly the dataset. Choices: cub, cars, sop, inshop')

    if args.dataset == 'cub' or args.dataset == 'cars' or args.dataset == 'sop':
        test_dataset = dataset.load(
                    name = args.dataset,
                    root = args.data_root,
                    mode = 'eval',
                    transform = dataset.utils.make_transform(
                        is_train = False, 
                        is_inception = (args.model == 'bn_inception')
                    ))

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = args.num_workers,
            pin_memory = True
        )
    
    elif args.dataset == 'inshop':
        # For In-Shop, set query and gallery datasets and dataloaders
        query_dataset = dataset.load(
                name = args.dataset,
                root = args.data_root,
                mode = 'query',
                transform = dataset.utils.make_transform(
                    is_train = False, 
                    is_inception = (args.model == 'bn_inception')
        ))

        query_loader = torch.utils.data.DataLoader(
            query_dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = args.num_workers,
            pin_memory = True
        )

        gallery_dataset = dataset.load(
                name = args.dataset,
                root = args.data_root,
                mode = 'gallery',
                transform = dataset.utils.make_transform(
                    is_train = False, 
                    is_inception = (args.model == 'bn_inception')
        ))

        gallery_loader = torch.utils.data.DataLoader(
            gallery_dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = args.num_workers,
            pin_memory = True
        )

    #num_classes = train_dataset.nb_classes()
    # Set model and send it to device
    if args.mode == 'feature':
        from net.feature_resnet import Resnet50
    model = Resnet50(embedding_size=args.embedding_size, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
    model.to(device)

    # Set loss function
    if args.loss == 'contrastive':
        criterion = ContrastiveLoss(neg_margin=0.5)
        distance = LpDistance()

        if args.mode == 'baseline':
            reducer_dict = {"pos_loss" : AvgNonZeroReducer(), "neg_loss" : AvgNonZeroReducer()}
            reducer = MultipleReducers(reducer_dict)
    
        elif args.mode == 'input' or args.mode == 'embed' or args.mode == 'feature':
            reducer_dict_pos = {"pos_loss" : AvgNonZeroReducer()}
            reducer_dict_neg = {"neg_loss" : AvgNonZeroReducer()}
            reducer_pos = MultipleReducers(reducer_dict_pos)
            reducer_neg = MultipleReducers(reducer_dict_neg)

    # Set parameter groups for optimizer
    param_groups = [
        {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu_id != -1 else 
                    list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
        {'params': model.model.embedding.parameters() if args.gpu_id != -1 else model.module.model.embedding.parameters(), 'lr':float(args.lr) * 1},
    ]

    # Set optimizer and scheduler
    opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma = args.lr_decay_gamma)

    # Initialize variables and lists for training
    losses_list = []
    best_recall = [0]
    best_epoch = 0

    # Training
    for epoch in range(0, args.num_epochs):
        model.train()

        if args.bn_freeze:
            modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
            for m in modules: 
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        losses_per_epoch = []
        
        # Warmup: Train only new params, helps stabilize learning
        if args.warm > 0:
            if args.gpu_id != -1:
                unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
            else:
                unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion.parameters())

            if epoch == 0:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True

        pbar = tqdm(enumerate(train_loader))

        for batch_idx, (inputs, target) in pbar: 

            inputs = inputs.cuda()
            target = target.cuda()  

            if args.loss == 'contrastive':
                if args.mode == 'baseline':
                    loss, losses_per_epoch = baseline_contrastive(inputs, target, model, distance, reducer, opt, losses_per_epoch)
                elif args.mode == 'input':
                    loss, losses_per_epoch = input_metrix_contrastive(inputs, target, model, distance, criterion, opt, losses_per_epoch, reducer_pos, reducer_neg)
                elif args.mode == 'embed':
                    loss, losses_per_epoch = embed_metrix_contrastive(inputs, target, model, distance, criterion, opt, losses_per_epoch, reducer_pos, reducer_neg)
                elif args.mode == 'feature':
                    loss, losses_per_epoch = feature_metrix_contrastive(inputs, target, model, distance, criterion, opt, losses_per_epoch, reducer_pos, reducer_neg, args.alpha)

            pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item()))

        mean_loss = np.mean(losses_per_epoch)
        print('Epoch: {} Loss = {:.4f} '.format(epoch, mean_loss))
        scheduler.step()
        torch.cuda.empty_cache()
        
        if(epoch >= 0):
            with torch.no_grad():
                print("Evaluating...")
                if args.dataset == 'inshop':
                    Recalls = evaluate_cos_Inshop(model, query_loader, gallery_loader, args.mode)
                elif args.dataset != 'sop':
                    Recalls = evaluate_cos(model, test_loader, args.mode)
                else:
                    Recalls = evaluate_cos_SOP(model, test_loader, args.mode)
                    
            # Best model save
            if best_recall[0] < Recalls[0]:
                
                print('Saving..')
                best_recall = Recalls
                best_epoch = epoch

                save_dir = os.path.join(args.save_root , args.dataset)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                state = {
                    'model': model,
                    'Recall': Recalls,
                    'epoch': epoch,
                    'rng_state': torch.get_rng_state()
                    }

                if args.save_model:
                    torch.save(state, os.path.join(save_dir, '{}_{}.t7'.format(args.loss, args.model)))
    
    print(best_recall)