
import argparse
import os
import random
import signal 
import subprocess 
import sys
import time 
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from datasets import load_dataset
from utils.dataloader_utils import MyCollate
from utils.utils import off_diagonal
from ectc import ContextEnhance

from transformers import BertModel, AutoTokenizer

import wandb 

#setting manual seed
MANUAL_SEED = 4444

random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='ectc')


#region: system config: 
parser.add_argument('--workers', default=5, type=int, metavar='N', 
                    help='number of data loader workers') 
parser.add_argument('--world_size', default=1, type=int, metavar='N', 
                    help='total number of GPUs across all nodes/world size') 
parser.add_argument('--print_freq', default=5, type=int, metavar='PF', 
                    help='write in the stats file and print after PF steps') 
parser.add_argument('--root', default='./', type=Path, metavar='CD', 
                    help='path to the root') 
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path, metavar='CD', 
                    help='path to directory in which checkpoint and stats are saved') 

#other args: 
parser.add_argument('--load',default=False, type=bool, 
        help='Load pretrained model')
parser.add_argument('--start_epoch',default=0, type=int, 
        help='start epoch of the pretraining phase')

#region: training hyperparameters: 
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=64, type=int, metavar='n',
                    help='mini-batch size')
parser.add_argument('--learning_rate', default=1e-5, type=float, metavar='LR',
                    help='base learning rate')
##################################################
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum for sgd')
parser.add_argument('--clip', default=100, type=float, metavar='GC',
                    help='Gradient Clipping')
parser.add_argument('--betas', default=(0.9, 0.98), type=tuple, metavar='B',
                    help='betas for Adam Optimizer')
parser.add_argument('--eps', default=1e-9, type=float, metavar='E',
                    help='eps for Adam optimizer')
parser.add_argument('--lambd', default=0.5, type=float, metavar='L',
                    help='lambd of barlow loss')
parser.add_argument('--optimizer', default='adam', type=str, metavar='OP',
                    help='selecting optimizer')
#endregion

#region: transformer hyperparameters: 

parser.add_argument('--proj_dim', default=512, type=int, metavar='D',
                    help='output dimension of the projection layer')

#region: tokenizer
parser.add_argument('--text_encoder', default='bert-base-multilingual-uncased', type=str, metavar='T',
                    help='text encoder')
parser.add_argument('--mbert_out_size', default=768, type=int, metavar='S',
                    help='dimension of encoder (mbert) output')
#endregion


args = parser.parse_args()

def tensor2img(img): 
    #inp.shape = 3, H, W

    #test_img = img.permute(1,2,0)
    test_img = img.unsqueeze(-1)
    test_img = test_img.cpu().detach().numpy()
    test_img = 255*test_img
    test_img = test_img.astype(np.uint8)
    return test_img
#endregion


#initializing the tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)

dataset = load_dataset("stas/wmt14-en-de-pre-processed")

def main(): 

    # single-node distributed training
    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.ngpus_per_node = args.world_size 
    torch.multiprocessing.spawn(pretrain, args=(args,), nprocs=args.ngpus_per_node)

def pretrain(gpu, args): 
    
    args.rank +=gpu
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, 
                                         world_size=args.world_size, rank=args.rank)
    
    
    if args.rank == 0: 

        wandb.init(config=args, project='ectc')
        wandb.run.name = f"pc-dry-run-{wandb.run.id}"
        wandb.config.update(args)
        config = wandb.config
        wandb.run.save()

        checkpoint_dir = args.root.joinpath(args.checkpoint_dir)
        if not os.path.exists(checkpoint_dir): 
            os.makedirs(checkpoint_dir)
        
        #stats_file = open(checkpoint_dir / 'stats_file.txt')

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    #load dataset
    # dataset =  load_dataset("stas/wmt14-en-de-pre-processed")

    #load dataset
    dataset = load_dataset("bentrevett/multi30k")

    assert args.batch_size%args.world_size == 0 
    per_device_batch_size = args.batch_size // args.world_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset['train'])
    train_loader = DataLoader(dataset['train'], batch_size=per_device_batch_size, 
                              shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler, 
                              collate_fn=MyCollate(tokenizer))

    
    #model to pretrain
    encoder = BertModel.from_pretrained(args.text_encoder).cuda(gpu)

    checkpoint = os.path.join(checkpoint_dir, "enc_ckpt.pth")
    if args.load and os.path.exists(checkpoint): 

        ckpt = torch.load(checkpoint, map_location="cpu") 
        encoder.load_state_dict(ckpt['model'])
        print("Model Loaded")

    encoder = encoder.cuda(gpu)

    proj_layer = nn.Linear(args.mbert_out_size, args.proj_dim).cuda(gpu) 
    pretrainer = ContextEnhance(encoder, proj_layer).cuda(gpu)

    if args.optimizer == 'adam': 
        optimizer = torch.optim.Adam(pretrainer.parameters(), lr=args.learning_rate)
    else: 
        optimizer = torch.optim.SGD( pretrainer.parameters(), lr=args.learning_rate)


    

    for epoch in range(args.start_epoch, args.epochs): 
        
        start_time = time.time()
        encoder.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0 

        counter = 0 
        
        for step, item in enumerate(train_loader, start=epoch*len(train_loader)): 
            
            src = item[0] 
            tgt = item[1]

            src = src.cuda(gpu)
            tgt = tgt.cuda(gpu)

            optimizer.zero_grad()

            corr = pretrainer(src, tgt)            
            
            img = tensor2image(corr[0]) 

            img = wandbImage(img, caption="corr_mat") 
            wandb.log({'image': img})  

            on_diag = torch.diagonal(corr).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(corr).pow_(2).sum()
            loss = on_diag + args.lambd * off_diag

            wandb.log({"iter loss": loss})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), args.clip)
            optimizer.step()
            counter += 1

        epoch_loss += loss.item()
        epoch_loss = epoch_loss/counter
        wandb.log({"epoch loss": loss})

    
if __name__ == '__main__': 
    main()
    wandb.finish()



