import numpy as np
import torch
import torch.optim as optim
import sys
from tqdm import trange, tqdm
import os
from logger import Logger
from test import valid
# from loss_ap import MatchLoss
from loss import MatchLoss
from utils import tocuda
from tensorboardX import SummaryWriter
# from loss import batch_episym

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


def train_step(step, optimizer, model, match_loss, data):
   
    res_logits, res_e_hat = model(data)
    loss = 0
    loss_val = []
 

    for i in range(len(res_logits)):
        loss_i, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, res_logits[i], res_e_hat[i])
        loss += loss_i
        loss_val += [geo_loss, cla_loss, l2_loss]
    
           
    optimizer.zero_grad()
    loss.backward()
    
   

    for name, param in model.named_parameters():
        if torch.any(torch.isnan(param.grad)):
            print('skip because nan')
            return loss_val
   
            

    optimizer.step()
    return loss_val

    


def train(model, train_loader, valid_loader, config):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)
    match_loss = MatchLoss(config)

    load_log_path = 'log_UMatch/train/'
    checkpoint_path = os.path.join(load_log_path, 'checkpoint.pth')
    
    config.resume = os.path.isfile(checkpoint_path)
    
    if config.resume:
        if config.local_rank==0:
            print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path,map_location='cuda:{}'.format(config.local_rank))
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    else:
        best_acc = -1
        start_epoch = 0
        
    train_loader_iter = iter(train_loader)
    if config.local_rank==0:
        writer=SummaryWriter(os.path.join(config.log_base,'log_file'))
    
    train_loader.sampler.set_epoch(start_epoch*config.train_batch_size//len(train_loader.dataset))
        
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            if config.local_rank==0:
                print('epoch: ',step*config.train_batch_size//len(train_loader.dataset))
            train_loader.sampler.set_epoch(step*config.train_batch_size//len(train_loader.dataset))    
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
        train_data = tocuda(train_data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']
        loss_vals = train_step(step, optimizer, model, match_loss, train_data)
        if config.local_rank==0:
            writer.add_scalar('geo_loss',loss_vals[-3],step)
            writer.add_scalar('cla_loss',loss_vals[-2],step)
            writer.add_scalar('l2_loss',loss_vals[-1],step)
           

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            
            va_res, _, _, _, _, _, _  = valid(valid_loader, model, step, config)
          
            model.train()
            if config.local_rank==0:
                writer.add_scalar('AUC5',va_res[0],step)
                writer.add_scalar('AUC10',va_res[1],step)
                writer.add_scalar('AUC20',va_res[2],step)
       
                if va_res[0] > best_acc:
                    print('AUC@5\t AUC@10\t AUC@20\t')
                    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(va_res[0]*100, va_res[1]*100, va_res[2]*100))
                    best_acc = va_res[0]
                    torch.save({
                    'epoch': step + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    }, os.path.join(config.log_path, 'model_best.pth'))
                

        if b_save:
            if config.local_rank==0:
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, checkpoint_path)
                
    if config.local_rank==0:
        writer.close()

