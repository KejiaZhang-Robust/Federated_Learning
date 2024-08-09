import os
import copy
import time
import pickle
import yaml
import logging
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F


# from update import LocalUpdate, test_inference
from model import *
from utils_dataset import create_dataloader
from update import LocalUpdate, average_weights, test_inference
from utils import save_checkpoint

with open('configs_train.yml') as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

device = 'cuda' if torch.cuda.is_available() and config.Operation.GPU else 'cpu'

if __name__ == '__main__':
    file_name = config.Operation.Prefix
    data_set = config.DATA.Dataset
    check_path = os.path.join('./checkpoint', data_set, file_name)
    
    
    if not os.path.isdir(os.path.join('./checkpoint', data_set)):
        os.mkdir(os.path.join('./checkpoint', data_set))
    if not os.path.isdir(check_path):
        os.mkdir(check_path)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(check_path, file_name + '_record.log')),
            logging.StreamHandler()
        ])
    
    logger.info(f'| Model Archetecture: {config.Operation.Prefix}, Dataset: {config.DATA.Dataset}, Distribution: {config.DATA.Split}|')
    logger.info(config.Operation.record_words)
    
    # load dataset and user groups
    train_dataset, test_dataset, user_groups = create_dataloader(config)
    
    # build model
    global_model = CNNMnist()
    
    global_model.to(device)
    global_model.train()
    
    # copy weights
    global_weights = global_model.state_dict()
    
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    best_acc = 0
    
    for epoch in range(config.Train.Epoch):
        local_weights, local_losses = [], []
        global_model.train()
        m = max(int(config.Train.C * config.Train.Num_users), 1)
        
        idxs_users = np.random.choice(range(config.Train.Num_users), m, replace=False)
        
        for idx in idxs_users:
            local_model = LocalUpdate(config=config, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(config.Train.Num_users):
            local_model = LocalUpdate(config=config, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        
        #TODO: save the model if the accuracy is the best
        is_best = best_acc < 100 * train_accuracy[-1]
        save_checkpoint({
            'epoch': epoch,
            'state_dict': global_model.state_dict(),
            'best_prec1': best_acc,
        }, is_best, os.path.join(check_path))        
        if is_best:
            best_acc = 100 * train_accuracy[-1]
        
        logger.info(
        f'''| Global Training Epoch : {epoch+1} | Loss: {np.mean(np.array(train_loss)):.2f}, Accuracy: {100 * train_accuracy[-1]:.2f}%''')

    # Test inference after completion of training
    test_acc, test_loss = test_inference(config, global_model, test_dataset)
    
    logger.info(f'Results after {config.Train.Epoch} global rounds of training:')
    logger.info("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    logger.info("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    logger.info("|---- Test Loss: {:.2f}".format(test_loss))
    
    logger.info(f'Results saved at: {check_path}')
    