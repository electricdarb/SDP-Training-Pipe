#
# train an SSD model on Pascal VOC or Open Images datasets
#
from numpy import DataSource
from createdataset import create_dataset 
import os
import sys
import logging
import argparse
import itertools
import torch
from filelock import FileLock

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd

from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

from vision.utils.misc import store_labels
from tqdm import trange

from segment import main as segment_main

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WOLF = """
##############################################################################
###################                                   ########################
###################   THANK YOU PROF WOLF AND TEAM    ########################
###################                                   ########################
##############################################################################   

                                                                           
                                                     .##%/              *(.     
                                                     (#&(#%#//(*/(*(/%(*,/#     
                                                    */,%,*#&&,(%%/(#&/.(&(*     
              .,/*,*,*/,,,**,                    ./(//(#((/##%#&%(#(/*/#(*.     
          (*/##%####%#//(#((,*(*,***,....   .,,,*(#**/#/%(%#(##&%####(##,,*.    
        ,((##%%%#######(((####&%%%%/,(/,,/#(/,//((,.(#%%#%%######%((###/(#(.*   
       *%##(###%%(/(#%%%%((#%&&&&#%/(*//(%#/*,#(((*(/((%%(&%%#(#%#(((%#&%(#,,.  
      .(#%#((#%%%#%#%%%%%%%((#%%&&(#(*(#&%%/,*((#/,/*. *(###*(/((/((,#((,.(*./  
     ,##%%(#(#%%###%%###%%%((((#%#%#*(%&&##/,,*(%%%(*,,,*.,(/(/(((,##*.*,.**/   
     ##%%(#/(#%&%((#(###%%&%%%%###%/*(%&&%%(/,,*((/#*(**/,../(/(#(,*..(*..*     
   ,*/#%##///(#&&#(#((###%%%##%%#(#(/#%%&%###(*,*#%#(((//*/,*.*%#//.(*. .,      
   ,(*##%#//*/#%&%(####(#%##%%%%##%%##%%&%%((#%(/*#%###/*%//(*%&&&/ *,..        
   ,(//(%#(/**(%&##(#/(#(###%%%####%###%&&######&&((#(//((%(/*/,*.*.*..         
    /(*(#%#(/*/%%#((((/(#(((##%#%((###%##%&#((((#%%%#(#((*//*(((/*,,            
   .*##(#%%(((/#%&((/(((((/((#%####%(##(##%%/(#(#((/##*,**,/(/****,             
   .**((/#%%##/(#%%(//*///*////(((((((//##/(##(((#####%##%#(%%%(*.              
    .//(#(####/***%(//*,,,*,***/////(/**/(*/((((*///(((#%%##(#/..               
      #((*/(#%(/,/(,*#%%#/*(,,*,,*//////**//(((##*,*,,,,/%#/.,..                
     /((#*((#((**(#/(##%%%#((/  ..,,***/**/(((#(((/*/,(%(((/(#/.                
    *,(/#,/((//(#(//(#####((         ..,*/***(((((#, .###(((((*.                
     .#(#(*/#(#%#.(%/###(.              .*/,**//((,   /*(/,***/.                
      //(#/*//%(.&#(##*                  ,,(**,**/*    */,..,/*                 
        ,/(*(#*..((#(*                     ,(//**//*    /*,,,/,                 
        .////.  ./#//.                       (/***,/,    (,,,*,.                
        .(,/,    (#/*                         //*,.,*.   ,/.,,,.                
         (*/,    /(/,.                          /*,,.,.   /,.,,.                
         **/,.   .*/*,                           ,/*,,.    /,..                 
         *,/*.    /*/...                          ***,,,.  /*...                
        ***,,,.   ***#/,.,.                        **,,,.  ./,.*,               
        .*%/##%/.      . .                          (,,,..  /*.,.               
                                                    ,*,*,./. ##,,,#,..,         
                                                    .,(//#%/%. *.,#**(//  
"""

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def get_data_loaders(dataset_path = 'data'):
    dataset_paths = [dataset_path]
    datasets = []

    config = mobilenetv1_ssd_config

    with FileLock(os.path.expanduser("~/data.lock")):

        for dataset_path in dataset_paths:

            train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
            target_transform = MatchPrior(config.priors, config.center_variance,config.size_variance, 0.5)

            test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

            dataset = VOCDataset(dataset_path, transform = train_transform, target_transform = target_transform)

            datasets.append(dataset)

        # create training dataset
        train_dataset = ConcatDataset(datasets)

        train_loader = DataLoader(train_dataset, 1, num_workers = 2, shuffle=True)

        # create training dataset
        train_dataset = ConcatDataset(datasets)

        train_loader = DataLoader(train_dataset, 1,
                                num_workers = 2, shuffle = True)
        
        val_dataset = VOCDataset(dataset_path, transform=test_transform, target_transform=target_transform, is_test=True)

        val_loader = DataLoader(val_dataset, 1, num_workers = 2, shuffle=False)

    return train_loader, val_loader


def train_net(epochs, 
        lr, 
        momentum, 
        weight_decay, 
        t_max, 
        base_lr, 
        num_classes, 
        dataset_path = 'data'):
     # select the network architecture and config     
    
    create_net = create_mobilenetv1_ssd
    config_ = mobilenetv1_ssd_config
        
    # create data transforms for train/val
    train_loader, val_loader = get_data_loaders(dataset_path)

    # create the network
    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1

    # freeze certain layers (if requested)
    base_net_lr = base_lr ####
    extra_layers_lr = lr ####
    
    params = [
        {'params': net.base_net.parameters(), 'lr': base_net_lr},
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': extra_layers_lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]

    net.init_from_pretrained_ssd("models/mobilenet-v1-ssd-mp-0_675.pth")

    # move the model to GPU
    net.to(DEVICE)

    # define loss function and optimizer
    criterion = MultiboxLoss(config_.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)

    optimizer = torch.optim.SGD(params, lr = lr, momentum = momentum, weight_decay = weight_decay)

    scheduler = CosineAnnealingLR(optimizer, t_max, last_epoch=last_epoch)

    for epoch in trange(epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer, device=DEVICE, epoch=epoch)

    return net 


def export_onnx(model, savepath = 'AshtonThisIsTheModel.onnx', input_size = (1, 3, 300, 300)):
    # create dummy input that you pass to onnx during compilation
    dummy_input = torch.zeros(input_size)

    print('Exporting Model')

    # export onnx model 
    torch.onnx.export(model, dummy_input, savepath, verbose = True)

    print('Exported ONNX model \n\n')


def main():
    # define vars
    item_folder = 'items'
    root_data_dir = 'data'
    
    # segment photos
    segment_main(
        image_in_dir = '', 
        image_out_dir = item_folder
    )

    # create dataset here
    info = create_dataset(
        root_dir = root_data_dir,
        item_folder = item_folder,
        backgrounds = 'backgrounds',
        dataset_size = 2000
    )

    # create net
    net = train_net(
        epochs = 4, 
        lr = .0183, 
        momentum = .49, 
        weight_decay = 0.000101129, 
        t_max = 116, 
        base_lr = 4.1e-5, 
        num_classes = info['num_classes'] + 1,
        dataset_path = root_data_dir # info['root_dir']
    )

    # export onnx net for ASHTON
    export_onnx(net)

    # print the easter egg
    print(WOLF)


if __name__ == "__main__":
    main()