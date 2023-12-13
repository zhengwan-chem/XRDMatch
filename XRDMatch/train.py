import sys
import numpy as np
import pandas as pd
from torchvision import transforms
from semilearn import get_data_loader, get_net_builder, get_algorithm, get_config, Trainer, split_ssl_data, BasicDataset
from semilearn.datasets.augmentation import xrd_augementation
from semilearn.datasets.augmentation.xrd_augementation import *
import global_xrd
from sklearn.metrics import confusion_matrix


ulb_dataset = pd.read_csv('ulbs.csv') 
img_list_train = np.array(ulb_dataset)
unlb_data = img_list_train[:,3:]

lb_dataset = pd.read_csv('lbs.csv')  
img_list = np.array(lb_dataset)
np.random.seed(0)
np.random.shuffle(img_list)
lb_data = img_list[:,5:] 
lb_target = img_list[:,4] 
lb_name = img_list[:,0]
lb_id = img_list[:,1]
a = 0
c = 0

pred_data = []
pred_target = []
pred_name = []
posi_data = []
posi_target = []
nega_data = []
nega_target = []    
posi_name = []
posi_id = []  
nega_name = []
nega_id = []      
     
for i in range(len(lb_target)):
    if(lb_target[i]==0):
        a =a+1
        if(a<20):
            posi_data.append(lb_data[i])
            posi_target.append(lb_target[i])
            posi_name.append(lb_name[i])
            posi_id.append(lb_id[i])            
        else:
            pred_data.append(lb_data[i])
            pred_target.append(lb_target[i])                
            pred_name.append(lb_name[i])
for i in range(len(lb_target)):
    if(lb_target[i]==1):
        c =c+1
        if(c<75):
            nega_data.append(lb_data[i])
            nega_target.append(int(lb_target[i]))
            nega_name.append(lb_name[i])
            nega_id.append(lb_id[i])                  
        else:
            pred_data.append(lb_data[i])
            pred_target.append(lb_target[i])  
            pred_name.append(lb_name[i])
file_name = open('lb_name.txt','w')
print(pred_name,file=file_name)
un_ratio = 0.8
for k in range(100):

    save_name = './flexmatch/'+str(k) 
    config = {

        'algorithm': 'flexmatch',
        'net': 'vgg_16',
        'use_pretrain': False, 
        'pretrain_path': None,
        # optimization configs
        'epoch': 100,  # set to 100
        'num_train_iter':1000,  # set to 102400  
        'num_eval_iter':10,   # set to
        'optim': 'AdamW',
        'lr': 3e-4,
    #    'optim': 'SGD',
    #    'lr': 0.005,
    #    'momentum': 0.99,
        'layer_decay': 1.0,
        'batch_size': 32,
        'eval_batch_size': 32,

        # dataset configs
        'dataset':'cifar10',
        'num_labels': 20,
        'num_classes': 2,
        'input_size': 32,
        'data_dir': './data',
        'save_name': save_name,
        # algorithm specific configs
        'hard_label': True,
        'uratio': 3,
        'ulb_loss_ratio': 1.0,

        # device configs
        'gpu': 4,
        'world_size': 1,
        'distributed': False,
    }
    config = get_config(config)

    # create model and specify algorithm
    algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)
    lb_num = int(config.num_labels/2)
    
    np.random.seed(k)            
    np.random.shuffle(posi_data)
    np.random.shuffle(nega_data)
    np.random.shuffle(posi_target)
    np.random.shuffle(nega_target)    
    np.random.shuffle(posi_name)
    np.random.shuffle(nega_name)
    np.random.shuffle(unlb_data)

    data = unlb_data[:int(len(unlb_data)*un_ratio)]
    target = np.random.random_integers(0,1,int(len(unlb_data)*un_ratio))    
    train_data = np.append(posi_data[:lb_num],nega_data[:lb_num]).reshape(lb_num*2,len(lb_data[0]))

    train_target = np.append(posi_target[:lb_num],nega_target[:lb_num])
    train_target = np.array(train_target)
    train_name = np.append(posi_name[:lb_num],nega_name[:lb_num])
    for i in range(len(train_name)):
    	print(train_name[i],file=file_name)
    	     
    n = len(train_data)+len(data)
    data = np.append(train_data,data).reshape(n,len(lb_data[0]))
    target = np.append(train_target,target)
    
    lb_data, lb_target, ulb_data, ulb_target = split_ssl_data(config, data, target,
                                                            lb_num_labels=config.num_labels,
                                                            num_classes=config.num_classes)
    

    train_transform = main_weak
    train_strong_transform = main_strong

    lb_dataset = BasicDataset(config.algorithm, lb_data, lb_target, config.num_classes, train_transform, is_ulb=False)
    ulb_dataset = BasicDataset(config.algorithm, ulb_data, ulb_target, config.num_classes, train_transform, is_ulb=True, strong_transform=train_strong_transform)
    
    eval_num = len(posi_data) + len(nega_data) - config.num_labels
    eval_data = np.append(posi_data[lb_num:],nega_data[lb_num:]).reshape(eval_num,len(lb_data[0]))
    eval_target = np.append(posi_target[lb_num:],nega_target[lb_num:])
    eval_target = np.array(eval_target)

    eval_transform = main_eval
    eval_dataset = BasicDataset(config.algorithm, eval_data, eval_target, config.num_classes, eval_transform, is_ulb=False)
    

    pred_transform = main_eval
    pred_dataset = BasicDataset(config.algorithm, pred_data, pred_target, config.num_classes, pred_transform, is_ulb=False)

    train_lb_loader = get_data_loader(config, lb_dataset, config.batch_size)
    train_ulb_loader = get_data_loader(config, ulb_dataset, int(config.batch_size * config.uratio))
    eval_loader = get_data_loader(config, eval_dataset, config.eval_batch_size)
    pred_loader = get_data_loader(config, pred_dataset, config.eval_batch_size)

    trainer = Trainer(config, algorithm)
    trainer.fit(train_lb_loader, train_ulb_loader, eval_loader, pred_loader)
    trainer.evaluate(eval_loader)
file_name.close()    
