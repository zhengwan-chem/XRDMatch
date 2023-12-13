import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset
from semilearn.datasets.augmentation import xrd_augementation
from semilearn.datasets.augmentation.xrd_augementation import *
import global_xrd
from semilearn import get_data_loader, get_net_builder, get_algorithm, get_config, Trainer, split_ssl_data, BasicDataset
from sklearn.metrics import confusion_matrix,recall_score,f1_score,accuracy_score,precision_score

config = {
    'algorithm': 'flexmatch',
    'net': 'vgg_16',
    'layer_decay': 0.5,
    'batch_size': 32,
    'eval_batch_size': 32,

    # dataset configs
    'dataset':'cifar10',
    'num_labels': 20,
    'num_classes': 2,
    'input_size': 32,
    # algorithm specific configs
    'hard_label': True,
    'uratio': 3,
    'ulb_loss_ratio': 1.0,

    # device configs
    'gpu': 7,
    'world_size': 1,
    'distributed': False,
}
config = get_config(config)

lb_dataset = pd.read_csv('lbs_3.csv')  
img_list = np.array(lb_dataset)
np.random.shuffle(img_list)
lb_data = img_list[:,5:] 
lb_target = img_list[:,4] 
lb_name = img_list[:,0]
lb_id = img_list[:,1]
lb_cond = img_list[:,3]

file_pre = open('en_eval.txt','w')

ulb_dataset = pd.read_csv('ulbs.csv') 
img_list_train = np.array(ulb_dataset)
ulb_data = img_list_train[:,3:]
print(len(ulb_data[0]))
ulb_name = img_list_train[:,0]
target = np.zeros(len(ulb_data)) 
ulb_id = img_list_train[:,2]

lb_pred = pd.read_csv('pred_lb.csv') 
img_list_pred = np.array(lb_pred)
pred_data = img_list_pred[:,3:]
pred_name = img_list_pred[:,0]
pred_target = np.zeros(len(pred_data)) 
pred_id = img_list_pred[:,2]

eval_loader = lb_data
eval_transform = main_eval
for i in range(len(lb_data)):
    eval_loader[i] = eval_transform(lb_data[i])
eval_loader = np.reshape(eval_loader,(len(eval_loader),1,4501))
eval_loader = torch.tensor(eval_loader.astype(float)) 
lb_target = torch.tensor(lb_target.astype(int))

ulb_loader = ulb_data
ulb_transform = main_eval
for i in range(len(ulb_data)):
    ulb_loader[i] = ulb_transform(ulb_data[i])
ulb_loader = np.reshape(ulb_loader,(len(ulb_loader),1,4501))
ulb_loader = torch.tensor(ulb_loader.astype(float)) 
ulb_target = torch.tensor(target.astype(int)) 
ulb_condidate = np.zeros(len(ulb_target))    

pred_loader = pred_data
pred_transform = main_eval
for i in range(len(pred_data)):
    pred_loader[i] = pred_transform(pred_data[i])
pred_loader = np.reshape(pred_loader,(len(pred_loader),1,4501))
pred_loader = torch.tensor(pred_loader.astype(float)) 
pred_target = torch.tensor(pred_target.astype(int)) 
pred_condidate = np.zeros(len(pred_target))    

f_ulb = open('pred_ulb.txt','w')
f_pred = open('compare_ml.txt','w')
f_1 = open('c1.txt','w')

en_preds = np.zeros(len(lb_target))
en_pred_log = np.zeros(len(lb_target))
#num = 100
n_1 = 0
num = [85,   52,   71,    4,   98,   22,   42,   46,    8,   75]
for k in num:
#for k in range(num):

    load_path = './saved_models/flexmatch/'+str(k)+'/model_best.pth' 
    config = {
        'algorithm': 'fixmatch',
        'net': 'vgg_16',
        'load_path': load_path, 

        'layer_decay': 0.5,
        'batch_size': 32,
        'eval_batch_size': 32,

        # dataset configs
        'dataset':'cifar10',
        'num_labels': 20,
        'num_classes': 2,
        'input_size': 32,
        # algorithm specific configs
        'hard_label': True,
        'uratio': 3,
        'ulb_loss_ratio': 1.0,

        # device configs
        'gpu': 7,
        'world_size': 1,
        'distributed': False,
    }
    config = get_config(config)

    checkpoint_path = os.path.join(config.load_path)
    print(config.load_path)
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path,map_location='cuda:7')
    load_model = checkpoint['model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    net = get_net_builder(config.net, config.net_from_name)(num_classes=config.num_classes)
    keys = net.load_state_dict(load_state_dict)
    if torch.cuda.is_available():
        net.cuda(config.gpu)
    net.eval()
    n_2 = 0
    acc = 0.0
    test_feats = []
    test_preds = []
    test_probs = []
    test_labels = []
    ulb_feats = []
    ulb_preds = []
    ulb_probs = []

    with torch.no_grad():

        image = eval_loader
        target = lb_target 
        image = image.type(torch.FloatTensor).cuda(config.gpu)
        feat = net(image, only_feat=True)
        logit = net(feat, only_fc=True)
        prob = logit.softmax(dim=-1)
        pred = prob.argmax(1) 
        acc += pred.cpu().eq(target).numpy().sum()
        

        test_feats.append(feat.cpu().numpy())
        test_preds.append(pred.cpu().numpy())
        test_probs.append(prob.cpu().numpy())
        test_labels.append(target.cpu().numpy())

        for i in range(int(len(ulb_loader)/128)):
            start = (i-1)*128
            end = i*128
            ulb_image = ulb_loader[start:end]         
        
            ulb_image = ulb_image.type(torch.FloatTensor).cuda(config.gpu)
            ulb_feat = net(ulb_image, only_feat=True)
            ulb_logit = net(ulb_feat, only_fc=True)
            ulb_prob = ulb_logit.softmax(dim=-1)
            ulb_pred = ulb_prob.argmax(1)
            ulb_feats.append(ulb_feat.cpu().numpy())
            ulb_preds.append(ulb_pred.cpu().numpy())
            ulb_probs.append(ulb_prob.cpu().numpy())

        pred_image = pred_loader
        pred_target = pred_target         
        pred_image = pred_image.type(torch.FloatTensor).cuda(config.gpu)
        pred_feat = net(pred_image, only_feat=True)
        pred_logit = net(pred_feat, only_fc=True)
        pred_prob = pred_logit.softmax(dim=-1)
        pred_pred = pred_prob.argmax(1)

    test_feats = np.concatenate(test_feats)
    test_preds = np.concatenate(test_preds)
    test_probs = np.concatenate(test_probs)
    test_labels = np.concatenate(test_labels)
    ulb_feats = np.concatenate(ulb_feats)
    ulb_preds = np.concatenate(ulb_preds)
    ulb_probs = np.concatenate(ulb_probs)
    
    
    cf_mat = confusion_matrix(test_labels, test_preds, normalize='true')    
    recall = recall_score(test_labels, test_preds, average='macro')
    f1 = f1_score(test_labels, test_preds, average='macro')
    precision = precision_score(test_labels, test_preds, average='macro')
#    for i in range(len(test_labels)):
#        en_preds[i] = test_preds[i] + en_preds[i]
    if(f1>0.1):
        n_1 = n_1 +1
        print(k,precision,f1,recall,file=f_pred)
        for i in range(len(test_labels)):
            en_pred_log[i] = en_pred_log[i] + test_probs[i,1]
            en_preds[i] = test_preds[i] + en_preds[i]   
        for i in np.nonzero(ulb_preds==0)[0]:
             ulb_condidate[i] += 1
             n_2 += 1
        for i in np.nonzero(np.array(pred_pred.cpu())==0)[0]:
             pred_condidate[i] += 1
    print(cf_mat,file=file_pre)
    print(f"Test Accuracy: {acc/len(eval_loader)}",file=file_pre) 
    print(f1,n_1,cf_mat,file=f_1)   
   
en_preds = en_preds/float(n_1)
en_preds = np.around(en_preds)    
en_pred_log = en_pred_log/float(n_1)
en_pred_log = np.around(en_pred_log)

for i in range(len(pred_target)):    
    print(int(pred_condidate[i]/n_1*2),pred_name[i],pred_condidate[i],n_1,file=f_pred)

for i in range(len(ulb_target)):    
    if(ulb_condidate[i]>4):
        print(ulb_condidate[i],ulb_name[i],ulb_id[i],file=f_ulb)

cf_mat = confusion_matrix(test_labels, en_preds, normalize='true')     
f1 = f1_score(test_labels, en_preds, average='macro')
top1 = precision_score(test_labels,en_preds, average='macro')


print(top1,f1,cf_mat,file=file_pre) 


cf_mat = confusion_matrix(test_labels, en_pred_log, normalize='true')
f1 = f1_score(test_labels, en_pred_log, average='macro')
top1 = precision_score(test_labels, en_pred_log, average='macro')
print(top1,f1,cf_mat,file=file_pre)

