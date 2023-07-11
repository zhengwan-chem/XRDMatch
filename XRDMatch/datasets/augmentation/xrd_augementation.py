import pandas as pd
import numpy as np  
import torch
import global_xrd
def normdata(data):  

    min_x = min(data)
    max_x = max(data)
    norm = max_x - min_x
    data = (data - min_x)/norm
   
    return data

def data_zero(data):  
    num = len(data)
    for i in range(num):
        if(data[i]<0.1):
            data[i]=0

    return data

def weak_augdata(data):
    w_noise_ratio,w_noise_peak,w_move_gap = global_xrd.get_value()
    ratio = np.random.random()
    if ratio <= 0.5:
        index = np.nonzero(data==0)[0]
        idx_num = len(index)
        noise_num = int(idx_num*w_noise_ratio*np.random.random())
        np.random.shuffle(index)       
        for i in index[:noise_num]: 
            data[i] = np.random.random()*w_noise_peak

    ratio = np.random.random()
    if ratio <= 0.5:
        cut = np.random.randint(50,w_move_gap,1)[0]
        if ratio <= 0.5:
            out = 4501 - cut
            data = np.append(np.zeros(cut),data[:out])
        else:
            data = np.append(data[cut:],np.zeros(cut))
            
    return data

def strong_augdata(data):
    s_noise_ratio = 0.0
    s_noise_peak = 0.15
    s_move_gap = 300
    s_scaling_ratio = 0.12
    s_elimin_ratio  = 0.12
    ratio = np.random.random()
    if ratio <=0.5:
        index = np.nonzero(data)[0]
        idx_num = len(index)
        scaling_num = int(idx_num*s_scaling_ratio*np.random.random() )  
        np.random.shuffle(index)       
        for i in index[:scaling_num]: 
            data[i] = np.random.random()*2*data[i]+data[i]

    ratio = np.random.random()
    if ratio <=0.5:
        index = np.nonzero(data)[0]
        idx_num = len(index)
        elimin_num = int(idx_num*s_elimin_ratio*np.random.random() )  
        np.random.shuffle(index)       
        for i in index[:elimin_num]: 
            data[i] = 0

    ratio = np.random.random()
    if ratio <= 0.5:
        ndata =data_zero(data)
        index = np.nonzero(ndata)[0]
        idx_num = len(index)
        old_idx = 0
        gap_left = []
        gap_right = []
        cut = np.random.randint(1,s_move_gap,1)[0]
               
        for i in range(idx_num):
            value = index[i] - old_idx
            if value > cut:
                gap_left.append(old_idx)
                gap_right.append(index[i])                   
            old_idx = index[i]
        
        ratio = np.random.random()
        if ratio <= 0.5:
            if (len(gap_right)!=0):
                np.random.shuffle(gap_right)
                sele_site = gap_right[0]
                out = sele_site - cut
                data = np.concatenate((data[:out],data[sele_site:],np.zeros([cut])),axis=0)
        else:
            if (len(gap_left)!=0):            
                np.random.shuffle(gap_left)
                sele_site = gap_left[0]+1 
                out = sele_site + cut       
                data = np.concatenate((np.zeros([cut]),data[:sele_site],data[out:]),axis=0)
    ratio = np.random.random()
    if ratio <= 0.5:
        index = np.nonzero(data==0)[0]
        idx_num = len(index)
        noise_num = int(idx_num*s_noise_ratio*np.random.random() )  
        np.random.shuffle(index)        
        for i in index[:noise_num]: 
            data[i] = np.random.random()*s_noise_peak
           
    return data

def main_strong(dataset):
    dataset = normdata(dataset)    
    dataset = data_zero(dataset)
    data = strong_augdata(dataset)
    dataset = normdata(data)    
    dataset = np.reshape(dataset,(1,len(dataset)))    
    dataset = dataset.astype(np.float32)    
    dataset=torch.Tensor(dataset)
    return dataset

def main_weak(dataset):

    dataset = normdata(dataset)
    dataset = data_zero(dataset)  
    data = weak_augdata(dataset)
    dataset = normdata(data) 
    dataset = np.reshape(dataset,(1,len(dataset)))        
    dataset = dataset.astype(np.float32)       
    dataset=torch.Tensor(dataset)
    return dataset

def main_eval(data):
    dataset = normdata(data)
    dataset = data_zero(dataset)    
    dataset = np.reshape(dataset,(1,len(dataset)))        
    dataset = dataset.astype(np.float32)       
    dataset=torch.Tensor(dataset)
    return dataset
