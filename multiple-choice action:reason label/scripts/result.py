#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:00:12 2019

@author: xiaoyin
"""
#%%
import csv
import json
import numpy as np

class Content:
    def __init__(self):
        self.annoList = []
        self.workerList = []
        self.anno = []
        self.worker = []
 #%%       
def read_csv(filename):
    
    AssignId = []
    WorkerId = []
    VideoId = []
    Answer = []
    
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[15] in ['AAFPXJU8F3FAJ','A2UM5VTQ1Q4TGU']:
                continue
            AssignId.append(row[14])
            WorkerId.append(row[15])
            VideoId.append(row[27])
            Answer.append(row[28])
    
    AssignId.pop(0)
    WorkerId.pop(0)
    VideoId.pop(0)
    Answer.pop(0)
    
    dic = {}
    for i in range(len(VideoId)):
        data = json.loads(Answer[i])
        if not VideoId[i] in dic:
            dic[VideoId[i]] = Content()
        dic[VideoId[i]].annoList.append(data[0]) 
        dic[VideoId[i]].workerList.append(WorkerId[i])
        
        if len(dic[VideoId[i]].annoList)>3:
            print('exceed worker number per video:', VideoId[i])
    
    return dic

dic = read_csv('Batch_3723305_batch_results_8.8.csv')
dic2 = read_csv('Batch_3730076_batch_results_8.12.csv')

#%%dic2
dic2_list = []
for key in dic2:
    action_dic = {}
    action_dic['video'] = key
    action_dic['label'] = dic2[key].annoList[0]
    dic2_list.append(action_dic)
#%%
#AllMatch
def allmatch(dic, writefile=False):
    allmatch_list = []  
    allmatch_key = []     
    count_all = 0
    divList = ['action', 'f_reason', 'l_reason','r_reason','no_r_reason','no_l_reason','s_reason']    
    for key in dic:
        all_dic = {'video':0,'label':0}
        allMatch = True
        num = len(dic[key].annoList)
        for div in divList:
            for field in dic[key].annoList[0][div]:
                temp = dic[key].annoList[0][div][field]
                for i in range(num):
                    if temp != dic[key].annoList[i][div][field]:                 
                        allMatch = False
        if allMatch == True:
            count_all += 1
            all_dic['video'] = key
            all_dic['label'] = dic[key].annoList[0]
            allmatch_list.append(all_dic)
            allmatch_key.append(key)

    if writefile: 
        with open('allmatch_list', 'w') as fout:
            json.dump(allmatch_list, fout)
    print('allmatch:',len(allmatch_list))
    return allmatch_list, allmatch_key

allmatch_list, allmatch_key = allmatch(dic, writefile=False)

#%%
#ActionMatch
def action_match(dic, writefile=False):
    count_2in3 = 0
    action2in3_list = []
    worker_list = []
    for key in dic:  
        if key in allmatch_key:
            continue
        
        action_dic = {'video':0,'label':0}
        if len(dic[key].annoList) == 3:
            cnt = 1
            flag = {'0': False, '1':False, '2':False}
            if dic[key].annoList[0]['action'] == dic[key].annoList[1]['action']:
                flag['0'] = True
                cnt += 1
            if dic[key].annoList[0]['action'] == dic[key].annoList[2]['action']:
                flag['2'] = True
                cnt += 1
            if dic[key].annoList[1]['action'] == dic[key].annoList[2]['action']:
                flag['1'] = True
                cnt += 1
    
            if cnt/3 >= 2/3:
                count_2in3 += 1
                action = [k for k, v in flag.items() if v == True]
                index = int(action[0])
                action_dic['video'] = key
                action_dic['label'] = dic[key].annoList[index]
                
                worker_list.append(dic[key].workerList[index])        
                action2in3_list.append(action_dic)
    
    if writefile:
        with open('action_match_list.json', 'w') as fout:
            json.dump(action2in3_list, fout)
    print('action match (2/3):',len(action2in3_list))
    return action2in3_list

action_match_list = action_match(dic, writefile=False)
dic1_list = allmatch_list + action_match_list
total_list = dic1_list + dic2_list
#%%
with open('total_label.json', 'w') as fout:
    json.dump(total_list,fout)

#%% count label number and convert to one-hot-code
def one_hot_label(total_list, writefile=False):

    action_dic = {'f':0,'s':0,'l':0,'r':0,'ch_l':0,'ch_r':0,'cf':0}
    one_hot_list = []
    for item in total_list:
        array = [0,0,0,0,0,0,0]
        one_hot_dic = {'video':0,'label':0}
        one_hot_dic['video'] = item['video']
        for action in item['label']['action']:
            if action == 'forward' and item['label']['action'][action] == True:
                action_dic['f'] += 1
                array[0] = 1
            elif action == 'stop' and item['label']['action'][action] == True:
                action_dic['s'] += 1
                array[1] = 1
            elif action == 'turn_left' and item['label']['action'][action] == True:
                action_dic['l'] += 1
                array[2] = 1
            elif action == 'turn_right' and item['label']['action'][action] == True:
                action_dic['r'] += 1
                array[3] = 1
            elif action == 'change_left' and item['label']['action'][action] == True:
                action_dic['ch_l'] += 1
                array[4] = 1
            elif action == 'change_right' and item['label']['action'][action] == True:
                action_dic['ch_r'] += 1
                array[5] = 1
            elif action == 'confuse' and item['label']['action'][action] == True:
                action_dic['cf'] += 1
                array[6] = 1
        one_hot_dic['label'] = array
        one_hot_list.append(one_hot_dic)
    
    if writefile:
        with open('one_hot_label.json', 'w') as fout:
            json.dump(one_hot_list,fout)
        
    return one_hot_list, action_dic

#one_hot_list, action_count = one_hot_label(total_list, writefile=False)
one_hot, action_count = one_hot_label(total_list, writefile=False)
#%% count forward
cnt = 0
for item in one_hot:
    if item['label'] == [1,0,0,0,0,0,0]:
        cnt += 1
#%% change suffix
def mov2img(one_hot_list, suffix):
    import re
    import os
    for item in one_hot_list:
        video = item['video']
        name = os.path.splitext(video)[0]
        item['video'] = name + suffix
    
    return one_hot_list

one_hot_list = mov2img(one_hot_list, '.jpg')
total_list = mov2img(total_list,'')
#%%
with open('one_hot_label.json', 'w') as fout:
    json.dump(one_hot_list,fout)
#%%  train/test split
import numpy as np
from sklearn.model_selection import train_test_split

label = []
data = []
for item in total_list:
    data.append(item['video'])
    label.append(item['label'])

data = np.asarray(data)
label = np.asarray(label)
#%%
x_train, x_test,  y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=0)  
#%%
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.33, random_state=0)
#%%
x_train = x_train.tolist()
x_test = x_test.tolist()
x_val = x_val.tolist()
y_train = y_train.tolist()
y_test = y_test.tolist()
y_val = y_val.tolist()

#%%
train_set = []
test_set = []
val_set = []

for i in range(len(x_train)):
    dic = {}
    dic['video'] = x_train[i]
    dic['label'] = y_train[i]
    train_set.append(dic)
    
for i in range(len(x_test)):
    dic = {}
    dic['video'] = x_test[i]
    dic['label'] = y_test[i]
    test_set.append(dic)
    
for i in range(len(x_val)):
    dic = {}
    dic['video'] = x_val[i]
    dic['label'] = y_val[i]
    val_set.append(dic)
#%%
with open('train.json', 'w') as fout:
    json.dump(train_set,fout)

with open('test.json', 'w') as fout:
    json.dump(test_set,fout)

with open('val.json', 'w') as fout:
    json.dump(val_set,fout)

#%%
with open('x_test.txt', 'w') as f:
    for item in x_test:
        f.write(item+'\n')
        
with open('x_train.txt', 'w') as f:
    for item in x_train:
        f.write(item+'\n')

with open('x_val.txt', 'w') as f:
    for item in x_val:
        f.write(item+'\n')
        
#%%
import json
with open('val_one_hot_label.json','r') as fout:
    val = json.load(fout)
with open('test_one_hot_label.json','r') as fout:
    test = json.load(fout)
with open('train_one_hot_label.json','r') as fout:
    train = json.load(fout)
#%%
bdd12k = {}
for item in val:
    sub_dic = {'subset':'val','actions':0}
    sub_dic['actions'] = item['label']
    bdd12k[item['video']] = sub_dic
#%%
for item in test:
    sub_dic = {'subset':'test','actions':0}
    sub_dic['actions'] = item['label']
    bdd12k[item['video']] = sub_dic
#%%
for item in train:
    sub_dic = {'subset':'train','actions':0}
    sub_dic['actions'] = item['label']
    bdd12k[item['video']] = sub_dic
#%%
with open('bdd12k.json','w') as f:
    json.dump(bdd12k,f)