#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:31:40 2019

@author: xiaoyin
"""

import json
with open('total_label.json','r') as fout:
    total_label = json.load(fout)
    
#%%
with open ('/Users/xiaoyin/Desktop/HIT test/train.csv') as f:
    train_set = f.readlines()
    
with open ('/Users/xiaoyin/Desktop/HIT test/test.csv') as f:
    test_set = f.readlines()
    
with open ('/Users/xiaoyin/Desktop/HIT test/val.csv') as f:
    val_set = f.readlines()

#%% reason to one-hot-label
import numpy as np

for item in totoal_label:
    reason = [0] *21
    for f_reason in item['label']['f_reason']:
        if f_reason == 'f_follow_traffic' and item['label']['f_reason'][f_reason] == True:
            reason[0] = 1
        elif f_reason == 'f_road_clear' and item['label']['f_reason'][f_reason] == True:
            reason[1] = 1
        elif f_reason == 'f_traffic_light' and item['label']['f_reason'][f_reason] == True:
            reason[2] = 1
    for s_reason in item['label']['s_reason']:
        if s_reason == 's_ob_car' and item['label']['s_reason'][s_reason] == True:
            reason[3] = 1
        elif s_reason == 's_ob_ped' and item['label']['s_reason'][s_reason] == True:
            reason[4] = 1
        elif s_reason == 's_ob_rider' and item['label']['s_reason'][s_reason] == True:
            reason[5] = 1
        elif s_reason == 's_other' and item['label']['s_reason'][s_reason] == True:
            reason[6] = 1
        elif s_reason == 's_traffic_light' and item['label']['s_reason'][s_reason] == True:
            reason[7] = 1
        elif s_reason == 's_traffic_sign' and item['label']['s_reason'][s_reason] == True:
            reason[8] = 1
    for l_reason in item['label']['l_reason']:
        if l_reason == 'l_front_car' and item['label']['l_reason'][l_reason] == True:
            reason[9] = 1
        elif l_reason == 'l_lane' and item['label']['l_reason'][l_reason] == True:
            reason[10] = 1
        elif l_reason == 'l_traffic_light' and item['label']['l_reason'][l_reason] == True:
            reason[11] = 1
    for r_reason in item['label']['r_reason']:
        if r_reason == 'r_front_car' and item['label']['r_reason'][r_reason] == True:
            reason[12] = 1
        elif r_reason == 'r_lane' and item['label']['r_reason'][r_reason] == True:
            reason[13] = 1
        elif r_reason == 'r_traffic_light' and item['label']['r_reason'][r_reason] == True:
            reason[14] = 1
    for no_l_reason in item['label']['no_l_reason']:
        if no_l_reason == 'no_l_car' and item['label']['no_l_reason'][no_l_reason] == True:
            reason[15] = 1
        elif no_l_reason == 'no_l_lane' and item['label']['no_l_reason'][no_l_reason] == True:
            reason[16] = 1
        elif no_l_reason == 'no_l_solid_line' and item['label']['no_l_reason'][no_l_reason] == True:
            reason[17] = 1
    for no_r_reason in item['label']['no_r_reason']:
        if no_r_reason == 'no_r_car' and item['label']['no_r_reason'][no_r_reason] == True:
            reason[18] = 1
        elif no_r_reason == 'no_r_lane' and item['label']['no_r_reason'][no_r_reason] == True:
            reason[19] = 1
        elif no_r_reason == 'no_r_solid_line' and item['label']['no_r_reason'][no_r_reason] == True:
            reason[20] = 1