import torch

content = torch.load('/home/selfdriving/mrcnn/output/inference/bdd100k_val/final_preds.pth')
train_dict = {}
print('-'*10,'train')
for vid, item in content['train'].items():
    print(vid)
    box_list = []
    for i in range(1,len(item)+1,2): # 16 frames sampled from 32 frames
        box_list.append(item[i])
    train_dict[vid] = box_list

torch.save(train_dict, '/home/selfdriving/mrcnn/output/inference/bdd100k_val/bbox_train_16.pth')

val_dict = {}
print('-'*10,'val')
for vid, item in content['val'].items():
    print(vid)
    box_list = []
    for i in range(1,len(item)+1,2): # 16 frames sampled from 32 frames
        box_list.append(item[i])
    val_dict[vid] = box_list
torch.save(train_dict, '/home/selfdriving/mrcnn/output/inference/bdd100k_val/bbox_val_16.pth')

