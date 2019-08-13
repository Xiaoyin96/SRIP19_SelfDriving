import os

path = '/data6/SRIP19_SelfDriving/bdd12k/data/train/img/'
os.chdir(path)
#
list = os.listdir(path)
# img = '12345_1_083.jpg'
# list = ['12345_1_001.jpg','12345_1_002.jpg','12345_1_003.jpg','12345_1_004.jpg','12345_2_002.jpg']
print(len(list))
name_dic = {}
for img in list:
    line = img.split('_')
    name = line[0] + '_' + line[1]
    num = os.path.splitext(line[2])[0]
    if name not in name_dic.keys():
        name_dic[name] = num
    else:
        if num > name_dic[name]:
            name_dic[name] = num
print(name_dic)

name_list = []
for key, value in name_dic.items():
    name = key +'_'+ value+'.jpg'
    name_list.append(name)