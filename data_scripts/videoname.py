import ast
import os
import shutil

with open('x_test.txt','r') as f:
    x_test = f.readlines()

with open('x_val.txt','r') as f:
    x_val = f.readlines()

test = []
val = []
for name in x_test:
    name = name.strip("\n")
    test.append(name)

for name in x_val:
    name = name.strip("\n")
    val.append(name)



path = '/data6/SRIP19_SelfDriving/bdd12k/data/train/'
os.chdir(path)
list = os.listdir(path)
for video in list:
    name = os.path.splitext(video)
    print(name[0])
    if name[0] in test:
        shutil.move(video,'/data6/SRIP19_SelfDriving/bdd12k/data/test/')
    elif name[0] in val:
        shutil.move(video,'/data6/SRIP19_SelfDriving/bdd12k/data/val/' )


