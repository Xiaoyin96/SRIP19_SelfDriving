import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join

imgpath = '/data6/SRIP19_SelfDriving/bdd12k/train/img/'
imglist = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]
imgName = imgpath+imglist[0]

# imgName = '/data6/SRIP19_SelfDriving/bdd100k/images/10k/val/a91b7555-00000495.jpg'
# img = np.load(imgName)

img = np.array(Image.open(imgName))
plt.imshow(img)
n = 1
n2 = 2

# import matplotlib.pyplot as plt
# import numpy as np
# x = np.linspace(0, 20, 100)
# plt.plot(x, np.sin(x))
# plt.show(block=False)
# input('press <ENTER> to continue')