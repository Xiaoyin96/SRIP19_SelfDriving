
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'home/selfdriving/maskrcnn-benchmark/demo'))
	print(os.getcwd())
except:
	pass


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "/home/selfdriving/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_C4_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
print(image)

predictions = coco_demo.run_on_opencv_image(image)
print(predictions)
