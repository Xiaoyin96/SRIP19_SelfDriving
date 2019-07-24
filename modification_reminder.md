How to Use maskrcnn_benchmark to train our own data?
maskrcnn_benchmark/data/build.py: Line 39
maskrcnn_benchmark/data/datasets/__init__.py: add your new dataset
markrcnn_benchmark/data/datasets/: add your new dataset decription .py file. See example https://github.com/xanderchf/faster-rcnn-KITTI-BDD100K/blob/master/maskrcnn_benchmark/data/datasets/bdd100k.py
markrcnn_benchmark/config/paths_catalog.py: add your new dataset directory.
