# SRIP19: Self-Driving and Multi-task Learning
## Content
+ [BDD_action_gt](./BDD_action_gt): use IMU and GPS info from BDD dataset to generate single action ground truth.
+ [multiple_action_labels](./multiple_action_label): use AWS Mturk to label multiple actions and reasons of selected 12k BDD videos.
+ [data_info](./data_info): contains names of train, test and validation datasets.
+ [mask-rcnn](./mask-rcnn): Mask-RCNN model, forker from Facebook AI group and modified with action prediction.
+ [I3D](./I3D): inflated Conv3D model, adapted to Pytorch 1.0 and our new annotated BDD multi-action dataset.
+ [maskrcnn-video](./ maskrcnn-video): Using our customized I3D backbone with 640x360 image sequences input to extract glob features and roi features with selectors, performing end-to-end training.
## Papers for reference
### Self-Driving Review
+ [Deep Multi-modal Object Detection and Semantic Segmentation for Autonomous Driving: Datasets, Methods, and Challenges.](https://arxiv.org/pdf/1902.07830.pdf)
### Existing Self-Driving Datasets
+ [BDD100K: A Diverse Driving Video Database with Scalable Annotation Tooling](https://arxiv.org/pdf/1805.04687.pdf)
+ [The ApolloScape Dataset for Autonomous Driving](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8575295&tag=1)
+ [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://www.cityscapes-dataset.com/wordpress/wp-content/papercite-data/pdf/cordts2016cityscapes.pdf)
### Multi-task Learning methods
+ [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098.pdf)
+ [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)
+ [MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving](https://arxiv.org/pdf/1612.07695.pdf)
+ [UberNet: Training a Universal Convolutional Neural Network for Low-, Mid-,and High-Level Vision using Diverse Datasets and Limited Memory](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kokkinos_Ubernet_Training_a_CVPR_2017_paper.pdf)
+ [Cross-stitch Networks for Multi-task Learning](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kokkinos_Ubernet_Training_a_CVPR_2017_paper.pdf)
### Video Prediction in self-driving
+ [Trajectory prediction summary](https://github.com/xuehaouwa/Awesome-Trajectory-Prediction)
+ [DESIRE: Distant Future Prediction in Dynamic Scenes with Interacting Agents](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lee_DESIRE_Distant_Future_CVPR_2017_paper.pdf)
	+ Code: https://github.com/yadrimz/DESIRE
+ [Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion
Forecasting with a Single Convolutional Net](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf)
+ [Predicting Deeper into the Future of Semantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Luc_Predicting_Deeper_Into_ICCV_2017_paper.pdf)
	+ Code: https://github.com/facebookresearch/SegmPred
+ [Predicting Future Instance Segmentation by Forecasting Convolutional Features](https://arxiv.org/pdf/1803.11496.pdf)
	+ Code: https://github.com/facebookresearch/instpred 
#### Some Prediction models
Summary: [Video prediction papers with code](https://paperswithcode.com/task/video-prediction)
+ [Deep Multi-scale video prediction beyond mean square error](https://arxiv.org/pdf/1511.05440.pdf)
	+ Code in lua: https://github.com/coupriec/VideoPredictionICLR2016 
	+ Code in tf: https://github.com/dyelax/Adversarial_Video_Generation
+ [Prediction Under Uncertainty with Error-Encoding Networks](https://arxiv.org/pdf/1711.04994.pdf)
	+ Code: https://github.com/mbhenaff/EEN
+ [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/pdf/1605.08104.pdf)
	+ Code: https://github.com/coxlab/prednet
+ [Peeking into the Future: Predicting Future Person Activities and Locations in Video](https://arxiv.org/pdf/1902.03748.pdf)
	+ Code: https://github.com/google/next-prediction
#### Some semantic segmentation approaches
Summary: [Semantic segmentation papers with code](https://paperswithcode.com/task/semantic-segmentation)
+ [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/pdf/1902.04502.pdf)
	+ Code: https://github.com/DeepVoltaire/Fast-SCNN
+ [DeeplabV3: Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)
	+ Code: https://github.com/fregu856/deeplabv3
----
### Useful github repo
+ [Mask-rcnn benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark)
+ [Inflated_convnets_pytorch](https://github.com/hassony2/inflated_convnets_pytorch)
+ [pytorch_i3d with training](https://github.com/piergiaj/pytorch-i3d)
