# Action Ground Truth using method of [BDD_drive_model](https://github.com/gy20073/BDD_Driving_Model)

#### Select videos with complex scenes
+ Standard: >=5 pedestrians and >=5 cars at the 10th second of each 40-second video (since the it's only frame with bbox ground truth)
+ Selected video clips id in train_id.txt (5070) and val_id.txt(743).

#### IMU info convert to action gt
+ Use location info in IMU, especially speed and course(angle).
+ Action dict: 'straight': 0, 'slow_or_stop': 1, 'turn_left': 2, 'turn_right': 3
+ threshold for stop: 1e-3, threshold for deceleration is 1.0.(detect significant slow down that is not due to going to turn)
+ Some selected videos have bad timestamp, so in action_gt, its action will be 'N/A'.

#### format of action ground truth
+ 'id': selected video name, 'action': action number(0-3) for 602 frames (40s).
