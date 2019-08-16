### multiple-choice action label

- Dataset: BDD

- format of annotation:

  - video: video id

  - label: contains action and reason for each video

  - action: forward, stop, turn left, turn right, change lane to left('change_left'), change lane to right('change_right'), can't change lane to left('cant_change_left'), can't change lane to right('cant_change_right'), confuse

  - reason: (bool type)

    - forward('f_reason'): 	
      - follow traffic('f_follow_traffic')
      - road clear('f_road_clear')
      - traffic light is green or yellow ('f_traffic_light')
    - stop('s_reason'):
      - obstacle: car ('s_ob_car')
      - obstacle: pedestrian ('s_ob_ped')
      - obstacle: rider ('s_ob_rider')
      - traffic light is red or yellow ('s_traffic_light')
      - other ('s_other')
    - turn left('l_reason'):
      - front car turning left ('l_front_car')
      - in the left lane ('l_lane')
      - traffic light or sign allows turning left ('l_traffic_light')

    - turn right('r_reason'):
      - front car turning right ('r_front_car')
      - in the right lane ('r_lane')
      - traffic light or sign allows turning right ('r_traffic_light')
    - can't change lane to left('no_l_reason'):
      - cars in the left lane('no_l_car')
      - no left lane('no_l_lane')
      - can't cross solid line('no_l_solid_line')
    - can't change lane to right('no_r_reason'):
      - cars in the left lane('no_r_car')
      - no left lane('no_r_lane')
      - can't cross solid line('no_r_solid_line')
- Statistics: 
  - total: 11624
  - forward: 8478
  - stop: 3325
  - left turn: 465
  - right turn: 503
  - change lane to left: 3335
  - change lane to right: 3620
  - confuse: 89
- Action one-hot-code format: 
  - number of class: 7
  - order: forward, stop, turn left, turn right, change lane to left, change lane to right, confuse
  - e.g. [1,0,0,0,1,0,0] means forward and change lane to left.
- Reason one-hot code format:
  - number of class: 21
  - order: f_follow_traffic, f_road_clear, f_traffic_light, s_ob_car, s_ob_ped, s_ob_rider, s_other, s_traffic_light, s_traffic_sign, l_front_car, l_lane, l_traffic_light, r_front_car, r_lane, r_traffic_light, no_l_car, no_l_lane, no_l_solid_line, 
  no_r_car, no_r_lane, no_r_solid_line.
- Train, test split:
  - train: test: val = 7:2:1


    
