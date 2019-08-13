import os

os.chdir('/data6/SRIP19_SelfDriving/bdd12k/data/val/')

list = os.listdir('/data6/SRIP19_SelfDriving/bdd12k/data/val/')
for video in list:
    video_name = os.path.splitext(video)[0]
    print(video_name)
    command = 'ffmpeg -i ' + video + ' -r 15 '+ 'img/' + video_name + '_%02d.jpg'
    os.system(command)
