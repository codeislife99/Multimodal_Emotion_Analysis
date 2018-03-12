import subprocess
import cv2
import dlib
import h5py
import numpy as np
import glob 
import os
import copy
from moviepy.editor import VideoFileClip
validation = ["1001","1002","1003","1004","1005"]
test = ["1006","1007","1008","1009","1010"]
emo_dict = { "ANG": "A" , "DIS":"D", "FEA":"F" , "HAP":"H", "NEU":"N", "SAD":"S"}
def getLength(filename):
	clip = VideoFileClip(filename)
	return clip.duration
min_frames =50
count = []
for filename in glob.iglob('./VideoFlash_30fps2/*.mp4'):
	# print(filename)
	video_number = str(filename[20:24])
	if video_number in validation:
		video_type = "val_30"
	elif video_number in test:
		video_type = "test_30"
	else:
		video_type = "train_30"
	emotion = str(emo_dict[filename[29:32]])
	dir_name = "./"+video_type+"/"+str(emotion)+"/"+str(filename[20:-4])

	# list = os.listdir(dir_name) # dir is your directory path
	# number_files = len(list)
	# seconds = getLength(filename)
	# if seconds*30-number_files > 30:
	# 	print number_files,seconds
	# 	print(filename)
	# 	if number_files >= 0:
	path_to_video = filename
	vc = cv2.VideoCapture(path_to_video)
	c = 0
	detector = dlib.get_frontal_face_detector()
	face_img = []
	while vc.isOpened():
	    try:
	        _, img_frame = vc.read()
	        rects = detector(img_frame, 0)
	    except:
	        break
	    if len(rects) > 0:
	        try:
	            face_img = img_frame[rects[0].top():rects[0].bottom(), rects[0].left():rects[0].right()]
	            face_img = cv2.resize(face_img, None, fx=224 / float(face_img.shape[1]), fy=224 / float(face_img.shape[0]),
	                                  interpolation=cv2.INTER_CUBIC)
	        except:
	            face_img = copy.copy(prev_image)
	            pass
	    c += 1
	    # if c%100 == 0:
	    # 	print(str(c))
	    if(len(face_img) == 0):
	    	continue
	    cv2.imwrite(dir_name+'_frame_' + str(c) + '.jpg', face_img)
	    # print(dir_name+'_frame_' + str(c) + '.jpg')
	    prev_image = copy.copy(face_img)
	    # print("c = " + c)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	vc.release()
	cv2.destroyAllWindows()
	if c<min_frames:
		count.append(str(filename[20:-4])+" = "+ str(c))
	print(str(filename[20:-4])," = ", str(c))
print(count)