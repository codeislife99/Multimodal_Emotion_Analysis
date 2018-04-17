import cv2
import glob 
for filename in glob.iglob('./VideoFlash/*.flv'):
	cap = cv2.VideoCapture(filename)  # Read test video into cap variable
	fourcc = cv2.VideoWriter_fourcc(*'MPEG')

	# if cap.isOpened():
	#     ret, frame = cap.read()  # Read the frame and store it in frame variable
	#     height = frame.shape[0]  # No of pixels along Y axis
	#     width = frame.shape[1]
	# print(width)
	# print(height)
	out = cv2.VideoWriter('./VideoFlash_30fps2/'+str(filename[13:-4])+'.mp4', fourcc, 30.0,
	                  (480, 360))
	i = 0
	while cap.isOpened():
		_, frame = cap.read()
		if frame is None:
			break
		# cv2.imshow("Frame", frame)

		out.write(frame)
		i+=1
		# print(i)
	print(str(filename[13:-4]), i)		
	cap.release()
	out.release()
