import serial 
import syslog
import time
import cv2
import dlib
import serial
#from threading import Thread
#from queue import Queue
import argparse
import cv2
import numpy as np
import os
from multiprocessing import Process, Queue


detector = dlib.get_frontal_face_detector()




#The following line is for serial over GPIO
try_ports = ['/dev/ttyACM0','/dev/ttyACM1','/dev/ttyACM2','/dev/ttyACM3']
port = '/dev/ttyACM1' # note I'm using Mac OS-X
os.system("sudo chmod a+rw "+port)
ard = serial.Serial(port,9600,timeout=5)
time.sleep(2) # wait for Arduino


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def sendtoard(q):
    condition = 1
    while (1):
    # Serial write section
	print ('Alexa')

        if (condition == 2):
            try:
                ard.flush()
                condition = 1
                print (condition)
            except:
                pass
        print("Sending1")
	print (q)
        val2x = q.get()
	print ('Made it past first')
	print (val2x)
	print (q)
        val2y = q.get()
	print ('Made it past second')
	print (val2y)
        print ("Python value sent: ")
        print (format(int(val2x), '03d')+format(int(val2y), '03d'))
        
        if (condition == 1):
            try:
                print ((str(val2x)+","+str(val2y)+"_").encode('utf-8'))
                ard.write((str(int(val2x))+","+str(int(val2y))+"_").encode('utf-8'))
                condition = 2
            except:
                pass
        time.sleep(0.15) # I shortened this to match the new value in your Arduino code

        
    else:
        print ("Exiting")
    exit()

def show_webcam(mirror, q):
    cam = cv2.VideoCapture(-1)
    while True:
        ret_val, img = cam.read()
        
        if mirror: 
            img = cv2.flip(img, 1)
            img = adjust_gamma(img, 1.5)
            try:
                dets = detector(img)

                for i, d in enumerate(dets):                    
                    cv2.rectangle(img, (d.left(), d.top()), ( d.right(), d.bottom()), (255,0,0), 2)
                    centroid=((d.left()+d.right())/2, (d.top()+ d.bottom())/2)
                    val1x=centroid[0]
                    val1y=centroid[1]
                    with q.mutex:
                    	q.queue.clear()
                    q.put(val1x)                    
                    q.put(val1y)
                    
                    # the frame seems to have 620 horizontal pixels the val tells you how many pixels from the 
		    # centre he centroid of face is ocated. -ve means image centroid is at the right f face centroid
                    #+ve means viceversa magnitude of val indicates how shifted it is

                cv2.imshow('my webcam', img)

                if cv2.waitKey(1) == 27: 
                    break  # esc to quit
            except:
                continue
    cv2.destroyAllWindows()





if __name__ == "__main__":
    q = Queue()
    t2 = Process(target = show_webcam, args=(True,q))
    t1 = Process(target = sendtoard, args=((q),))

    t1.start()
    t2.start()

    
    q.close()
    q.join_thread()
    t2.join()
    t1.join()


