import matlab.engine
from multiprocessing import Process,Queue,Value,Array,Lock


def run():
	eng = matlab.engine.start_matlab("-desktop")
	# eng.COVAREP_feature_extraction(nargout = 0)
	# print("YOLO")
	# eng.COVAREP_feature_extraction(nargout = 0)
	# print("YOLO2")	

    for idx in range(3):
        # start_time = time.time()
        with open('COVAREP_feature_extraction.m','r') as f:
            data = f.readlines()
        data[50] = "in_dir = "+"'.'"+"\n" # Put your path here
        with open("COVAREP_feature_extraction.m",'w') as file:
            file.writelines(data)
        eng.COVAREP_feature_extraction(nargout = 0)
        print("YOLO" + str(idx))
if __name__ == '__main__':
	p1 = Process(target=run, args=())
	p1.start()
