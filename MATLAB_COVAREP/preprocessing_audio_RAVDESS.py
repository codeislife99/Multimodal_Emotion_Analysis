import matlab.engine
import time

# for actor_no in range(1,25,1):
# 	actor_str = str(actor_no).zfill(2)
# with open('COVAREP_feature_extraction.m','r') as f:
# 	data = f.readlines()
# data[50] = "in_dir = './WAVFiles/train/';"+"\n"
# with open("COVAREP_feature_extraction.m",'w') as file:
# 	file.writelines(data)

# eng = matlab.engine.start_matlab("-desktop")
# # future = matlab.engine.start_matlab(async=True)
# print("Engine Started !!")
# # eng = future.result()
# eng.COVAREP_feature_extraction(nargout = 0)
# eng.quit()
# print("Engine Stopped")

with open('COVAREP_feature_extraction.m','r') as f:
	data = f.readlines()
data[50] = "in_dir = './WAVFiles/test/';"+"\n"
with open("COVAREP_feature_extraction.m",'w') as file:
	file.writelines(data)

eng = matlab.engine.start_matlab("-desktop")
# future = matlab.engine.start_matlab(async=True)
print("Engine Started !!")
# eng = future.result()
eng.COVAREP_feature_extraction(nargout = 0)
eng.quit()
print("Engine Stopped")

with open('COVAREP_feature_extraction.m','r') as f:
	data = f.readlines()
data[50] = "in_dir = './WAVFiles/valid/';"+"\n"
with open("COVAREP_feature_extraction.m",'w') as file:
	file.writelines(data)

eng = matlab.engine.start_matlab("-desktop")
# future = matlab.engine.start_matlab(async=True)
print("Engine Started !!")
# eng = future.result()
eng.COVAREP_feature_extraction(nargout = 0)
eng.quit()
print("Engine Stopped")

