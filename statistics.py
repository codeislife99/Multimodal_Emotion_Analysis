import cPickle as pickle
import os
import glob

splits = ['train','val']
bins = ["0-1","1-2","2-3","3-4","4-5","5-6","6-7","7-8","8-9","9-10"]
emo_bins = [{"0-1":0,"1-2":0,"2-3":0,"3-4":0,"4-5":0,"5-6":0,"6-7":0,"7-8":0,"8-9":0,"9-10":0},
			{"0-1":0,"1-2":0,"2-3":0,"3-4":0,"4-5":0,"5-6":0,"6-7":0,"7-8":0,"8-9":0,"9-10":0}]

for i,split in enumerate(splits):
	path = './gt_emotions_files/'+split+'/*.pkl'
	intensity = []
	non_zero_intensity = []
	max_intensity = -1
	for file in glob.glob(path):
		with open(file,'rb') as f:
			data=pickle.load(f)
			for ele in data:
				intensity.append(ele)
				max_intensity = max(ele,max_intensity)
				if ele>0.01:
					non_zero_intensity.append(ele)
				for limit in range(1,11,1):
					if ele <= limit:
						emo_bins[i][bins[limit-1]] = emo_bins[i][bins[limit-1]]+1
						break

	print(split + " emotion stats:")
	for limit in range(1,11,1):
		print(bins[limit-1]+" = "+ str(emo_bins[i][bins[limit-1]]))
	print("Max Intensity = ", max_intensity)

