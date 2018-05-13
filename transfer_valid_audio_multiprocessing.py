import cPickle as pickle
import os
import glob

<<<<<<< HEAD
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

def transefer_audio_files(split):

=======
os.system("mkdir -p audio_files_74")
splits = ['train','test','val']
for split in splits:
	os.system("mkdir -p audio_files_74/"+split)
>>>>>>> ad8638062d08f715fc8203abbfbe558f6df30d0b
	path = './audio_files/'+split+'/*.pkl'
	count = 0
	for file in glob.glob(path):
		with open(file,'rb') as f:
			data = pickle.load(f)
			# print(data.shape[1])
			if data.shape[1] == 74:
				os.system("mv "+file+" ./audio_files_74/"+split+"/")
				# print("mv "+file+" ./audio_files_74/"+split+"/")
<<<<<<< HEAD
			count += 1
	print(split, count)
	return count


pool = Pool(cpu_count()*2)

splits = ['train','test','val']

count = 0
try:
	for cnt in pool.imap_unordered(transefer_audio_files, splits):
		count += cnt
		print (cnt)

except Exception as e:
	print("exception:",e)
	pool.close()
	pool.terminate()

print(count)
=======
				count += 1
	print(split, count)
>>>>>>> ad8638062d08f715fc8203abbfbe558f6df30d0b
