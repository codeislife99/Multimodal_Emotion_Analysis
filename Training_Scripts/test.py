import glob
import pandas as pd
import numpy as np
for csv_file_path in glob.glob("../processed/01-01-*-"+"02.csv"):
	print(csv_file_path)
	df = pd.read_csv(csv_file_path)
	df0 = df.iloc[list(range(0,45,1)),list(range(5,22,1))]
	df1 = df.iloc[list(range(45,90,1)),list(range(5,22,1))]
	df2 = df.iloc[list(range(-90,-45,1)),list(range(5,22,1))]
	df3 = df.iloc[list(range(-45,0,1)),list(range(5,22,1))]	
	target_df0 = np.array(df0.values , dtype = np.float32)
	target_df1 = np.array(df1.values , dtype = np.float32)
	target_df2 = np.array(df2.values , dtype = np.float32)
	target_df3 = np.array(df3.values , dtype = np.float32)
	print("Target Df 0 = ",  target_df0)
	print("Target Df 1 = ",  target_df1)
	print("Target Df 2 = ",  target_df2)
	print("Target Df 3 = ",  target_df3)

	break