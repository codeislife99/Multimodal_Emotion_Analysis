#### Download Links

Please download both the datasets _before_ cloning this repository <br/>

Processed Dataset - http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI/ <br/>
Raw Dataset - http://sorena.multicomp.cs.cmu.edu/downloads_raw/MOSEI <br/>


#### Structure of Pickle Files

##### 1. Emotions.pkl 

Let *emo_intsts* be  = array([Anger_Intensity, Disgust_Intensity, Fear_Intensity, Happy_Intensity ,Sad_Intensity,   Surprise_Intensity] <br/>
{"Video Name": {"Segment ID i_1 ": emo_intsts,"Segment ID i_2 ": emo_intsts, .... ,"Segment ID i_n ": emo_intsts}}

##### 2. Words.pkl 

##### 3. Embeddings.pkl 

##### 4. Train/Test/Valid.pkl 

Contains a set of all the train/test/validation video names
Length of Dataset - 3228 Videos divided into 17728 Video Clips of ~3-8 seconds
Length of Training Set - 2250 Videos
Length of Validation Set - 300 Videos
Length of Test Set - 678 Videos

##### 5. Facet.pkl 

##### 6. Sentiments.pkl 

##### 7. Covarep.pkl 



