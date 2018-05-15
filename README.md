#### Download Links

Please download both the datasets _before_ cloning this repository <br/>

Processed Dataset - http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI/ <br/>
Raw Dataset - http://sorena.multicomp.cs.cmu.edu/downloads_raw/MOSEI <br/>

Specify Environment variable : 
export LC_ALL=C.UTF-8


#### Baselines and Metrics:
      The following metrics are defined :
      Metric 1 = MSE with sum across categories
      Random with Metric 1 :
          Train Set      : 0.73
          Validation Set : 0.63
          Test Set       : 0.66
      Vanilla Triple Attention with Metric 1 :
          Validation Set : 0.4899
          Test Set       : 0.4857
      Vanilla Dual Attention(V+A) with Metric 1  :
          Validation Set : [0.5157, 0.5211]
          Test Set       : [0.5103, 0.4986]
      Text Only with Metric 1 (LSTM -Only) :
          Validation Set : 0.59
          Test Set       : 
      Text Only with Metric 1 (TorchMoji Features) :
          Validation Set :
          Test Set       :
      Vision Only with Metric 1 :
          Validation Set :
          Test Set       :
      
      
#### Structure of Pickle Files

##### 1. Emotions.pkl 

Let *emo_intsts* be  = array([Anger_Intensity, Disgust_Intensity, Fear_Intensity, Happy_Intensity ,Sad_Intensity,   Surprise_Intensity] <br/>
{"Video Name": {"Segment ID i_1 ": emo_intsts,"Segment ID i_2 ": emo_intsts, .... ,"Segment ID i_n ": emo_intsts}} <br/>

There are 23453 segments in total in mosei (train+val+test)
>>> k=0
>>> for i in mosei_emotions.keys():
...   for j in mosei_emotions[i].keys():
...     k = k + 1

These emotion labels are emotion intensities, out of 23453 segments 6542 of them gives indecisive classes                                    
>>> k2=0
>>> for i in mosei_emotions.keys():
...   for j in mosei_emotions[i].keys():
...     if(max(mosei_emotions[i][j])==min(mosei_emotions[i][j])) or sorted(mosei_emotions[i][j],reverse=True)[0]==sorted(mosei_emotions[i][j],reverse=True)[1]:
...       k2 = k2+ 1






##### Train Set Emotion Intensity Stats: <br/>
 
        0-1 = 94964
        1-2 = 3275 
        2-3 = 515 
        Max Intensity  = 3.0
        Min Intensity  = 0.0 
        Mean Intensity = 0.17
        Mean Non-Zero Intensity = 0.74
        Mean Per Emotion Intensity = [ 0.1565  0.1233  0.0401  0.4836  0.1596  0.04842]
        
##### Validation Set Emotion Intensity Stats: <br/>
        
        0-1 = 11031 
        1-2 = 278 
        2-3 = 37 
        Max Intensity  = 3.0
        Min Intensity  = 0.0
        Mean Intensity = 0.15
        Mean Non-Zero Intensity = 0.68
        Mean Per Emotion Intensity = [ 0.1207   0.0888  0.0436  0.4341  0.1656  0.0497]
        
##### Test Set Emotion Intensity Stats: <br/>
        
        0-1 = 29574 
        1-2 = 914 
        2-3 = 130 
        Max Intensity  = 3.0
        Min Intensity  = 0.0
        Mean Intensity = 0.16
        Mean Non-Zero Intensity = 0.72
        Mean Per Emotion Intensity = [ 0.1602  0.1140   0.0409   0.4685  0.1407  0.0437]

##### 2. Words.pkl 

##### 3. Embeddings.pkl 

To load the embeddings from CMU-MOSEI, run the following
1. build softlinks in mmdata/data/pickled/
$ cd mmdata/data/pickled/
$ ln -s ../../../*.pkl .
2. run this
$ python3 creating_text_files-SDKload.py

Two folders will be created: 
text_files_segbased   : segment-base embeddings
text_files_videobased : segment-base embeddings, but each embedding files has a scope covering the whole video




##### 4. Train/Test/Valid.pkl 

Contains a set of all the train/test/validation video names <br/>

Length of Dataset - 3228 Videos divided into 22677 Video Clips of ~3-8 seconds <br/>
Length of Training Set - 2250 Videos divided into 16303 Video Clips <br/>
Length of Validation Set - 300 Videos divided into 1861 Video Clips <br/>
Length of Test Set - 678 Videos divided into 4645 Video Clips <br/>
Length of Truncated Set: <br/>
('train', 11112) <br/>
('test', 3303) <br/>
('val', 1341) <br/>

##### Train Set Video Length Stats: <br/>
 
        0-2 = 322
        2-4 = 2975
        4-6 = 3979
        6-8 = 3340
        8-10 = 2111
        10-15 = 2398
        15-20 = 780
        20+ = 398

        
##### Validation Set Video Length Stats: <br/>
        
        0-2 = 28
        2-4 = 260
        4-6 = 434
        6-8 = 400
        8-10 = 291
        10-15 = 324
        15-20 = 80
        20+ = 44

        
##### Test Set Video Length Stats: <br/>
        
        0-2 = 80
        2-4 = 845
        4-6 = 1246
        6-8 = 1019
        8-10 = 743
        10-15 = 689
        15-20 = 214
        20+ = 139


##### 5. Facet.pkl 
Let *facet_features* be  = array([feature_1_val,feature_2_val,....,feature_35_val]) <br/>
There are 35 features for each frame <br/>
{ "facet" :{"Video Name": {"Segment ID i_1 ": ((start_time_frame_1,end_time_frame_1,facet_features),...      (start_time_frame_n,end_time_frame_n,facet_features)),"Segment ID i_2 ": ..., .... ,"Segment ID i_n ": ....}}}

##### 6. Sentiments.pkl 

##### 7. Covarep.pkl 

Let *covarep_features* be  = array([feature_1_val,feature_2_val,....,feature_74_val]) <br/>
COVAREP features are taken at a time interval of 0.01sec(10ms) which is the original sampling rate of the COVAREP authors.<br/>
There are 74 features for each 0.01 segment. <br/>
{ "facet" :{"Video Name": {"Segment ID i_1 ": ((start_time_frame_1,end_time_frame_1,covarep_features),...      (start_time_frame_n,end_time_frame_n,covarep_features)),"Segment ID i_2 ": ..., .... ,"Segment ID i_n ": ....}}}

Due to 43 features in some files the following number of features were present for vision but couldn't be used for audio(and hence removed from all three folds): <br/>
1. Test : 1322 <br/>
2. Train : 5105 <br/>
3. Val : 494 <br/>

#### TO DOs After Cloning the Repository

1. Put all downloaded pickle files in the same directory. <br/>
2. Run creating_audio_files.py using python2.7 (This might take a while depending on your machine). This will generate a folder with all the audio_files containing covarep features with the corresponding video name. <br/>
3. Run transfer_valid_audio.py <br/>
4. Run dual_attention.py <br/>
