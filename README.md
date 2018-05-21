#### Download Links

Please download both the datasets _before_ cloning this repository <br/>

Processed Dataset - http://sorena.multicomp.cs.cmu.edu/downloads/MOSEI/ <br/>
Raw Dataset - http://sorena.multicomp.cs.cmu.edu/downloads_raw/MOSEI <br/>

Specify Environment variable : 
export LC_ALL=C.UTF-8

Random seed initialization
https://discuss.pytorch.org/t/random-seed-initialization/7854
torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)


#### Baselines and Metrics:
      The following metrics are defined :
      Metric 1 = MSE with sum across categories [0.73]
      Metric 2 = MAE with sum across categories [0.8686]
      Metric 3 = Huber Loss (Smooth L1 Loss)    [0.3263]
      

 | Model    | Modality   | Metric 1 Val  | Metric 1 Test  | Metric 1 Train | Metric 2 Val  | Metric 2 Test | Metric 3 Val| Metric 3 Test| 
 |:--------:|:----------:|:-------------:|:--------------:|:--------------:|:-------------:|:-------------:|:-----------:|:------------:|
 | Random              | -          | 0.63    |   0.647   |  | 0.7938|0.8121|0.2822|0.2946|
 | Triple Attention    | V+A+T      | 0.4765  |   0.4709  |  |         | | | |
 | Triple Attention-scalar | V+A+T(scalarAttTime) __5.pth | 0.5193 | 0.5346 | 0.5986 |   |   | | |
 | Triple Attention-scalar | V+A+T(scalarAttTime) __6.pth | 0.5439 | 0.5520 | 0.5742 |   | | | |
 | Triple Attention-scalar-1024 | V+A+T(scalarAttTime)- __3.pth | 0.4816 | 0.4790 | 0.4605 |   | | | |
 | Triple Attention-scalar-1024 | V+A+T(scalarAttTime)- __4.pth | 0.4884 | 0.4806 | 0.4345 |  | | | |
 | Triple Attention-scalar-1024 | V+A+T(scalarAttTime)- __5.pth | 0.4789 | 0.4745 | 0.4087 | | | | |
 | Triple Attention-scalar-1024 | V+A+T(scalarAttTime)- __6.pth | 0.4870 | 0.4929 | 0.3830 | | | | |
 | Triple Attention-1024 | V+A+T(attElement)- __3.pth | 0.4910 | 0.4859 | 0.4604 |   | | | |
 | Triple Attention-1024 | V+A+T(attElement)- __4.pth | 0.5003 | 0.4793 | 0.4303 |  | | | |
 | Triple Attention-1024 | V+A+T(attTime)- __3.pth | 0.4855 | 0.4919 | 0.4671 |    | | | |
 | Triple Attention-1024 | V+A+T(attTime)- __4.pth | 0.4888 | 0.4889 | 0.4409 |     | | | |
 | Triple Attention-1024 | V+A+T(attTime)- __5.pth | 0.4761 | 0.4816 | 0.4144 |     | | | |
 | Triple Attention-1024 | V+A+T(attTime)- __6.pth | 0.4970 | 0.4973 | 0.3879 |     | | | |
 | Triple Attention-1024 | V+A+T(attTime)- __7.pth | 0.5103 | 0.5121 | 0.3608 |     | | | |
 | Triple Attention-1024-gated | V+A+T(attTime)- __4.pth | 0.4746 | 0.4631 | 0.4512 |     | | | |
 | Triple Attention-1024-gated | V+A+T(attTime)- __5.pth | 0.4911 | 0.4734 | 0.4225 |     | | | |
 | Triple Attention-scalar-1024-gated | V+A+T(scalarAttTime)- __4.pth | 0.4812 | 0.4683 | 0.4470 |     | | | |
 | Triple Attention-scalar-1024-gated | V+A+T(scalarAttTime)- __5.pth | 0.4838 | 0.4649 | 0.4262 |     | | | |
 | Early Concatenation | V+A+T      |         |           |   |        | | | |
 | Late Weighting      | V+A+T __0.pth |         |           |   |         | | | |
 | Late Weighting      | V+A+T __1.pth |         |           |   |         | | | |
 | Late Weighting      | V+A+T __2.pth |         |           |   |         | | | |
 | Late Weighting      | V+A+T __3.pth |         |           |   |         | | | |
 | Dual Attention      | V+T        |         |           |   |        | | | |
 | Dual Attention      | V+A        |  0.5157 | 0.5103    |   |        | | | |
 | Dual Attention      | A+T        |         |           |   |        | | | |
 | Early Concatenation | V+T        |         |           |   |        | | | |
 | Early Concatenation | V+A        |         |           |   |        | | | |
 | Early Concatenation | A+T        |         |           |   |        | | | |
 | Late Weighting      | V+T        |         |           |   |        | | | |
 | Late Weighting      | V+A        |         |           |   |        | | | |
 | Late Weighting      | A+T        |         |           |   |        | | | |
 | LSTM + Attention    | V          |         |           |   |        | | | |  
 | LSTM + Attention    | T          |  0.6285 |           |   |        | | | |  
 | LSTM + Attention    | A          |         |           |   |        | | | |  
 | LSTM                | V          | 0.5170  | 0.5106    |   |        | | | |        
 | LSTM                | T          | 0.6026  | 0.6056    |   |        | | | |        
 | LSTM                | A          |         |           |   |        | | | |        
      
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
