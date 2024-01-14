import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

#convert string to ineteger
#ord function gives us the unicode representation

def int_rep(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord('0')
    return n

# making folder 

split_names=['train','test']
label_names=['angry' , 'disgusted' , 'fearful' , 'happy' , 'sad' , 'surprised' , 'neutral']
os.makedirs('data' , exist_ok=True)
for split_name in split_names:
    os.makedirs(os.path.join('data' , split_name), exist_ok=True)
    for label_name in label_names:
        os.makedirs(os.path.join('data' , split_name , label_name), exist_ok=True)
        
# To keep count of each category
angry=0
disgusted=0
fearful=0
happy=0
sad=0
surprised=0
neutral=0
angry_test=0
disgusted_test=0
fearful_test=0
happy_test=0
sad_test=0
surprised_test=0
neutral_test=0

df=pd.read_csv('./fer2013.csv')
mat=np.zeros((48,48),dtype=np.uint8)
print('saving images ...')

#reading the csv file line by line

for i in tqdm(range(len(df))): #this is for making a progress bar
    txt=df['pixels'][i]
    words= txt.split()

    #for making each content in the pixle column of dataset to the size of 48*48,
    #// is the maghsum alaih and will be zero until the j reaches to 48 but the % will 
    #become zero when j reaches 48 so it is like we are making the grid and then we put 
    # the integer value of the pixles in it one by one

    for j in range(2304):
        row_res=j//48
        column_res=j%48
        mat[row_res][column_res]=int_rep(words[j])
    #convertimg a numpy array to an image
    img=Image.fromarray(mat)
    
   #train part
    if i<28709:
        
        if df['emotion'][i]==0:
           img.save('./data/train/angry/im'+str(angry)+'.png')
           angry += 1
        elif df['emotion'][i]==1:
            img.save('./data/train/disgusted/im'+str(disgusted)+'.png')
            disgusted +=1
        elif df['emotion'][i]==2:
            img.save('./data/train/fearful/im'+str(fearful)+'.png')
            fearful +=1
        elif df['emotion'][i]==3:
             img.save=('./data/train/happy/im'+str(happy)+'.png')
             happy +=1
        elif df['emotion'][i]==4:
            img.save('./data/train/sad/im'+str(sad)+'.png')
            sad +=1
        elif df['emotion'][i]==5:
            img.save('./data/train/surprised/im'+str(surprised)+'.png')
            surprised +=1
        elif df['emotion'][i]==6:
            img.save('./data/train/neutral/im'+str(neutral)+'.png')
            neutral +=1
    #test part        
    else:
        
        if df['emotion'][i]==0:
           img.save('./data/test/angry/im'+str(angry_test)+'.png')
           angry_test += 1
        elif df['emotion'][i]==1:
            img.save('./data/test/disgusted/im'+str(disgusted_test)+'.png')
            disgusted_test +=1
        elif df['emotion'][i]==2:
            img.save('./data/test/fearful/im'+str(fearful_test)+'.png')
            fearful_test +=1
        elif df['emotion'][i]==3:
             img.save=('./data/test/happy/im'+str(happy_test)+'.png')
             happy_test +=1
        elif df['emotion'][i]==4:
            img.save('./data/test/sad/im'+str(sad_test)+'.png')
            sad_test +=1
        elif df['emotion'][i]==5:
            img.save('./data/test/surprised/im'+str(surprised_test)+'.png')
            surprised_test +=1
        elif df['emotion'][i]==6:
            img.save('./data/test/neutral/im'+str(neutral_test)+'.png')
            neutral_test +=1
            
print('Done')
        
        
            
         
        
            
        
                    
        
    