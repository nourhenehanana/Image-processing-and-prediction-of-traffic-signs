import tensorflow 
from tensorflow.keras.layers import Conv2D, Input,Dense,MaxPool2D,BatchNormalization, GlobalAvgPool2D,Flatten
from tensorflow.keras import Model

def streetsigns_model(nbr_classes):
    #hre we should note that our images contain rgb colors which mean that they're not gray scaled 
    #in this case they contain 3channel
    #the shape contains width, hight and number of channels|60 by 60 is great as a shape since most of all the images have smaller or greater to 60 as hight and width 
    #for the hight and weight we coudl have passed accross all the images and search for the average height and width 
    my_input=Input(shape=(60,60,3))
    x=Conv2D(32,(3,3),activation='relu')(my_input)
    x=Conv2D(64,(3,3),activation='relu')(x)
       
    x=MaxPool2D()(x)
        
    x=BatchNormalization()(x)
    x=Conv2D(128,(3,3),activation='relu')(x)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)
    
    #x=Flatten()(x)
    x=GlobalAvgPool2D()(x)
    x=Dense(64,activation='relu')(x)
    x=Dense(43,activation='softmax')(x) 
    #now to compile the model
    return Model(inputs=my_input, outputs=x)




   