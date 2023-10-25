import matplotlib.pyplot as plt
import numpy as np 
import os
from sklearn.model_selection import train_test_split
import shutil
import glob
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def display_some_examples(examples,labels): 
    plt.figure(figsize=(10,10))
    for i in range(25):
        #25 is the number of images that we will take from the datase"t
        #what we will do is that we will choose a random image from the dataset
        idx= np.random.randint(0,examples.shape[0]-1)
        img=examples[idx]
        label=labels[idx]
        plt.subplot(5,5,i+1)
        plt.title(str(label))
        #we can add space with tight_layout so that we see the labels and the images in a more understandable way
        plt.tight_layout()
        #in order to tell matplotlib the images are gray we should add cmap attribute
        plt.imshow(img,cmap='gray')
    plt.show()

def split_data(path_to_data,path_to_save_train,path_to_save_val,split_size=0.1):
    folders=os.listdir(path_to_data)
    for folder in folders: 
        #here we want to have the path to data joined to the folder that contain the differen signs  (from 0 to 42 )
        full_path=os.path.join(path_to_data,folder)
        #glob is a method that will allow me to look insde a folder and load all of the files that exist in that folder depending on the extension we choose
        #here we're telling the glob module to give us a list of all the files that have extension of png inside the full path
        images_paths=glob.glob(os.path.join(full_path,'*.png'))

        x_train,x_val=train_test_split(images_paths,test_size=split_size)
        '''train_set'''
        for x in x_train:
            # x is the fullpath and what we want to do is to get the name of the image 

            #basename=os.path.basename(x)
            '''we don't need the above instruction since we want to construct the folder with the same name as the image '''
            #now we will reconstruct the file by joining it into path to save train
            path_to_folder=os.path.join(path_to_save_train,folder )

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            #shutil contain several functions one of them is a function "copy" that will allow us to copy our image and put it inside a new 
            #directory which is the one that we constructed it earlier 
            shutil.copy(x, path_to_folder)
        '''val_set'''
        for x in x_val:
            path_to_folder=os.path.join(path_to_save_val,folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            shutil.copy(x,path_to_folder)

def order_test_set(path_to_images,path_to_csv):
    #what we want here is to basically construct a dictionary tha has the name of the image as the key and the corresponding label as the value
    #the dictionary is not really necessary, in fact we want it only ot keep track of the dataset
    #the dictionary contains the paths and their corresponding labels
    #testset={}
    try: 
        with open(path_to_csv,'r') as csvfile:
            reader=csv.reader(csvfile,delimiter=',')
            for i,row in enumerate(reader):
                if i==0:
                    continue
                #row takes the element delimited by , 
                img_name=row[-1].replace('Test/','')
                label=row[-2]

                path_to_folder=os.path.join(path_to_images,label)
                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder) 
                img_full_path=os.path.join(path_to_images,img_name)
                shutil.move(img_full_path,path_to_folder)
    except: 
        print('[INFO]: error reading csv file')
    
def create_generator(batch_size,train_data_path,val_data_path,test_data_path): 
    #the objective of the preprocesssor is to preprocess the data befor that they will be fed to our deep learning model
    train_preprocessor= ImageDataGenerator(
        #divide the pixels by 255
        rescale= 1/255.,
        #use of data augmentation techniques
        #here we can use rotation to rotate images from -10 to 10 degrees
        rotation_range=10,
        #here we want our images to be shifted 10 percent to the left and 10 percent to the right
        width_shift_range=0.1
    )
    test_preprocessor= ImageDataGenerator(
        resale=1/255.
    )

    #flow from directory: will consider all the images inside the same folder belong to one specific class
    train_generator=train_preprocessor.flow_from_directory(
        train_data_path,
        #when we are compiling our model we have to choose categorical cross entropy 
        class_mode="categorical",
        #we want all the images to have the same zie which is 60*60
        target_size=(60,60),
        #our images are colorful
        color_mode='rgb',
        #shuffle equal to True: in each batch in each epoch the images will be shuffled, this mean that the order of images willnot be the same 
        #between two epochs
        #randomness ensure robustness to our models and let them be more generalized and learn features and ignore the order'''
        shuffle=True,
        batch_size=batch_size
    )
    val_generator=test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )
    test_generator=test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )
    #we set the shuffle false for both test and val generator since that shuffling data doesn't matter
    return train_generator,val_generator,test_generator


