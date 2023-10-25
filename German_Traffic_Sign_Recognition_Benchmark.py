import os 
import glob
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
from my_utils import split_data, order_test_set,create_generator
from deep_learning_model import streetsigns_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
#90% of the data will be classified for training and the rest is for validation 
'''1/split data into training and validation set'''
'''2/prepare the test set'''

if __name__=="__main__":
    if False:
        path_to_data="C:\\Users\\WhiteLuce SIS\\OneDrive\\deep learning\\computer vision\\German traffic signs\\Train"
        path_to_save_data="C:\\Users\WhiteLuce SIS\\OneDrive\\deep learning\\computer vision\\German traffic signs\\training_data\\train"
        path_to_save_val="C:\\Users\\WhiteLuce SIS\\OneDrive\\deep learning\\computer vision\\German traffic signs\\training_data\\validation"
        split_data(path_to_data,path_to_save_data,path_to_save_val)
    
    if False: 
        path_to_images="C:\\Users\\WhiteLuce SIS\\OneDrive\\deep learning\\computer vision\\German traffic signs\\Test"
        path_to_csv="C:\\Users\\WhiteLuce SIS\\OneDrive\\deep learning\\computer vision\\German traffic signs\\Test.csv"
        order_test_set(path_to_images,path_to_csv)
    
    path_to_train="C:\\Users\WhiteLuce SIS\\OneDrive\\deep learning\\computer vision\\German traffic signs\\training_data\\train"
    path_to_val="C:\\Users\\WhiteLuce SIS\\OneDrive\\deep learning\\computer vision\\German traffic signs\\training_data\\validation"
    path_to_test="C:\\Users\\WhiteLuce SIS\\OneDrive\\deep learning\\computer vision\\German traffic signs\\Test"
    #recommended batch_size is multiplied by 2
    #we can face an error f oom which is out of memory when we try to use batch_size=512 since we don't have the memory t load that amount of data at once
    batch_size=64
    epochs=5
    lr=0.0001
    train_generator,val_generator,test_generator=create_generator(batch_size,path_to_train,path_to_val,path_to_test)
    #we can  get the number of classes from the train_generator since it contains the most images 
    nbr_classes= train_generator.num_classes

    '''save the best model during training'''
    path_to_save_model='./Models'
    ckpt_saver= ModelCheckpoint(
        path_to_save_model,
        monitor="val_accuracy",
        #the saver here depending from the mode which is maximum, will register the model having an accuracy higheror greater than the previous one
        mode="max",
        #we want only one model to be saved
        save_best_only=True,
        #here we want to save only at the end of the epoch
        save_freq='epoch',
        #verbose =1 is only for program debbuging to see whether the model is been saved or not
        verbose=1

    )

    '''define earlystopping'''
    #if after 10 epochs my validation score doesn't get higher than the training will be stopped
    early_stop=EarlyStopping(monitor="val_accuracy",patience=10)

    '''compile the model'''
    model=streetsigns_model(nbr_classes)
    #we can use the optimizer by instanciating it
    #sometimes the adam algorithm fails to converge means that the loss function can't go any lower, amsgrad will be used in that case
    #in order to make the model converge
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001,amsgrad=True )
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    #since train generator contains all the data and knows the labels of each of the data, we don't need to specify y_train
    model.fit(train_generator,epochs=epochs,batch_size=batch_size,validation_data=val_generator,callbacks=[ckpt_saver,early_stop])





    
