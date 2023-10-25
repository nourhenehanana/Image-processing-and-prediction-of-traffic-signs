import tensorflow as tf
import numpy as np 
from deep_learning_model import streetsigns_model


def predict_with_model(model,img_path):
    #this method will take the image from the image path and use the model to predict the result
    '''1 st approach usin tensorflow'''
    image=tf.io.read_file(img_path)
    #since the image is in rgb so it will take into consideration 3 channels
    image=tf.image.decode_png(image,channels=3)
    #rescale the image 0 and 1 to 0 and 255| and convert pixels from int into 32floattype
    image=tf.image.convert_image_dtype(image,dtype=tf.float32)
    #we should keep on the same size of image that we defined in our model which is 60 by 60 in our case
    image=tf.image.resize(image,[60,60]) #(60,60,3)
    #expand the dimensions of our tensor to say that it's one image
    image=tf.expand_dims(image,axis=0) #(1,60,60,3) since that wht the layer expect as we've seen while makeing a model summary (None,..,..,..)
    #here the predictions take a list of probabilities to define a threshold that an image belongs to a particular class [0,005,0,0003,0.99,0.0001,..]
    #so below the thresholds, we can conclude that the image mostly belongs to class 2
    predictions=model.predict(image)
    #next we want to take only the best probability
    #argmax will give me exactly the index of the best value
    prediction=np.argmax(predictions) #2
    return predictions


#load the model and predict on new set of images
if __name__=="__main__":
    img_path="C:\\Users\\WhiteLuce SIS\\OneDrive\\deep learning\\computer vision\\German traffic signs\\Test\\2\\00409.png"
    model=tf.keras.models.load_model('./German traffic signs')
    prediction=predict_with_model(model,img_path)
    print ("prediction={prediction}")