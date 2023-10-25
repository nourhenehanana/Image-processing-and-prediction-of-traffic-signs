# Image-processing-and-prediction-of-traffic-signs

### Dataset Preparation

The dataset is meticulously organized with labeled images. In the root directory, there are two main folders: "train" and "test." Each of these folders contains subdirectories for different traffic signal categories. This structure facilitates seamless training and testing.

### Data Augmentation and Scaling

Data augmentation techniques have been successfully applied to the training set. Images in the "train" directory underwent transformations like shear, zoom, and horizontal flip using TensorFlow's ImageDataGenerator. Furthermore, all images were rescaled to a standard format between 0 and 1, contributing to the model's improved generalization.

### Model Construction

The Convolutional Neural Network (CNN) architecture is designed to capture intricate features for traffic signal recognition. It consists of convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification. The model's depth and width were adjusted based on the complexity of the dataset.

### Training the Model

The model was trained using the augmented and scaled training dataset. The choice of a binary crossentropy loss function and the Adam optimizer proved effective for this binary classification task. Training parameters, including the number of epochs and batch size, were fine-tuned to achieve optimal performance. Validation on the test set confirmed the model's ability to generalize.


