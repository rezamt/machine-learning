# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices,
# along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib: Visualization with Python https://matplotlib.org/
import matplotlib.pyplot as plt

from util import plot_image, plot_value_array

if __name__ == '__main__':
    # checking everything is okay ? Libraries are installed ?
    print(tf.__version__)

    '''
    Here, 
        60,000 images are used to train the network (Training Data Set) 
        and 10,000 images to evaluate how accurately the network learned to (Test Data Set) 
    classify images. You can access the Fashion MNIST directly from TensorFlow. 
    Import and load the Fashion MNIST data directly from TensorFlow:
    '''
    fashion_mnist = tf.keras.datasets.fashion_mnist

    '''
    Loading the dataset returns four NumPy arrays:

      - The train_images (Train Model) and train_labels arrays are the training set—the data the model uses to learn.
      - The model is tested against the test set, (Test the accuracy of train model) the test_images, and test_labels arrays.
        
        
    '''
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    '''
    The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. 
    The labels are an array of integers, ranging from 0 to 9. 
    These correspond to the class of clothing the image represents:
    

    Each image is mapped to a single label. Since the class names are not included with the dataset, 
    store them here to use later when plotting the images:
    
    T-shirt/top     0
    Trouser         1
    Pullover        2
    Dress           3
    
    '''
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    '''
    Let's explore the format of the dataset before training the model. The following shows there are 60,000 images in 
    the training set, with each image represented as 28 x 28 pixels:
    '''

    print("train_images.shape {} ".format(train_images.shape))
    print("labels in the training set: {}".format(len(train_labels)))
    print("Each label is an integer between 0 and 9: {}".format(train_labels))

    '''
    
    1 Preprocess the data
    
    The data must be preprocessed before training the network. If you inspect the first image in the training set, 
    you will see that the pixel values fall in the range of 0 to 255:
    
    '''
    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    '''
    1-1 Scaling 
        Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
        To do so, divide the values by 255. It's important that the training set and the testing set be preprocessed 
        in the same way:
        
        pixel 1 : 186  then 186 / 255 = 0.7 
        
    
    '''

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    '''
    To verify that the data is in the correct format and that you're ready to build and train the network, let's display 
    the first 25 images from the training set and display the class name below each image.
    '''

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()


    '''
    2 Build the model
        Model: Perceptron - Neural Network
        Layers : 3 Layers (1 Flatten+ 2x Dense)
        Building the neural network requires configuring the layers of the model, then compiling the model.
        
        The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. 
        
        Hopefully, these representations are meaningful for the problem at hand.
        
        Layer 1: 
          The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional 
            array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels).
            Think of this layer as unstacking rows of pixels in the image and lining them up. 
            This layer has no parameters to learn; it only reformats the data.
            
        Layer 2:
            After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. 
            These are densely connected, or fully connected, neural layers. 
                The first Dense layer has 128 nodes (or neurons). 
                The second (and last) layer returns a logits array with length of 10. Each node contains a score that
                    indicates the current image belongs to one of the 10 classes.
    '''

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    '''
      2.2 Compile the model
        Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
        
          - Loss function 
            This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
          - Optimizer
            This is how the model is updated based on the data it sees and its loss function.
          - Metrics
            Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images 
            that are correctly classified.
          
    '''
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    '''
      2-3 Train the model
      Training the neural network model requires the following steps:
      
          1- Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
          2- The model learns to associate images and labels.
          3- You ask the model to make predictions about a test set—in this example, the test_images array.
          4- Verify that the predictions match the labels from the test_labels array.
    
    '''
    # Feed the model
    # As the model trains, the loss and accuracy metrics are displayed.
    # This model reaches an accuracy of about 0.91 (or 91%) on the training data.
    model.fit(train_images, train_labels, epochs=10)



    # Evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    '''
    Looking at above results:
    It turns out that the accuracy on the test dataset is a little less than the accuracy on the training dataset. This gap between training accuracy and test accuracy represents overfitting. Overfitting happens when a machine learning model performs worse on new, previously unseen inputs than it does on the training data. An overfitted model "memorizes" the noise and details in the training dataset to a point where it negatively impacts the performance of the model on the new data. For more information, see the following:
        Demonstrate overfitting: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#demonstrate_overfitting
        Strategies to prevent overfitting:  https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#strategies_to_prevent_overfitting


        Now
    '''

    '''
    Make predictions
        With the model trained, you can use it to make predictions about some images. The model's linear outputs, logits. 
        Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
    '''
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    print("Prediction for the first test image: \n {}".format( predictions[0]))

    print("Label which has the highest confidence value: {}".format( np.argmax(predictions[0])))

    print("Graph this to look at the full set of 10 class predictions.")

    '''
    Verify predictions
    '''
    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()