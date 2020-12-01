# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Computer Vision with CNNs
#
# For this exercise you will build a cats v dogs classifier
# using the Cats v Dogs dataset from TFDS.
# Be sure to use the final layer as shown 
#     (Dense, 2 neurons, softmax activation)
#
# The testing infrastructre will resize all images to 224x224 
# with 3 bytes of color depth. Make sure your input layer trains
# images to that specification, or the tests will fail.
#
# Make sure your output layer is exactly as specified here, or the 
# tests will fail.


import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


dataset_name = 'cats_vs_dogs'
train_dataset, info = tfds.load(name=dataset_name, split='train[:80%]', with_info=True)
valid_dataset, info = tfds.load(name=dataset_name, split='train[-20%:]', with_info=True)

def preprocess(features):
    # YOUR CODE HERE
    img, lbl = tf.cast(features['image'], tf.float32) / 255.0, features['label']
    img = tf.image.resize(img, size=(224, 224))
    return img, lbl

def solution_model():
    batch_size = 32
    train_data = train_dataset.map(preprocess).batch(batch_size)
    valid_data = valid_dataset.map(preprocess).batch(batch_size)

    print(train_data)

    total_image_size = info.splits['train'].num_examples
    steps_per_epoch= int(total_image_size * 0.8) // batch_size + 1
    validation_steps= int(total_image_size * 0.2) // batch_size + 1

    model = Sequential([
        Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.8),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax'),
    ])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                     factor=0.1,
                                                     patience=3, 
                                                     min_lr=0.00001)

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint_path = 'my_checkpoint.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path, 
                                 save_best_only=True, 
                                 save_weights_only=True, 
                                 monitor='val_loss', 
                                 verbose=1)
    
    model.fit(train_data, 
              validation_data=(valid_data),
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              epochs=40, 
              callbacks=[checkpoint, reduce_lr],
              )
    
    model.load_weights(checkpoint_path)

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
