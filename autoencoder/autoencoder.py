#import libraries
import tensorflow as tf
import numpy as np

# Some hyperparameters
latent_dim = 50
epochs = 100
latent_dim = 50
num_examples_to_generate = 16
TRAIN_BUF = 60000
TEST_BUF = 10000
BATCH_SIZE = 32

# Import and normalize datas
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

# Get dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)
# random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

# Create model
autoencoder = tf.keras.Sequential(
          [
              tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
              tf.keras.layers.Conv2D(
                  filters=32,
                  kernel_size=3,
                  strides=(2, 2),
                  activation='relu'),
              tf.keras.layers.Conv2D(
                  filters=64,
                  kernel_size=3,
                  strides=(2, 2),
                  activation='relu'),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(latent_dim),
              tf.keras.layers.Dense(7*7*32),
              tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
              tf.keras.layers.Conv2DTranspose(
                  filters=64,
                  kernel_size=3,
                  strides=(2, 2),
                  padding="SAME",
                  activation='relu'),
              tf.keras.layers.Conv2DTranspose(
                  filters=32,
                  kernel_size=3,
                  strides=(2, 2),
                  padding="SAME",
                  activation='relu'),
              # No activation
              tf.keras.layers.Conv2DTranspose(
                  filters=1,
                  kernel_size=3,
                  strides=(1, 1),
                  padding="SAME"),
          ]
)

if __name__ == '__main__':
    print(autoencoder.summary())
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            compute_apply_gradients(model, train_x, optimizer)
        end_time = time.time()

    if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, '
              'time elapse for current epoch {}'.format(epoch,
                                                        elbo,
                                                        end_time - start_time))
        generate_and_save_images(
            model, epoch, random_vector_for_generation)
