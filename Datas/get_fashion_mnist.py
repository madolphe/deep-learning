import tensorflow as tf
import matplotlib.pyplot as plt

# First let's try mnist dataset:
(fashion_train, _), (fashion_test, _) = tf.keras.datasets.fashion_mnist.load_data()
fashion_train = tf.data.Dataset.from_tensor_slices(fashion_test)
fashion_test = tf.data.Dataset.from_tensor_slices(fashion_test)


def augment(image):
    image = tf.expand_dims(image, -1)
    # image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    return image


if __name__ == '__main__':
    SHUFFLE_BUFFER_SIZE = 64
    BATCH_SIZE = 16
    fashion_train = fashion_train.shuffle(SHUFFLE_BUFFER_SIZE).map(augment)
    fashion_test = fashion_train.shuffle(SHUFFLE_BUFFER_SIZE).map(augment)
    batch = next(iter(fashion_train.batch(32)))
    print(batch.shape)
    plt.imshow(batch[0,:,:,0], cmap='gray')
    plt.show()
