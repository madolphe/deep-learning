import tensorflow as tf
import matplotlib.pyplot as plt
import os

# First try, use the raw dataset
# Open the dataset directly within a dataset source:
path = os.path.join('..','..', 'Datas', 'victorian400', 'gray')
list_ds = tf.data.Dataset.list_files(os.path.join(path, '*'))


def parse_image(filename):
    parts = tf.strings.split(filename, '/')
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128, 128])
    return image


def show(image):
    plt.figure()
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.axis('off')
    plt.show()


# Second try, use tf_Record file to generate datasource:
def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary:
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)

    image = tf.image.decode_png(parsed_example['image_raw'], channels=1)
    # img_shape = tf.stack([parsed_example['height'], parsed_example['width'], parsed_example['depth']])
    return image


def preprocess(sample):
    return tf.image.resize(sample, [128, 128])/255


# Read the created tf.record:
raw_dataset = tf.data.TFRecordDataset(os.path.join(path, 'victorian_gray.images.tfrecords'))
# Let's map it with the _parse_image_function to parse tf.Examples:
dataset = raw_dataset.map(_parse_image_function).map(preprocess)


if __name__ == '__main__':
    # First method with the raw dataset:
    # We could iter through list_ds to get a file_image:
    file_image = next(iter(list_ds))
    # Then we read the file and parse it to tensor (and resize it to 128x128)
    image = parse_image(file_image)
    # Then show parsed image:
    show(image)

    # With an iterator let's try our dataset:
    iterator = iter(dataset.repeat().batch(32))
    sample = next(iterator)
    print(sample.shape)
    plt.imshow(sample[0, :, :, 0])
    plt.show()

