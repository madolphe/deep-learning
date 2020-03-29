import tensorflow as tf
import glob as glob
import os


# Let's build a record file !

# The following functions can be used to convert a value to a type compatible
# with tf.Example (Functions taken from tensorflow tutorials)
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features that may be relevant
def image_to_example(image_string):
    image_shape = tf.image.decode_png(image_string).shape
    feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tf_record(files):
    with tf.io.TFRecordWriter(record_file) as writer:
        for filename in files:
            image_string = open(filename, 'rb').read()
            tf_example = image_to_example(image_string)
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    path = 'victorian400/gray/'
    # Write the raw image files to `images.tfrecords`.
    # First, process images into `tf.Example` messages.
    # Then, write to a `.tfrecords` file.
    record_file = 'victorian_gray.images.tfrecords'
    files = glob(os.path.join(path, '*'))
    create_tf_record(files)
