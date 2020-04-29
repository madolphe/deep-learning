# @TODO: add argument parsers

import tensorflow as tf
from GAN.DCGAN.model_DCGAN_fashion import generative_net, discriminative_net, discriminator_optimizer, \
     generator_optimizer, dcgan_train_step
from Datas.get_fashion_mnist import fashion_train, augment
import time
import matplotlib.pyplot as plt
import os

# Let's define some parameters to train our model:
EPOCHS = 50
num_examples_to_generate = 16
noise_dim = 100
BATCH_SIZE = 12
SHUFFLE_BUFFER_SIZE = 64

# Datas:
fashion_train = fashion_train.shuffle(SHUFFLE_BUFFER_SIZE).map(augment)
fashion_test = fashion_train.shuffle(SHUFFLE_BUFFER_SIZE).map(augment)

# Let's define a checkpoint object to save our model:
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generative_net=generative_net,
                                 discriminative_net=discriminative_net)
checkpoint_path = 'training_checkpoints'


# Let's define a fixed seed to follow the progress of our model:
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def train(dataset, epochs, batch_size):
    for epoch in range(epochs):
        start = time.time()
        i = 0
        for image_batch in dataset.batch(batch_size):
            i += 1
            dcgan_train_step(image_batch, batch_size, noise_dim, epoch)
            # print("epoch: {}, batch num: {}".format(epoch, i))
        if (epoch + 1) % 20 == 0:
            # Produce images for the GIF as we go
            generate_and_save_images(generative_net, epoch + 1, seed)
        # Save the model every 15 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix=checkpoint_path)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generative_net, epochs, seed)


def generate_and_save_images(generator, epoch, seed, show=False):
    images = generator(seed, training=False)
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join('training_images', 'image_at_epoch_{:04d}.png'.format(epoch)))
    if show:
        plt.show()


if __name__ == '__main__':
    # generate_and_save_images(generative_net, 0, seed, show=True)
    train(fashion_train, 150, 32)
