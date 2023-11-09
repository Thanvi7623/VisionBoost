import os
import numpy as np
import tensorflow as tf
import rawpy
import imageio

class SeeInDark(tf.Module):
    def __init__(self, num_classes=10):
        super(SeeInDark, self).__init__()
# Set the paths to your directories and checkpoint
input_dir = './dataset/Sony/short/'
checkpoint_dir = './checkpoint/Sony/'


# Define a function for Leaky ReLU
def lrelu(x):
    return tf.maximum(x, 0.2 * x)

# Define a function for upsampling and concatenation
def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, output_shape=tf.shape(x2), strides=[1, pool_size, pool_size, 1])
    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])
    return deconv_output

# Define your neural network architecture
def network(input):
    # Define your network layers here
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(input)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(conv5)

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(up6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(conv6)

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(conv7)

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(up8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(conv8)

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(up9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=lrelu, padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = tf.keras.layers.Conv2D(12, (1, 1), activation=None)(conv9)
    out = tf.nn.depth_to_space(conv10, 2)
    return out

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

# Create a TensorFlow session
sess = tf.compat.v1.Session()

# Define placeholders for input and output images
in_image = tf.compat.v1.placeholder(tf.float32, [1, None, None, 4])
out_image = network(in_image)

# Create a Saver object to restore the model
saver = tf.compat.v1.train.Saver()
sess.run(tf.compat.v1.global_variables_initializer())

# Try to restore the model from the checkpoint
ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)

if ckpt and ckpt.model_checkpoint_path:
    print('Restoring checkpoint:', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print('No checkpoint found in the specified directory.')

# Replace 'input_image_path' with the path to your single input image
input_image_path = '/content/drive/MyDrive/Colab Notebooks/SID/dataset/Sony/short/10227_00_0.033s.ARW'

in_exposure = 100  # Set the desired exposure value
ratio = in_exposure  # You can adjust this ratio as needed

raw = rawpy.imread(input_image_path)
input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

output = sess.run(out_image, feed_dict={in_image: input_full})
output = np.minimum(np.maximum(output, 0), 1)

imageio.imsave( 'output.jpg', (output[0] * 255).astype(np.uint8))
