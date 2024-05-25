import os
import time
import numpy as np
import glob
import rawpy
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
tf.compat.v1.disable_eager_execution()

# Set your input, gt, and checkpoint directories
input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './result_Sony/'
result_dir = './result_Sony/'

# Rest of the code remains the same
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps = 512  # patch size for training
save_freq = 500

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]

# Define the LeakyReLU activation function
def lrelu(x):
    return tf.maximum(x * 0.2, x)

# Define the upsampling and concatenation function
def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, output_shape=tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

# Define the generator network
def network(input):
    conv1 = layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(input)
    conv1 = layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2), padding='same')(conv3)

    conv4 = layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv4)
    pool4 = layers.MaxPooling2D((2, 2), padding='same')(conv4)

    conv5 = layers.Conv2D(512, (3, 3), activation=lrelu, padding='same')(pool4)
    conv5 = layers.Conv2D(512, (3, 3), activation=lrelu, padding='same')(conv5)

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(up6)
    conv6 = layers.Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv6)

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(up7)
    conv7 = layers.Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv7)

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(up8)
    conv8 = layers.Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv8)

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(up9)
    conv9 = layers.Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv9)

    conv10 = layers.Conv2D(12, (1, 1), activation=None)(conv9)
    out = tf.nn.depth_to_space(conv10, 2)
    return out

# Rest of the code remains the same
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

# Create a session and define placeholders
sess = tf.compat.v1.Session()
in_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

# Rest of the code remains the same

# Define the optimizer and loss function
G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
t_vars = tf.compat.v1.trainable_variables()
lr = tf.compat.v1.placeholder(tf.float32)
G_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)
saver = tf.compat.v1.train.Saver()

sess.run(tf.compat.v1.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images = [None] * len(train_ids)
input_images = {ratio: [None] * len(train_ids) for ratio in ['300', '250', '100']}
g_loss = np.zeros((5000, 1))

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
for epoch in range(lastepoch, 1):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        learning_rate = 1e-5

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.randint(0, len(in_files))]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[ind] = G_current

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            Image.fromarray((temp * 255).astype(np.uint8)).save(result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

    saver.save(sess, checkpoint_dir + 'model.ckpt')
    print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))
