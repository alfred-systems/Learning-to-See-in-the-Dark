# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import itertools
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob

input_dir = './dataset/LG2/'
checkpoint_dir = './checkpoint/Sony/'
result_dir = './result_LG/'



def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out


def pack_raw(raw, black_level=512, raw_pattern=[[0, 1], [3, 2]]):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    vmax = im.max()
    im = np.maximum(im - black_level, 0) / (16383 - black_level)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    channel_to_coord = {
        raw_pattern[y][x]: (y, x)
        for y, x in itertools.product(range(2), range(2))
    }
    color_maps = [None] * 4
    for k, v in channel_to_coord.items():
        color_maps[k] = im[v[0]:H:2, v[1]:W:2, :]

    out = np.concatenate(color_maps, axis=2)
    return out


with tf.Session() as sess:
    in_image = tf.placeholder(tf.float32, [None, None, None, 4])
    gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
    out_image = network(in_image)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    if not os.path.isdir(result_dir + 'final/'):
        os.makedirs(result_dir + 'final/')
    raw_files = glob.glob(os.path.join(input_dir, '*.dng'))

    for in_path in raw_files:
        in_fn = os.path.basename(in_path)
        print(in_fn)
        ratio = min(100, 300)

        raw = rawpy.imread(in_path)
        blevel = raw.black_level_per_channel
        blevel = int(sum(blevel) / len(blevel))
        raw_np = pack_raw(raw, black_level=blevel, raw_pattern=raw.raw_pattern)
        print('black level: ', blevel)
        print(raw.raw_pattern)

        pypost = raw.postprocess()
        output_size = [
            pypost.shape[1] // 2,
            pypost.shape[0] // 2,
        ]
        scipy.misc.toimage(
            pypost
        ).resize(
            output_size
        ).save(
            os.path.join(result_dir, in_fn.replace('.dng', '.rawpy.png'))
        )

        campost = raw.postprocess(use_camera_wb=True, half_size=False,
                                no_auto_bright=True, output_bps=8)
        scipy.misc.toimage(
            campost
        ).resize(
            output_size
        ).save(
            os.path.join(result_dir, in_fn.replace('.dng', '.cam.png'))
        )

        out_per_ratio = []
        for ratio in [100, 200, 300]:
            input_full = np.expand_dims(raw_np, axis=0) * ratio
            input_full = np.minimum(input_full, 1.0)

            output = sess.run(out_image, feed_dict={in_image: input_full})
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            output = np.flip(np.transpose(output, (1, 0, 2)), axis=1)

            scipy.misc.toimage(
                output * 255, high=255, low=0, cmin=0, cmax=255
            ).resize(
                output_size
            ).save(
                os.path.join(result_dir, in_fn.replace('.dng', f'.{ratio}.png'))
            )
            out_per_ratio.append((output * 255).astype(np.uint8))
            # scipy.misc.toimage(out_per_ratio[-1]).save(
            #     os.path.join(result_dir, 'test.png')
            # )
            # import pdb; pdb.set_trace()
        
        concated = np.concatenate([campost, pypost] + out_per_ratio, axis=1)
        concated2 = np.concatenate(out_per_ratio, axis=1)
        ccat_size = [concated.shape[1] // 4, concated.shape[0] // 4]
        scipy.misc.toimage(concated).resize(ccat_size).save(
            os.path.join(result_dir, in_fn.replace('.dng', f'.concat.png'))
        )
