
import tensorflow as tf
import tensorflow.contrib.slim as slim

import conv_blocks

def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(
        tf.truncated_normal(
            [pool_size, pool_size, output_channels, in_channels],
            stddev=0.02))
    deconv = tf.nn.conv2d_transpose(
        x1, deconv_filter, tf.shape(x2),
        strides=[1, pool_size, pool_size, 1]
    )

    deconv_output = tf.concat([deconv, x2], 3)
    # deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def upsample(x1, output_channels, in_channels):
    pool_size = 2
    x_shape = tf.shape(x1)
    output_shape = [
        x_shape[0],
        x_shape[1] * pool_size,
        x_shape[2] * pool_size,
        output_channels
    ]
    deconv_filter = tf.Variable(tf.truncated_normal(
        [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(
        x1, deconv_filter, output_shape,
        strides=[1, pool_size, pool_size, 1])
    return deconv


def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(
        pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(
        conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(
        pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(
        conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(
        pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(
        conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(
        conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(
        conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1,
                        activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(
        conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out


def lite_network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(
        pool2, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv3_1')
    conv3 = slim.conv2d(
        conv3, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(
        pool3, 256, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv4_1')
    conv4 = slim.conv2d(
        conv4, 256, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(
        pool4, 512, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv5_1')
    conv5 = slim.conv2d(
        conv5, 512, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv6_1')
    conv6 = slim.conv2d(
        conv6, 256, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv7_1')
    conv7 = slim.conv2d(
        conv7, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv9_2')

    conv10 = slim.conv2d(
        conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    return conv10


def lite_network_2(input, alpha=0.5):
    # raise NotImplementedError('Wrong implmentation')
    def c_mul(c, a): return max(int(c * a), 24)
    
    conv1 = slim.conv2d(input, c_mul(32, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, c_mul(32, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv1_2')
    pool1 = slim.conv2d(conv1, c_mul(32, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv2 = slim.conv2d(pool1, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv2_2')
    pool2 = slim.conv2d(conv2, c_mul(64, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv3 = slim.conv2d(
        pool2, c_mul(128, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv3_1')
    conv3 = slim.conv2d(
        conv3, c_mul(128, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv3_2')
    pool3 = slim.conv2d(conv3, c_mul(128, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv4 = slim.conv2d(
        pool3, c_mul(256, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv4_1')
    conv4 = slim.conv2d(
        conv4, c_mul(256, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv4_2')
    pool4 = slim.conv2d(conv4, c_mul(256, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv5 = slim.conv2d(
        pool4, c_mul(512, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv5_1')
    conv5 = slim.conv2d(
        conv5, c_mul(512, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv5_2')

    up6 = upsample_and_concat(
        conv5, conv4, c_mul(256, alpha), c_mul(512, alpha))
    conv6 = slim.conv2d(up6, c_mul(256, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv6_1')
    conv6 = slim.conv2d(
        conv6, c_mul(256, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv6_2')

    up7 = upsample_and_concat(
        conv6, conv3, c_mul(128, alpha), c_mul(256, alpha))
    conv7 = slim.conv2d(up7, c_mul(128, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv7_1')
    conv7 = slim.conv2d(
        conv7, c_mul(128, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv7_2')

    up8 = upsample_and_concat(
        conv7, conv2, c_mul(64, alpha), c_mul(128, alpha))
    conv8 = slim.conv2d(up8, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, c_mul(32, alpha), c_mul(64, alpha))
    conv9 = slim.conv2d(up9, c_mul(32, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, c_mul(32, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv9_2')

    conv10 = slim.conv2d(
        conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    return conv10


def lite_network_3(input, alpha=0.5, D2S=True):
    def c_mul(c, a): return int(c * a)
    conv0 = slim.conv2d(input, c_mul(32, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv0, c_mul(32, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv1_2')
    pool1 = slim.conv2d(conv1, c_mul(32, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv2 = slim.conv2d(pool1, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv2_2')
    pool2 = slim.conv2d(conv2, c_mul(64, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv3 = slim.conv2d(
        pool2, c_mul(128, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv3_1')
    conv3 = slim.conv2d(
        conv3, c_mul(128, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv3_2')
    pool3 = slim.conv2d(conv3, c_mul(128, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv4 = slim.conv2d(
        pool3, c_mul(256, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv4_1')
    conv4 = slim.conv2d(
        conv4, c_mul(256, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv4_2')
    pool4 = slim.conv2d(conv4, c_mul(256, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv5 = slim.conv2d(
        pool4, c_mul(512, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv5_1')
    conv5 = slim.conv2d(
        conv5, c_mul(512, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv5_2')

    up6 = upsample_and_concat(
        conv5, conv4, c_mul(256, alpha), c_mul(512, alpha))
    conv6 = slim.conv2d(up6, c_mul(256, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv6_1')
    conv6 = slim.conv2d(
        conv6, c_mul(256, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv6_2')

    up7 = upsample_and_concat(
        conv6, conv3, c_mul(128, alpha), c_mul(256, alpha))
    conv7 = slim.conv2d(up7, c_mul(128, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv7_1')
    conv7 = slim.conv2d(
        conv7, c_mul(128, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv7_2')

    up8 = upsample_and_concat(
        conv7, conv2, c_mul(64, alpha), c_mul(128, alpha))
    conv8 = slim.conv2d(up8, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, c_mul(32, alpha), c_mul(64, alpha))
    conv9 = slim.conv2d(up9, c_mul(32, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, c_mul(32, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv9_2')
    up10 = upsample(
        conv9, c_mul(32, alpha), c_mul(32, alpha))

    conv10 = slim.conv2d(
        up10, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    
    conv_list = [
        conv0,
        conv1,
        conv2,
        conv3,
        conv4,
        conv5,
        conv6,
        conv7,
        conv8,
        conv9,
        conv10,
    ]
    for i, c in enumerate(conv_list):
        print(i, c)
    if D2S:
        out = tf.depth_to_space(conv10, 2)
        return out
    else:
        return conv10


def lite_network_4(input, alpha=0.5, D2S=True):
    def c_mul(c, a): return max(int(c * a), 16)
    
    stem = slim.conv2d(input, c_mul(32, alpha), [3, 3], stride=1,
                        activation_fn=tf.nn.relu, scope='stem')
    conv0 = slim.conv2d(stem, c_mul(32, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv0, c_mul(32, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv1_2')
    pool1 = slim.conv2d(conv1, c_mul(32, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv2 = slim.conv2d(pool1, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv2_2')
    pool2 = slim.conv2d(conv2, c_mul(64, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv3 = slim.conv2d(
        pool2, c_mul(128, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv3_1')
    conv3 = slim.conv2d(
        conv3, c_mul(128, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv3_2')
    pool3 = slim.conv2d(conv3, c_mul(128, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv4 = slim.conv2d(
        pool3, c_mul(256, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv4_1')
    conv4 = slim.conv2d(
        conv4, c_mul(256, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv4_2')
    pool4 = slim.conv2d(conv4, c_mul(256, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu)

    conv5 = slim.conv2d(
        pool4, c_mul(512, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv5_1')
    conv5 = slim.conv2d(
        conv5, c_mul(512, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv5_2')

    up6 = upsample_and_concat(
        conv5, conv4, c_mul(256, alpha), c_mul(512, alpha))
    conv6 = slim.conv2d(up6, c_mul(256, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv6_1')
    conv6 = slim.conv2d(
        conv6, c_mul(256, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv6_2')

    up7 = upsample_and_concat(
        conv6, conv3, c_mul(128, alpha), c_mul(256, alpha))
    conv7 = slim.conv2d(up7, c_mul(128, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv7_1')
    conv7 = slim.conv2d(
        conv7, c_mul(128, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv7_2')

    up8 = upsample_and_concat(
        conv7, conv2, c_mul(64, alpha), c_mul(128, alpha))
    conv8 = slim.conv2d(up8, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, c_mul(64, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, c_mul(32, alpha), c_mul(64, alpha))
    conv9 = slim.conv2d(up9, c_mul(32, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, c_mul(32, alpha), [3, 3], rate=1,
                        activation_fn=tf.nn.relu, scope='g_conv9_2')
    
    up10 = upsample_and_concat(conv9, stem, c_mul(32, alpha), c_mul(32, alpha))
    conv10 = slim.conv2d(up10, c_mul(32, alpha), [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv10_1')
    conv10 = slim.conv2d(
        conv10, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10_2')
    
    conv_list = [
        stem,
        conv0,
        conv1,
        conv2,
        conv3,
        conv4,
        conv5,
        conv6,
        conv7,
        conv8,
        conv9,
        conv10,
    ]
    for i, c in enumerate(conv_list):
        print(i, c)
    if D2S:
        out = tf.depth_to_space(conv10, 2)
        return out
    else:
        return conv10


def mobile_v2_se(input, alpha=0.5, D2S=True):
    def c_mul(c, a): return int(c * a)
    block_kwargs = {
        'expansion_size': conv_blocks.expand_input_by_factor(6),
        'split_expansion': 1,
        'normalizer_fn': tf.contrib.layers.group_norm,
        'residual': True
    }

    conv0 = slim.conv2d(input, c_mul(32, alpha), [3, 3], stride=2,
                        activation_fn=tf.nn.relu, scope='g_conv1_1')
    block1 = conv_blocks.expanded_conv(
                conv0,
                num_outputs=16,
                expansion_size=conv_blocks.expand_input_by_factor(1, divisible_by=1))
    
    block2 = conv_blocks.expanded_conv(block1, stride=2, num_outputs=32, **block_kwargs)
    se2 = conv_blocks.squeeze_excite(block2)
    block3 = conv_blocks.expanded_conv(se2, stride=1, num_outputs=32, **block_kwargs)
    se3 = conv_blocks.squeeze_excite(block3)
    block4 = conv_blocks.expanded_conv(se3, stride=1, num_outputs=32, **block_kwargs)
    se4 = conv_blocks.squeeze_excite(block4)

    block5 = conv_blocks.expanded_conv(se4, stride=2, num_outputs=64, **block_kwargs)
    se5 = conv_blocks.squeeze_excite(block5)
    block6 = conv_blocks.expanded_conv(block5, stride=1, num_outputs=64, **block_kwargs)
    se6 = conv_blocks.squeeze_excite(block6)
    block7 = conv_blocks.expanded_conv(block6, stride=1, num_outputs=64, **block_kwargs)
    se7 = conv_blocks.squeeze_excite(block7)

    block8 = conv_blocks.expanded_conv(se7, stride=2, num_outputs=128, **block_kwargs)
    se8 = conv_blocks.squeeze_excite(block8)
    block9 = conv_blocks.expanded_conv(se8, stride=1, num_outputs=128, **block_kwargs)
    se9 = conv_blocks.squeeze_excite(block9)
    block10 = conv_blocks.expanded_conv(se9, stride=1, num_outputs=128, **block_kwargs)
    se10 = conv_blocks.squeeze_excite(block10)

    block11 = conv_blocks.expanded_conv(se10, stride=2, num_outputs=256, **block_kwargs)
    se11 = conv_blocks.squeeze_excite(block11)
    block12 = conv_blocks.expanded_conv(se11, stride=1, num_outputs=256, **block_kwargs)
    se12 = conv_blocks.squeeze_excite(block12)

    up1 = upsample_and_concat(se12, se10, 128, 256)
    de_block1 = conv_blocks.expanded_conv(up1, stride=1, num_outputs=128, **block_kwargs)
    de_se1 = conv_blocks.squeeze_excite(de_block1)
    de_block2 = conv_blocks.expanded_conv(de_se1, stride=1, num_outputs=128, **block_kwargs)
    de_se2 = conv_blocks.squeeze_excite(de_block2)

    up2 = upsample_and_concat(de_se2, se7, 64, 128)
    de_block3 = conv_blocks.expanded_conv(up2, stride=1, num_outputs=64, **block_kwargs)
    de_se3 = conv_blocks.squeeze_excite(de_block3)
    de_block4 = conv_blocks.expanded_conv(de_se3, stride=1, num_outputs=64, **block_kwargs)
    de_se4 = conv_blocks.squeeze_excite(de_block4)

    up3 = upsample_and_concat(de_se4, se4, 32, 64)
    de_block5 = conv_blocks.expanded_conv(up3, stride=1, num_outputs=32, **block_kwargs)
    de_se5 = conv_blocks.squeeze_excite(de_block5)
    de_block6 = conv_blocks.expanded_conv(de_se5, stride=1, num_outputs=32, **block_kwargs)
    de_se6 = conv_blocks.squeeze_excite(de_block6)

    up4 = upsample_and_concat(de_se6, block1, 16, 32)
    # de_block7 = conv_blocks.expanded_conv(up4, stride=1, num_outputs=16, **block_kwargs)
    up5 = upsample(up4, 12, 32)

    if D2S:
        out = tf.depth_to_space(up5, 2)
        return out
    else:
        return up5

