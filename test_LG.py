# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import itertools
import glob
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
from skimage.transform import resize as ski_resize
from networks import *

input_dir = './dataset/LG/'
checkpoint_dir = './result_lite_v2/'
result_dir = './result_lite_v2/'


def pack_raw(raw, black_level=512, raw_pattern=[[0, 1], [3, 2]], resize=None):
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
    if resize:
        h, w, c = out.shape
        nh, nw = int(resize * h), int(resize * w)
        nout = ski_resize(out, [nh, nw], preserve_range=True)
        assert nout.dtype == out.dtype
        out = nout
    return out


def test():
    with tf.Session() as sess:
        in_image = tf.placeholder(tf.float32, [None, None, None, 4])
        gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
        out_image = lite_network_4(in_image)

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
            raw_np = pack_raw(raw, black_level=blevel, raw_pattern=raw.raw_pattern, resize=0.125)
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
            # concated = np.concatenate([campost, pypost] + out_per_ratio, axis=1)
            concated = np.concatenate(out_per_ratio, axis=1)
            print('Concated result image size: ', concated.shape)

            ccat_size = [concated.shape[1], concated.shape[0]]
            # ccat_size = [concated.shape[1] // 4, concated.shape[0] // 4]
            scipy.misc.toimage(concated).resize(ccat_size).save(
                os.path.join(result_dir, in_fn.replace('.dng', f'.concat.png'))
            )


def convert_tflite():
    input_shape = [1, 2400, 3200, 1]

    def preprocess():
        raw_image = tf.placeholder(tf.int16, input_shape)
        black_level = tf.placeholder(tf.int16, shape=[])
        peak_value = tf.placeholder(tf.int16, shape=[])
        
        fp_black_level = tf.cast(black_level, tf.float32)
        fp_peak_value = tf.cast(peak_value, tf.float32)

        x = tf.cast(raw_image, tf.float32) + 2**15
        x = tf.nn.space_to_depth(x, 2)
        x = tf.image.resize_nearest_neighbor(x, [240, 320])
        x -= fp_black_level
        x /= fp_peak_value - fp_black_level
        return [raw_image, black_level, peak_value], x

    with tf.Session() as sess:
        # input_shape = [1, 1200, 1600, 4]
        
        input_tensors, norm_tenosr = preprocess()
        out_image = lite_network_2(norm_tenosr, alpha=0.5)

        # tf.contrib.quantize.create_eval_graph()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(out_image, feed_dict={
            input_tensors[0]: np.zeros(input_shape, dtype=np.float32),
            input_tensors[1]: 512.0,
            input_tensors[2]: 16383.0,
        })
        # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # if ckpt:
        #     print('loaded ' + ckpt.model_checkpoint_path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        
        converter_preprocess = tf.lite.TFLiteConverter.from_session(
            sess, input_tensors, [norm_tenosr])
        converter_preprocess.inference_input_type = tf.int16
        converter_preprocess.inference_type = tf.float32
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_output_type = tf.uint8
        tflite_pb = converter_preprocess.convert()
        with open('preprocessor.tflite', mode='wb') as pbfile:
            pbfile.write(tflite_pb)

        converter_model = tf.lite.TFLiteConverter.from_session(
            sess, [norm_tenosr], [out_image])
        tflite_pb = converter_model.convert()
        with open('unet.tflite', mode='wb') as pbfile:
            pbfile.write(tflite_pb)
        
        # from tensorflow.python.framework import graph_util

        # cons_graph = sess.graph_def
        # cons_graph = graph_util.remove_training_nodes(cons_graph)
        # cons_graph = graph_util.convert_variables_to_constants(
        #     sess, cons_graph, [out_image.name.replace(':0', '')])
        # tflite_pb = tf.lite.toco_convert(
        #     cons_graph, [in_image], [out_image],
        #     allow_custom_ops=True, inference_type=tf.uint8,
        #     quantized_input_stats={0: (0, 2)},
        #     default_ranges_stats=(-6, 6)
        # )
        


if __name__ == "__main__":
    # test()
    convert_tflite()
