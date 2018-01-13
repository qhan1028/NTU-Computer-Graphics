#
#   Fully Convolutional Networks -- for Background Replacer
#   Written by Qhan
#

from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import os.path as osp
import argparse
import time

import tensorflow_utils as utils
import utility as mu
from reader import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

class FCN:
    
    def __init__(self, hw, base_dir='fcn/', logs_dir='logs/', res_dir='res/', model_dir='vgg19/', filters=7):
        self.height, self.width = hw
        self.num_classes = 2
        self.f = filters
        self.graph = tf.Graph()
        self.base_dir = base_dir
        self.logs_dir = self.base_dir + logs_dir
        self.res_dir = self.base_dir + res_dir
        self.model_dir = self.base_dir + model_dir
        self.construct_model()

    def vgg_net(self, weights, image):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3'
        )

        net = {}
        current = image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + '_w')
                bias = utils.get_variable(bias.reshape(-1), name=name + '_b')
                current = utils.conv2d_basic(current, kernels, bias)
                #print('conv ' + name[4:] + ':', current.shape)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = utils.avg_pool_2x2(current)
                #print('pool ' + name[4:] + ':', current.shape)
            net[name] = current

        return net

    def inference(self, image, keep_prob):
        '''
        Semantic segmentation network definition
        :param image: input image. Should have values in range 0-255
        :param keep_prob:
        :return:
        '''
        model_data = utils.get_model_data(self.model_dir, MODEL_URL)

        mean = model_data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))

        weights = np.squeeze(model_data['layers'])

        processed_image = utils.process_image(image, mean_pixel)

        with tf.variable_scope('inference'):
            print('> [FCN] Setup vgg initialized conv layers... ', end=''); s = time.time()
            image_net = self.vgg_net(weights, processed_image)
            e = time.time(); print('%.4f ms' % ((e-s) * 1000))

            conv_final_layer = image_net['conv5_3']
            #print('----------------------------------------------------')

            print('> [FCN] Setup deconv layers... ', end=''); s = time.time()
            pool5 = utils.max_pool_2x2(conv_final_layer)
            #print('pool 5:', pool5.get_shape())

            W6 = utils.weight_variable([self.f, self.f, 512, 4096], name='W6')
            b6 = utils.bias_variable([4096], name='b6')
            conv6 = utils.conv2d_basic(pool5, W6, b6)
            #print('conv 6:', conv6.get_shape())
            relu6 = tf.nn.relu(conv6, name='relu6')
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

            W7 = utils.weight_variable([1, 1, 4096, 4096], name='W7')
            b7 = utils.bias_variable([4096], name='b7')
            conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
            #print('conv 7:', conv7.get_shape())
            relu7 = tf.nn.relu(conv7, name='relu7')
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

            W8 = utils.weight_variable([1, 1, 4096, self.num_classes], name='W8')
            b8 = utils.bias_variable([self.num_classes], name='b8')
            conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
            #print('conv 8:', conv8.get_shape())
            # annotation_pred1 = tf.argmax(conv8, dimension=3, name='prediction1')

            # now to upscale to actual image size
            deconv_shape1 = image_net['pool4'].get_shape()
            W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, self.num_classes], name='W_t1')
            b_t1 = utils.bias_variable([deconv_shape1[3].value], name='b_t1')
            conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net['pool4']))
            #print('conv t1:', conv_t1.get_shape())
            fuse_1 = tf.add(conv_t1, image_net['pool4'], name='fuse_1')
            #print('fuse 1:', fuse_1.get_shape())

            deconv_shape2 = image_net['pool3'].get_shape()
            W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name='W_t2')
            b_t2 = utils.bias_variable([deconv_shape2[3].value], name='b_t2')
            conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net['pool3']))
            #print('conv t2:', conv_t2.get_shape())
            fuse_2 = tf.add(conv_t2, image_net['pool3'], name='fuse_2')
            #print('fuse 2:', fuse_2.get_shape())

            shape = tf.shape(image)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.num_classes])
            W_t3 = utils.weight_variable([16, 16, self.num_classes, deconv_shape2[3].value], name='W_t3')
            b_t3 = utils.bias_variable([self.num_classes], name='b_t3')
            conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
            #print('conv t3:', conv_t3.get_shape())

            annotation_pred = tf.argmax(conv_t3, dimension=3, name='prediction')
            #print('prediction:', annotation_pred.get_shape())
            e = time.time(); print('%.4f ms' % ((e-s) * 1000))
        
        return tf.expand_dims(annotation_pred, dim=3), conv_t3

    def construct_model(self):
        with self.graph.as_default():
            keep_probability = tf.placeholder(tf.float32, name='keep_probabilty')
            image = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 3], name='input_image')
            pred_annotation, logits = self.inference(image, keep_probability)
            softmax = tf.nn.softmax(logits)

            print('> [FCN] Create session and saver... ', end=''); s = time.time()
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
            sess = tf.Session( graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options) )
            saver = tf.train.Saver()
            e = time.time(); print('%.4f ms' % ((e-s) * 1000))

            print('> [FCN] Restore model... ', end=''); s = time.time()
            ckpt = tf.train.get_checkpoint_state(self.logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                e = time.time(); print(ckpt.model_checkpoint_path + ', %.4f ms' % ((e - s) * 1000))
            else:
                print('> [FCN] Model not found.')
            
            self.session = sess
            self.predict_op = pred_annotation
            self.softmax_op = softmax
            self.image_dict = image
            self.dropout_dict = keep_probability

    def get_amap(self, im):
        with self.graph.as_default(), self.session.as_default():
            im = im.reshape( (1,) + im.shape )
            feed_dict = {self.image_dict: im, self.dropout_dict: 0.9}
            sft = self.session.run(self.softmax_op, feed_dict=feed_dict)
            res = sft[0, :, :, 1]
        return res

    def amap_sigmoid(self, amap, dx=2, sx=0):
        sigmoid = lambda x: 1. / (1. + np.exp(-x))
        amap_transform = (amap * 2 - 1) * dx + sx
        amap_sigmoid = sigmoid(amap_transform)
        return amap_sigmoid

    def list_variables(self):
        with self.graph.as_default(), self.session.as_default():
            print('[FCN] All variables')
            print(tf.global_variables())


def merge_result(images):
    merged = images[0]
    for im in images[1:]: merged = np.concatenate((merged, im), axis=1)
    return merged

# example programs for FCN
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', metavar='TAG', default='224h_3', help='tag')
    parser.add_argument('-i', metavar='DIR', default='images/', help='input image directory')
    parser.add_argument('-height', metavar='N', default=224, type=int, help='FCN height')
    parser.add_argument('-width', metavar='N', default=224, type=int, help='FCN width')
    parser.add_argument('-f', metavar='N', default=3, type=int, help='FCN core filters')
    parser.add_argument('-bg', metavar='FILE', default='bg/Hawaii.jpg', help='background target')
    args = parser.parse_args()

    print('====================================================')
    # create dirs
    tag = str(args.height) + 'h_' + str(args.f)
    logs_dir = 'logs_' + tag + '/'
    res_dir = 'res_' + tag + '/'
    dirpath = args.i
    if not osp.exists(res_dir): os.mkdir(res_dir)
    
    # create tensorflow session and prediction operation
    fcn = FCN((args.height, args.width), base_dir='', logs_dir=logs_dir, filters=args.f)

    # read input images, original (h, w)
    print('====================================================')
    images, names = read_data(dirpath)

    # load background
    bg_image = mu.bgr2rgb(cv2.imread(args.bg))

    # matting
    print('====================================================')
    for i, (name, image) in enumerate(zip(names, images)):
        
        print('image:', name, end='...', flush=True)
        fg = image
        bg = cv2.resize(bg_image, fg.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

        # get alpha map
        amap = fcn.get_amap(cv2.resize(fg, (fcn.width, fcn.height), interpolation=cv2.INTER_NEAREST))
        amap = cv2.resize(amap, fg.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        amap = mu.blur(amap, mask=7)
        amap = fcn.amap_sigmoid(amap, dx=10, sx=-2)
        
        res_rgb = mu.blend(fg, bg, amap)
        res = mu.rgb2bgr(res_rgb)
        
        cv2.imwrite(res_dir + name + '_res.png', merge_result( [mu.rgb2bgr(fg), res, mu.to3dim(mu.amap2im(amap))] ) )
        print('saved')
