from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import json
import random
import collections
import math
import time
import cv2

class pix2pixClass:
    def __init__(self):
        self.input_dir = ''  # 변수받는곳
        self.mode = 'test'
        self.output_dir = ''
        self.seed = 0
        self.checkpoint = ''
        self.max_steps = None
        self.max_epochs = 200
        self.summary_freq = 100
        self.summary_freq = 50
        self.trace_freq = 0
        self.separable_conv = False
        self.aspect_ratio = 1.0
        self.lab_colorization = False
        self.batch_size = 1
        self.which_direction = "AtoB"
        self.ngf = 64
        self.ndf = 64
        self.scale_size = 286
        self.flip = False
        self.lr = 0.0002
        self.beta1 = 0.5
        self.l1_weight = 100.0
        self.gan_weight = 1.0
        self.EPS = 1e-12
        self.CROP_SIZE = 256
        self.concat_dir = ''

        self.Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
        self.Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, "
                                                     "discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, "
                                                     "gen_grads_and_vars, train")

    def preprocess(self, image):
        with tf.name_scope("preprocess"):
            # [0, 1] => [-1, 1]
            return image * 2 - 1

    def deprocess(self, image):
        with tf.name_scope("deprocess"):
            # [-1, 1] => [0, 1]
            return (image + 1) / 2

    def discrim_conv(self, batch_input, out_channels, stride):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                                kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def gen_conv(self, batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        if self.separable_conv:
            return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                              depthwise_initializer=initializer, pointwise_initializer=initializer)
        else:
            return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                    kernel_initializer=initializer)

    def gen_deconv(self, batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        if self.separable_conv:
            _b, h, w, _c = batch_input.shape
            resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same",
                                              depthwise_initializer=initializer, pointwise_initializer=initializer)
        else:
            return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                              kernel_initializer=initializer)

    def lrelu(self, x, a):
        with tf.name_scope("lrelu"):
            # adding these together creates the leak part and linear part
            # then cancels them out by subtracting/adding an absolute value term
            # leak: a*x/2 - a*abs(x)/2
            # linear: x/2 + abs(x)/2

            # this block looks like it has 2 inputs on the graph unless we do this
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def batchnorm(self, inputs):
        return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                             gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

    def check_image(self, image):
        assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
        with tf.control_dependencies([assertion]):
            image = tf.identity(image)

        if image.get_shape().ndims not in (3, 4):
            raise ValueError("image must be either 3 or 4 dimensions")

        # make the last dimension 3 so that you can unstack the colors
        shape = list(image.get_shape())
        shape[-1] = 3
        image.set_shape(shape)
        return image

    def rgb_to_lab(self, srgb):
        with tf.name_scope("rgb_to_lab"):
            srgb = self.check_image(srgb)
            srgb_pixels = tf.reshape(srgb, [-1, 3])

            with tf.name_scope("srgb_to_xyz"):
                linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
                exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
                rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
                            ((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
                rgb_to_xyz = tf.constant([
                    #    X        Y          Z
                    [0.412453, 0.212671, 0.019334],  # R
                    [0.357580, 0.715160, 0.119193],  # G
                    [0.180423, 0.072169, 0.950227],  # B
                ])
                xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

            # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
            with tf.name_scope("xyz_to_cielab"):
                # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

                # normalize for D65 white point
                xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

                epsilon = 6 / 29
                linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
                exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
                fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                            xyz_normalized_pixels ** (1 / 3)) * exponential_mask

                # convert to lab
                fxfyfz_to_lab = tf.constant([
                    #  l       a       b
                    [0.0, 500.0, 0.0],  # fx
                    [116.0, -500.0, 200.0],  # fy
                    [0.0, 0.0, -200.0],  # fz
                ])
                lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

            return tf.reshape(lab_pixels, tf.shape(srgb))

    def lab_to_rgb(self, lab):
        with tf.name_scope("lab_to_rgb"):
            lab = self.check_image(lab)
            lab_pixels = tf.reshape(lab, [-1, 3])

            # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
            with tf.name_scope("cielab_to_xyz"):
                # convert to fxfyfz
                lab_to_fxfyfz = tf.constant([
                    #   fx      fy        fz
                    [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                    [1 / 500.0, 0.0, 0.0],  # a
                    [0.0, 0.0, -1 / 200.0],  # b
                ])
                fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

                # convert to xyz
                epsilon = 6 / 29
                linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
                exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
                xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (
                            fxfyfz_pixels ** 3) * exponential_mask

                # denormalize for D65 white point
                xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

            with tf.name_scope("xyz_to_srgb"):
                xyz_to_rgb = tf.constant([
                    #     r           g          b
                    [3.2404542, -0.9692660, 0.0556434],  # x
                    [-1.5371385, 1.8760108, -0.2040259],  # y
                    [-0.4985314, 0.0415560, 1.0572252],  # z
                ])
                rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
                # avoid a slightly negative number messing up the conversion
                rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
                linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
                exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
                srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
                            (rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

            return tf.reshape(srgb_pixels, tf.shape(lab))

    def load_examples(self):
        if self.input_dir is None or not os.path.exists(self.input_dir):
            raise Exception("input_dir does not exist")

        input_img = cv2.imread(os.path.join(self.input_dir))
        input_img = cv2.resize(input_img, dsize=(256, 256))
        black_img = np.zeros((256, 256, 3), np.uint8)

        concat_img = cv2.hconcat([input_img, black_img])
        self.concat_dir = os.path.join(os.path.dirname(self.input_dir), os.path.splitext(os.path.basename(self.input_dir))[0]
                                  + '-concat' + os.path.splitext(os.path.basename(self.input_dir))[1])

        cv2.imwrite(self.concat_dir, concat_img)  # 옆에 검은색 붙이고 덮어서 저장하기

        input_paths = list()
        input_paths.append(self.concat_dir)  # 추후 파일명 변경

        if os.path.splitext(self.input_dir)[1] == ".jpg":
            decode = tf.image.decode_jpeg

        elif os.path.splitext(self.input_dir)[1] == ".png":
            decode = tf.image.decode_png

        if len(input_paths) == 0:
            raise Exception("input_dir contains no image files")

        def get_name(path):
            name, _ = os.path.splitext(os.path.basename(path))
            return name

        # if the image names are numbers, sort by the value rather than asciibetically
        # having sorted inputs means that the outputs are sorted in test mode
        if all(get_name(path).isdigit() for path in input_paths):
            input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
        else:
            input_paths = sorted(input_paths)


        with tf.name_scope("load_images"):
            path_queue = tf.train.string_input_producer(input_paths, shuffle=self.mode == "train")
            reader = tf.WholeFileReader()
            paths, contents = reader.read(path_queue)
            raw_input = decode(contents)
            raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

            assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
            with tf.control_dependencies([assertion]):
                raw_input = tf.identity(raw_input)

            raw_input.set_shape([None, None, 3])

            if self.lab_colorization:
                # load color and brightness from image, no B image exists here
                lab = self.rgb_to_lab(raw_input)
                L_chan, a_chan, b_chan = self.preprocess_lab(lab)
                a_images = tf.expand_dims(L_chan, axis=2)
                b_images = tf.stack([a_chan, b_chan], axis=2)
            else:
                # break apart image pair and move to range [-1, 1]
                width = tf.shape(raw_input)[1]  # [height, width, channels]
                a_images = self.preprocess(raw_input[:, :width // 2, :])
                b_images = self.preprocess(raw_input[:, width // 2:, :])

        if self.which_direction == "AtoB":
            inputs, targets = [a_images, b_images]
        elif self.which_direction == "BtoA":
            inputs, targets = [b_images, a_images]
        else:
            raise Exception("invalid direction")

        # synchronize seed for image operations so that we do the same operations to both
        # input and output images
        seed = random.randint(0, 2 ** 31 - 1)

        def transform(image):
            r = image
            if self.flip:
                r = tf.image.random_flip_left_right(r, seed=seed)

            # area produces a nice downscaling, but does nearest neighbor for upscaling
            # assume we're going to be doing downscaling here
            r = tf.image.resize_images(r, [self.scale_size, self.scale_size], method=tf.image.ResizeMethod.AREA)

            offset = tf.cast(tf.floor(tf.random_uniform([2], 0, self.scale_size - self.CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
            if self.scale_size > self.CROP_SIZE:
                r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], self.CROP_SIZE, self.CROP_SIZE)
            elif self.scale_size < self.CROP_SIZE:
                raise Exception("scale size cannot be less than crop size")
            return r

        with tf.name_scope("input_images"):
            input_images = transform(inputs)

        with tf.name_scope("target_images"):
            target_images = transform(targets)

        paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images],
                                                                  batch_size=self.batch_size)
        steps_per_epoch = int(math.ceil(len(input_paths) / self.batch_size))

        return self.Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )


    def create_generator(self, generator_inputs, generator_outputs_channels):
        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = self.gen_conv(generator_inputs, self.ngf)
            layers.append(output)

        layer_specs = [
            self.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            self.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = self.lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = self.gen_conv(rectified, out_channels)
                output = self.batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (self.ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (self.ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (self.ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (self.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (self.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (self.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (self.ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = self.gen_deconv(rectified, out_channels)
                output = self.batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = self.gen_deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

        return layers[-1]


    def create_model(self, inputs, targets):
        def create_discriminator(discrim_inputs, discrim_targets):
            n_layers = 3
            layers = []

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([discrim_inputs, discrim_targets], axis=3)

            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            with tf.variable_scope("layer_1"):
                convolved = self.discrim_conv(input, self.ndf, stride=2)
                rectified = self.lrelu(convolved, 0.2)
                layers.append(rectified)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = self.ndf * min(2 ** (i + 1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = self.discrim_conv(layers[-1], out_channels, stride=stride)
                    normalized = self.batchnorm(convolved)
                    rectified = self.lrelu(normalized, 0.2)
                    layers.append(rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = self.discrim_conv(rectified, out_channels=1, stride=1)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]

        with tf.variable_scope("generator"):
            out_channels = int(targets.get_shape()[-1])
            outputs = self.create_generator(inputs, out_channels)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = create_discriminator(inputs, targets)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = create_discriminator(inputs, outputs)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + self.EPS) + tf.log(1 - predict_fake + self.EPS)))

        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + self.EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss = gen_loss_GAN * self.gan_weight + gen_loss_L1 * self.l1_weight

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(self.lr, self.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(self.lr, self.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step + 1)

        return self.Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=ema.average(discrim_loss),
            discrim_grads_and_vars=discrim_grads_and_vars,
            gen_loss_GAN=ema.average(gen_loss_GAN),
            gen_loss_L1=ema.average(gen_loss_L1),
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            train=tf.group(update_losses, incr_global_step, gen_train),
        )


    def save_images(self, fetches, step=None):
        image_dir = os.path.join(self.output_dir)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        filesets = []
        for i, in_path in enumerate(fetches["paths"]):
            name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
            fileset = {"name": name, "step": step}

            # for kind in ["inputs", "outputs", "targets"]:
            #     filename = name + "-" + kind + ".png"
            #     if step is not None:
            #         filename = "%08d-%s" % (step, filename)
            #     fileset[kind] = filename
            #     out_path = os.path.join(image_dir, filename)
            #     contents = fetches[kind][i]
            #     with open(out_path, "wb") as f:
            #         f.write(contents)

            filename = name.split('-')[0] + '.jpg'  # + "-" + "outputs.png"
            fileset["outputs"] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches["outputs"][i]
            with open(out_path, "wb") as f:
                f.write(contents)

            filesets.append(fileset)
        return filesets


    def main(self, input_dir, output_dir, checkpoint):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.checkpoint = checkpoint

        if self.seed is None:
            self.seed = random.randint(0, 2 ** 31 - 1)

        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if self.mode == "test" or self.mode == "export":
            if self.checkpoint is None:
                raise Exception("checkpoint required for test mode")

            # load some options from the checkpoint
            options = {"which_direction", "ngf", "ndf"}
            with open(os.path.join(self.checkpoint, "options.json")) as f:
                for key, val in json.loads(f.read()).items():
                    if key in options:
                        print("loaded", key, "=", val)
                        setattr(self, key, val)
            # disable these features in test mode
            self.scale_size = self.CROP_SIZE
            self.flip = False

        # args 프린트 해주는 부분
        # for k, v in a._get_kwargs():
        #    print(k, "=", v)

        # with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        #     f.write(json.dumps(vars(a), sort_keys=True, indent=4))

        examples = self.load_examples()
        print("examples count = %d" % examples.count)

        # inputs and targets are [batch_size, height, width, channels]
        model = self.create_model(examples.inputs, examples.targets)

        # undo colorization splitting on images that we use for display/output
        inputs = self.deprocess(examples.inputs)
        targets = self.deprocess(examples.targets)
        outputs = self.deprocess(model.outputs)

        def convert(image):
            if self.aspect_ratio != 1.0:
                # upscale to correct aspect ratio
                size = [self.CROP_SIZE, int(round(self.CROP_SIZE * self.aspect_ratio))]
                image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

        # reverse any processing on images so they can be written to disk or displayed to user
        with tf.name_scope("convert_inputs"):
            converted_inputs = convert(inputs)

        with tf.name_scope("convert_targets"):
            converted_targets = convert(targets)

        with tf.name_scope("convert_outputs"):
            converted_outputs = convert(outputs)

        with tf.name_scope("encode_images"):
            display_fetches = {
                "paths": examples.paths,
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            }

        # summaries
        # with tf.name_scope("inputs_summary"):
        #     tf.summary.image("inputs", converted_inputs)
        #
        # with tf.name_scope("targets_summary"):
        #     tf.summary.image("targets", converted_targets)
        #
        # with tf.name_scope("outputs_summary"):
        #     tf.summary.image("outputs", converted_outputs)
        #
        # with tf.name_scope("predict_real_summary"):
        #     tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
        #
        # with tf.name_scope("predict_fake_summary"):
        #     tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))
        #
        # tf.summary.scalar("discriminator_loss", model.discrim_loss)
        # tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
        # tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
        #
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        #
        # for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        saver = tf.train.Saver(max_to_keep=1)

        logdir = self.output_dir if (self.trace_freq > 0 or self.summary_freq > 0) else None
        sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)

        with sv.managed_session() as sess:  #  여기서 이상한것들 저장됨
            print("parameter_count =", sess.run(parameter_count))

            if self.checkpoint is not None:
                print("loading model from checkpoint")
                checkpoint = tf.train.latest_checkpoint(self.checkpoint)
                saver.restore(sess, checkpoint)

            max_steps = 2 ** 32
            if self.max_epochs is not None:
                max_steps = examples.steps_per_epoch * self.max_epochs
            if self.max_steps is not None:
                max_steps = self.max_steps

            if self.mode == "test":
                # testing
                # at most, process the test data once
                # start = time.time()
                max_steps = min(examples.steps_per_epoch, max_steps)
                for step in range(max_steps):
                    results = sess.run(display_fetches)
                    filesets = self.save_images(results)
                    # for i, f in enumerate(filesets):
                    #    print("evaluated image", f["name"])

                os.remove(self.concat_dir)

# pix_class = pix2pixClass()
# pix_class.main(input_dir='src/joonggi.jpg', output_dir='dst', checkpoint='facial_train')
