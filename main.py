import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from model import cyclegan


def main(_):

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    src = './src/test.jpg'
    model = 'vangogh2photo'
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess,model, src)
        dst = model.test()

    print(dst)


if __name__ == '__main__':
    tf.app.run()