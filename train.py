import tensorflow as tf
import numpy as np
import os
from skimage import io, transform
import glob
import math
import network_model
import datetime
import h5py

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def charb(x, y):
    return tf.reduce_sum(tf.square(x - y) + 0.01)


def getBatch(x, y, bsize):
    h, w, c = x[0].shape
    batch_x = np.zeros((bsize, h, w, c))
    batch_y = np.zeros((bsize, h, w, int(c / 2)))

    for i in range(0, bsize):
        r = np.random.randint(0, len(x) - 1)
        batch_x[i] = x[r]
        batch_y[i] = y[r]

        batch_x[i] = (batch_x[i] - 127) / 127
        batch_y[i] = (batch_y[i] - 127) / 127

    return batch_x, batch_y


################# load data ##############

"""
GLOBAL CONFIG
"""
n_epochs = 20
n_samples = 5462
bsize = 4
b_per_epoc = math.floor(n_samples/bsize)
run_code = None

"""
Directories
"""
#at each run, create a folder
#   . run_YYYYMMDD_hhmm
#   |----model
#       |----model checkpoints
#   |----log
#       |----log files for tensorboard

if run_code is None:
    # new run: we create all the necessary folders
    run_code = datetime.datetime.now().strftime('%Y%m%d__%H%M')

    dir_run = 'run_' + run_code
    dir_log = dir_run + "/log"
    dir_model = dir_run + "/model"
    dir_img = dir_run + "/sample_images"
    tf.gfile.MakeDirs(dir_run)
    tf.gfile.MakeDirs(dir_model)
    tf.gfile.MakeDirs(dir_log)
    tf.gfile.MakeDirs(dir_img)
    print('*** RUN FOLDER : {}'.format(dir_run))
else:
    # not a new run: we just delete LOG folder. Later, we will also load previous model from dir_model
    dir_run = run_code
    dir_log = dir_run + "/log"
    dir_model = dir_run + "/model"
    dir_img = dir_run + "/sample_images"


    if tf.gfile.Exists(dir_log):
        tf.gfile.DeleteRecursively(dir_log)
    print('*** CONTINUE EXISTING RUN')
    print('*** RUN FOLDER : {}'.format(dir_run))


dir_data = "data\\"

images = []
fnames = glob.glob(dir_data + "*.png")

if tf.gfile.Exists('images.data'):
    with h5py.File('images.data', 'r') as hf:
        print('Loading data...')
        images = hf['data'][:]
        print('Done loading!')
else:
    for i in sorted(fnames):
        tmp = transform.resize(io.imread(i), (96, 160), preserve_range=True)
        print(i)
        images.append(tmp)
    with h5py.File('images.data', 'w') as hf:
        print('Dumping...')
        hf.create_dataset('data', data=images)
        print('Dumping done!')


x = []
y = []

for i in images:
    i = tf.image.per_image_standardization(i)

for i in range(0, len(images) - 3, 3):
    x.append(np.dstack((images[i], images[i + 2])))
    y.append(images[i + 1])

h, w, c = x[0].shape

with tf.Graph().as_default():
    inp = tf.placeholder(tf.float32, [None, h, w, c])
    gt = tf.placeholder(tf.float32, [None, h, w, int(c / 2)])

    # net = discriminator(inp)
    cnn_autoencoder = network_model.ConvolutionalInterpolatorNetwork(X=inp, Y=gt)
    # loss = charb(net, gt)
    print('%% Total parameters : ',
          np.sum([np.prod(dim) for dim in [variable.get_shape() for variable in tf.trainable_variables()]]))

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        # restore model from disk
        if tf.gfile.Exists(os.path.join(dir_model, 'model.ckpt.meta')):
            print("Previous model found. Restoring...")
            saver.restore(sess, os.path.join(dir_model, 'model.ckpt'))
            print("Model Restored")
        else:
            print("No previous model found: starting from scratch")
            sess.run(init)

        # with tf.Session() as sess:
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(dir_log, sess.graph)

        for i in range(0, n_epochs):
            for j in range(0, int(b_per_epoc)):
                b_x, b_y = getBatch(x, y, bsize)
                feed_dict = {inp: b_x, gt: b_y}
                _, loss_v = sess.run([cnn_autoencoder.training, cnn_autoencoder.loss], feed_dict=feed_dict)
                print(loss_v)

                if (j % 50) == 0:
                    b_x = np.expand_dims(((x[1415] - 127) / 127), axis=0)
                    b_y = np.expand_dims(((y[1415] - 127) / 127), axis=0)
                    feed_dict = {inp: b_x, gt: b_y}
                    image = sess.run(cnn_autoencoder.inference, feed_dict=feed_dict)
                    fname = dir_img + "/zibbibbulu_ep{:02d}_{:05d}".format(i,j) + ".png"
                    print(i, j, fname)
                    io.imsave(fname, (image[0] + 1) / 2)

            # b_x = (x[1500] - 127) / 127
            # b_y = (y[1500] - 127) / 127
            # feed_dict = {inp: b_x, gt: b_y}
            # summary_str = sess.run(summary, feed_dict=feed_dict)
            # summary_writer.add_summary(summary_str, i)
            # summary_writer.flush()

            save_path = saver.save(sess, os.path.join(dir_model, 'model.ckpt'))
            print("Model saved in file: %s" % save_path)

            # # Evaluation
            # image = sess.run(cnn_autoencoder.inference, feed_dict=feed_dict)
            # print(np.max(image), np.min(image))
            # io.imsave(dir_img+"/zibbibbulu{0:05d}".format(i) + ".png", (image[0] + 1) / 2)
