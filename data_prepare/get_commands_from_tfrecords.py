"""
Tensorflow 1.8!
"""
import tensorflow as tf
import os
import logging
import shutil

#tf.enable_eager_execution()

DIR = "/unreliable/DATASETS/europilot/tfrecords-com-normal/train/"
filenames = []
TARGET = "/unreliable/DATASETS/europilot/tfrecords-com-normal/lane/"

logFormatter = logging.Formatter("%(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler("image_sets/train-normal-commands.txt")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

n_files = 0
for f in os.listdir(DIR):
    filenames.append(DIR + f)
    n_files += 1
dataset = tf.data.TFRecordDataset(filenames)

# Use `tf.parse_single_example()` to extract data from a `tf.Example`
# protocol buffer, and perform any additional per-record preprocessing.
def parser(record):
    keys_to_features = {
        'image/class/video_name': tf.FixedLenFeature([1], dtype=tf.string, default_value=''),
        'image/command' : tf.FixedLenFeature([], tf.int64),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    name = parsed['image/class/video_name']
    command = parsed['image/command']
    # print(name)
#    tf.Print(name,[name])
#    rootLogger.info("{0}:{1}".format(name, command))
    return {"name": name}, {"command" : command}

# Use `Dataset.map()` to build a pair of a feature dictionary and a label
# tensor for each example.
dataset = dataset.map(parser)
#dataset = dataset.shuffle(buffer_size=2000)
dataset = dataset.batch(1)
dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
features, labels = iterator.get_next()
sess = tf.Session()
# `features` is a dictionary in which each value is a batch of values for
# that feature; `labels` is a batch of labels.
for i in range(n_files):
#    features, labels = iterator.get_next()
    tfrecord, command = sess.run([features, labels])
    name = str(tfrecord['name'][0])[-23:-5] + "tfrecords"
    command = command['command'][0]
    rootLogger.info("{0}:{1}".format(name, command))

    if command == 2:
        shutil.move(DIR + name, TARGET + name)


#features['name'][0][0],float(labels['command'][0])))

