
# Top Left: 280, 170
# Bottom Right: 835, 485
# Shape: (315, 535, 3)
# Width: 535
# Height: 315
# img[170:485,300:835]]

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from json_to_speed import get_interpolated_speed_xy
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import random

from data_providers.nexar_large_speed import MyDataset
from util_car import write_text_on_image, visualize_images

tf.app.flags.DEFINE_string('video_index', '/unreliable/DATASETS/europilot/sequences',
                           'folder containing folders of sequences of frames')

tf.app.flags.DEFINE_string('csv',
                           '/unreliable/DATASETS/europilot/all_final.csv',
                           # '9d0_7a1_3f8_15b_final.csv',
                           # '/home/ivan/Downloads/europilot_data/9d0c3c2b_7a13935b_3f80cca8_final.csv',
                    'path to .csv containing image name, wheel-axis and speed mapping')

tf.app.flags.DEFINE_string('output_directory', '/unreliable/DATASETS/europilot/tfrecords/train/',
                           'Output tfrecords data directory')

tf.app.flags.DEFINE_integer('truncate_frames', 360, 'Number of frames to leave in the saved tfrecords')

# constant for the high resolution
HEIGHT = 315
WIDTH = 535

FLAGS = tf.app.flags.FLAGS
df = None
logfile = open("tfrecords_course.log", "w")

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def get_timestamp_from_filename(fname):
    s = fname.split('.')[0].split('_')
    # print (s)
    s = '_'.join(s[1:len(s)])
    # print(s)
    datetime_object = datetime.strptime(s, '%Y_%m_%d_%H_%M_%S_%f')
    # print(datetime_object)
    return  (datetime_object - datetime(1970, 1, 1)).total_seconds() * 1000


def get_wheels_to_fake_course_rescaled(wh, names, speed_list):
    """
    Wheels to course transformation using sequence-based orientation framework
    """
    res = []
    c = random.randint(0, 360)
    for w in wh:
        c += (w / 10000.0) * 180
        c = c % 360
        res.append(c)
    for i in range(len(res)):
        logfile.write(names[i]  + ' ' + str(res[i]) + ' ' + str(wh[i]) + '\n')
    return res, 10


def get_fake_speeds(speed_list, wheel_list, timestamps, video_path):
    """
    Get interpolated speed for Europilot data
    :param speed_list:
    :param wheel_list:
    :param video_path:
    :return:
    """
    json = {}
    json['speed'] = speed_list
    json['course'] = wheel_list
    json['timestamp'] = timestamps
    json['startTime'] = timestamps[0]
    json['endTime'] = timestamps[-1]

    return get_interpolated_speed_xy(json, 10)



def read_one_video(video_path):
    """
    Here's a video is actually already a sequence of frames.
    :param video_path: folder containing images
    :return:
    """
    image_list = []
    wheel_list = []
    speed_list = []
    timestamps = []
    print ( video_path)
    logfile.write(video_path + '\n')
    names = []
    m = 0
    last = None
    try:
        for subdir, dirs, files in os.walk(video_path): 
            for fname in sorted(files): 
                with open(os.path.join(subdir, fname), 'r') as f:
                    image_data = f.read()
                    image_list.append(image_data)
                    if m % 10 == 0:
                    # try:
                        w = int(df[df['img'] == str(fname)].values[0][1])
                        s = float(df[df['img'] == str(fname)].values[0][2])  /  3.6  # km/h --> m/s
                        speed_list.append(s)
                        wheel_list.append(w)
                        timestamps.append(get_timestamp_from_filename(fname))
                        names.append(fname)
                    # except Exception:
                        pass
                    m += 1
                    last = fname
    except Exception as e:
        print(e)
        return None, False

    with open(os.path.join(video_path, last), 'r') as f:
        image_data = f.read()
        # image_list.append(image_data)
        wheel_list.append(int(df[df['img'] == str(last)].values[0][1]))
        speed_list.append(float(df[df['img'] == str(last)].values[0][2]) / 3.6)  # km/h --> m/s
        timestamps.append(get_timestamp_from_filename(last))
        names.append(last)

    print("images:{0}".format(len(image_list)))
    print("wheel-axis:{0}".format(len(wheel_list)))
    print(len(image_list))
    if len(image_list) != 360:
        return None, False
    wheel_list, number_of_turns = get_wheels_to_fake_course_rescaled(wheel_list, names, speed_list)

    speeds = get_fake_speeds(speed_list, wheel_list, timestamps, video_path)
    print("speeds:{0}".format(len(speeds)))
    logfile.write("speeds:{0}\n".format(len(speeds)))
    # print (speeds[:2])
    if speeds is None:
        # if speed is none, the error message is printed in other functions
        return None, False

    if speeds.shape[0] < FLAGS.truncate_frames:
        print("skipping since speeds are too short!")
        return None, False

    speeds = speeds[:FLAGS.truncate_frames, :]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(HEIGHT),
        'image/width': _int64_feature(WIDTH),
        'image/channel': _int64_feature(3),
        'image/class/video_name': _bytes_feature([video_path]),
        'image/format': _bytes_feature(['JPEG']),
        'image/encoded': _bytes_feature(image_list),
        'image/wheel': _float_feature(wheel_list),
        'image/speeds': _float_feature(speeds.ravel().tolist())
    }))

    return example, True


def parse_path(video_path):
    fd, fname = os.path.split(video_path)
    # fprefix = fname.split(".")[0]
    fprefix = fname
    out_name = os.path.join(FLAGS.output_directory, fprefix + ".tfrecords")

    return (fprefix, out_name)


def convert_one(video_path):
    fprefix, out_name = parse_path( video_path)
    if not os.path.exists(out_name):
        example, state = read_one_video( video_path)
        if state:
            writer = tf.python_io.TFRecordWriter(out_name)
            writer.write(example.SerializeToString())
            writer.close()


if __name__ == '__main__':
    df = pd.read_csv(str(FLAGS.csv))
    print(df.shape)
    j = 1
    if not tf.gfile.Exists(FLAGS.output_directory):
        tf.gfile.MakeDirs(FLAGS.output_directory)

    for video_path in os.listdir(FLAGS.video_index):
        if video_path == "1":
            continue
        convert_one(os.path.join(FLAGS.video_index, video_path))
        print(j)
        j += 1

    print('Finished processing all files')
