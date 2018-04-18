"""
04/12/2017
@author Ivan Chernukha

Preprocessing raw data from Europilot simulator.

Steps including:
1. Extracting speed from images. (if not provided already extracted)
2. Cropping to front-view only.
3. Creating sequences of length 360 frames. 

Input parameter is an absolute path to the root folder of europilot data,
which contains directory named 'raw' containing all raw data folders
like 9be24hs etc.
"""

import cv2
import pandas as pd
import os
from shutil import copyfile
from PIL import Image
import sys
import multiprocessing
#from enhance import NeuralEnhancer
import numpy as np
import scipy.ndimage, scipy.misc

EUROPILOT_DATA_ROOT = None
CROPPED_DIR = None
SPEED_DIR = None
OUT_DIR = None
RUN_ID = None
FINAL_CSV = None
enhancer = None
# template = cv2.imread('digits-template.png')
# template[:,:55,:] = 255
####

digits_template = []
for d in sorted(os.listdir('enhanced_digits_templates/')):
    digits_template.append(cv2.imread('enhanced_digits_templates/' + d, 0))
#    print (d)


def extract_and_crop():
    """
    Steps 1 and 2.
    """
    
#    global enhancer
    enhancer = NeuralEnhancer(loader=False)
    
    df = pd.read_csv(EUROPILOT_DATA_ROOT + "/data_v1.csv")
    try:
        os.mkdir(CROPPED_DIR)
        os.mkdir(SPEED_DIR)
    except:
        pass

    # run extraction and cropping in parallel over raw directories
    # num_processes = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=num_processes) #
    # pool.map(process_dirs, os.listdir(EUROPILOT_DATA_ROOT + "/raw"))
    # unite cv2 reading and do extract and crop and save
    process_dirs(DATA_RAW, df)


def match_templates(img):
    # try to match one by one all digits
    for i in range(10): # 10 digits in digits_template:
        w, h = digits_template[i].shape[::-1]
        res = cv2.matchTemplate(img,digits_template[i],cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]): # it is assumed there will be only one digit, however this function searches for all occrences
#             cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            return i
    return -1


def get_speed_by_template_matching(img_name, j):
    """
    # get speed box
    # enhance it - enhance and increase box size
    # get two parts of box - two digit place
    # recognize via template matching
    # return speed
    """
    speed = -1
    # get speed box

    img = cv2.imread(DATA_RAW + img_name)
    print(DATA_RAW + img_name)
    # crop to front-view
    img_front = img[170:485, 300:835]
    cv2.imwrite(CROPPED_DIR + img_name, img_front)
    # as we have already extracted speed, we finish here, otherwise comment this line
    return -2

    # go over each 10-th image, otherwise skip extraction
    # (will be interpolated at prepare_eurotruck_records.py)
    if j % 10 != 0:
        return -2

    # crop to speed box
    speed_box = img[488:499, 765:789]
    cv2.imwrite('temp.png', speed_box)

    # enhance it - enhance and increase box size
    img = scipy.ndimage.imread('temp.png', mode='RGB')
    out = enhancer.process(img)
    out.save('temp.png')

    # get two parts of box - two digit place
    img_rgb = cv2.imread("temp.png")
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # match digits template for each digit separately
    # print(img_gray.shape)
    head = match_templates(img_gray[:,:75])
    # erase first digit
    img_gray[:, 45:65] = 80
    tail = match_templates(img_gray)
    # print ("head " + str(head))
    # print ("tail " + str(tail))
    if head == 0:  # bad fix for zero false positive detection
        head = 8
    if tail != -1 and head != -1:
        speed = head * 10 + tail  # multiply by 10 to shift digit to decimal
    elif tail == -1 and head != -1:
        speed = head * 10
    elif tail != -1 and tail != 0 and head == -1:
        speed = tail

    if speed == -1:
        raise Exception
    print ("{0}\n, speed: {1} ".format(DATA_RAW + img_name, speed))
    return speed


def process_dirs(root_dir, df):

    bad = 0
    labels = []
    imgs = []
    speeds = []
    print (df.shape)
    print (df[df['img'] == '9d0c3c2b_2017_07_27_14_55_08_26.jpg'])
    count = 10
    prev_s = -1
    for i, r in df.iterrows():
        if RUN_ID in r[0]:
            try:
                s = get_speed_by_template_matching(r[0], count)
                count += 1
                if s != -2:
                    prev_s = s
            except Exception as e:
                print (str(e))
                bad += 1
                if bad % 100 == 0:
                    print (bad)
                continue # skip bad images
            speeds.append(prev_s)
            imgs.append(r[0])
            labels.append(r[1])
            if count % 1000 == 0:
                print("Processed images: " + str(count ))
            # if count == 50000:
            #     break


    print("Bad images: " + str(bad))

    df_cropped = pd.DataFrame()
    df_cropped['img'] = imgs
    df_cropped['wheel'] = labels
    df_cropped['speed'] = speeds
    # TODO: check index in the written csv!
    df_cropped.to_csv(EUROPILOT_DATA_ROOT + FINAL_CSV, index=False)
        

def save_sequence(sequence, seq_i):
    os.mkdir(OUT_DIR + str(seq_i))
    for f in sequence:
        copyfile(CROPPED_DIR + f[0],OUT_DIR + str(seq_i)+ '/' + f[0])


def create_sequences():
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    df_cropped = pd.read_csv(EUROPILOT_DATA_ROOT + FINAL_CSV)
    sequence = []
    seq_i = 1
    diff_thresh_mins = 5
    diff_thresh_hours = 1
    max_seq_lenght = 360  # 36 / 2 sec of driving
    # take the first minutes value of df:
    prev_frame_mins = int(df_cropped[:1].values[0][0].split('_')[5])
    prev_frame_hours = int(df_cropped[:1].values[0][0].split('_')[4])
    seq_info = []
    for r in sorted(os.listdir(CROPPED_DIR)):
        cur_mins = int(r.split('_')[5])
        cur_hours = int(r.split('_')[4])
        if cur_hours - prev_frame_hours < diff_thresh_hours and \
                cur_mins - prev_frame_mins < diff_thresh_mins and \
                len(sequence) < max_seq_lenght:

            sequence.append((r, r))

        else:  # another sequence started
            # save seq to a new folder
            save_sequence(sequence, seq_i)
            s = "Sequence lenght: {0}, index: {1} \n".format(len(sequence), seq_i)
            print(s)
            seq_info.append(s)
            sequence = []
            seq_i += 1

        prev_frame_mins = cur_mins
        prev_frame_hours = cur_hours

    save_sequence(sequence, seq_i)
    s = "Sequence lenght: {0}, index: {1} \n".format(len(sequence), seq_i)
    print(s)
    seq_info.append(s)
    with open(RUN_ID + '-seqlog.txt', "w") as f:
        for se in seq_info:
            f.write(se)


if __name__ == "__main__":
    EUROPILOT_DATA_ROOT = sys.argv[1]
    RUN_ID = sys.argv[2]
    FINAL_CSV = "/" + RUN_ID + "all_final.csv"
#    FINAL_CSV = "/all_final.csv"
    DATA_RAW = EUROPILOT_DATA_ROOT + '/raw/'
    CROPPED_DIR = EUROPILOT_DATA_ROOT + 'cropped/'
    SPEED_DIR = EUROPILOT_DATA_ROOT + 'speed_boxes/'
    OUT_DIR = EUROPILOT_DATA_ROOT + '/sequences/'

    extract_and_crop()

    create_sequences()























