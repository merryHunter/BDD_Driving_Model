import json
import matplotlib.pyplot as plt
import os
import numpy as np
import overpass
import logging

api = overpass.API()

logFormatter = logging.Formatter("%(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler("normal/command-train-02.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

STRAIGHT = 5.0
TURN_RIGHT = 4.0
TURN_LEFT = 3.0
LANE_FOLLOW = 2.0
MIN_AVG = 25
DIR = "/unreliable/DATASETS/bdd-data-v1/videos/train-parts/train-02/info/"
SAVE_DIR = "/unreliable/DATASETS/carla/train-02-info-carla/"

def get_rolling_window_diff(c, window=3):
    c_shifted = np.roll(c, window)
    return np.subtract(c, c_shifted)


def find(key, dictionary):
    for k, v in dictionary.iteritems():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                if isinstance(d, dict):
                    for result in find(key, d):
                        yield result


def is_traffic_lights_from_api(gps, around="1.0,"):
    """
    @param gps: gps point
    @param around: proximity distance in miles
    """
    query = "node(around:" + around
    query += gps + ");(._;>;);way(around:" + around + gps +");(._;>;);out body;"
    try:
        response = api.get(query)
#         print(response["features"])
        if list(find('highway', response))[0] == u'traffic_signals':
            return True
    except Exception as e:
        pass
    return False

def count_turns(x,k=2):
    sign = -1
    count = 0
    for i in range(len(x) - k):
        if x[i+k] > x[i] and abs(x[i+k] - x[i]) >1:
            if sign != -1:
                count += 1
                sign = -1
        elif x[i+k] < x[i]and abs(x[i+k] - x[i]) >1:
            if sign != 1:
                count +=1
                sign = 1
    return count


def get_command_from_course(course):
    min_c = min(course)
    max_c = max(course)
    min_ind = course.index(min_c)
    max_ind = course.index(max_c)
    
    window = 3
    c = LANE_FOLLOW
    diffs = get_rolling_window_diff(course[:max_ind], window)
    avg_d = np.mean(diffs[window:])
    # check if it's over zero course
    if min_c < 10 and max_c > 345:
        # check if it's follow lane
        if avg_d > MIN_AVG:
            print("=" * 40)
            c = LANE_FOLLOW
        elif course[0] > course[len(course) - 1]:
            c = TURN_RIGHT
        else:
            c = TURN_LEFT
    else:
        if abs(max_c - min_c) < 45:  # maybe even around 90 ????
            if is_traffic_lights_from_api(gps[10]):
                c = STRAIGHT
            else:
                c = LANE_FOLLOW
        elif abs(min_ind - max_ind) > 30:
            c = LANE_FOLLOW
        elif min_ind > max_ind:
            c = TURN_LEFT
        elif min_ind < max_ind:
            c = TURN_RIGHT
    return c        
    

def is_vehicle_stopped(speed, stop_threshold=3.0, max_threshold=4.2):
    if np.average(speed) < stop_threshold:
        return True
    if max(speed) < max_threshold:
        return True
    return False
    
    
if __name__ == "__main__":
    courses = []
    commands = []
    gps = []
    gps_list = []
    files = []
    i = 0
    for f in os.listdir(DIR):
        info_file = open(DIR + f, "r")
        info = json.load(info_file)
        speed = []
        course = []
        gps = []
        k = 0
        i += 1
        for l in info['locations']:
            lat = l['latitude']
            lon = l['longitude']
            cur = l['course']
            sp = l['speed']
            speed.append(sp)
            course.append(cur)
            gps.append(str(lat) + ',' + str(lon))
        if len(course) == 0 or len(speed) == 0 or len(gps) < 12:
            rootLogger.info("corrupted data")
            continue
        is_moving = True
        if is_vehicle_stopped(speed):
            rootLogger.info("No moving: {0}\n".format(f))
            c = LANE_FOLLOW
            commands.append(c)
            is_moving = False
#            continue
#        course_dec = [int(i / 10) for i in course]
        # dicard multiturn videos
#        turns = count_turns(course_dec)
#        if turns > 2:
#            rootLogger.info("Multiturns:{0}".format(turns))
            #continue
        #courses.append(course)
        #gps_list.append(gps)
 
        files.append(f)
        # get pseudo high-level command
        # ================   
        if is_moving:  
            c = get_command_from_course(course)
            commands.append(c)
        # =================
        rootLogger.info("{0}:{1}".format(f, c))
        info['command'] = int(c)
        x = json.dumps(info, indent=4) #, sort_keys=True)
        with open(SAVE_DIR + f, "w") as out:
            out.write(x)
        if i % 100 == 0:
            print("Processed:{0}".format(i))
