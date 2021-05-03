import numpy as np
import math
import datetime
import pandas as pd
import h5py
import json
import glob
import os
import astropy.io.fits as fitsio

# Timepix class
class GROWTH:
    def __init__(self, logname, dirname):
        # Constructor
        self.confidence = 1

    def read_data(self, channels, filelist):
        # Reads .fits file data
        for file in filelist:
            if (os.path.exists(file) == False):
                print("Error: File " + file + ' could not be found.')
                exit()
            fits_file = fitsio.open(file)
            if len(fits_file) < 3:
                print("Warning: This FITS file is not properly finalized. Pipeline stopped.")
                gps_status = False
                exit()
            else:
                data = fits_file[1].data
                gps = fits_file[2].data
                gps_status = gps_status_test(gps[0][2])
                if (gps_status == True):
                    time_standard=gps_base_time(gps)
                else:
                    time_standard=non_gps_base_time(input_file, event, gps)

                for channel in channels:
                    df = extract_data(channel, data, time_standard)
        self.data = df

def extract_data(adc_channel, event, time_standard):
    clock = time_standard[2]
    data_mask_channel = event[(event["boardIndexAndChannel"]==adc_channel)]
    if (len(data_mask_channel) != 0):
        unixTime = np.array(data_mask_channel["timeTag"], np.float64)
        unixTime = check_loop(unixTime, 2.0**40)
        unixTime = time_standard[0] + (unixTime - time_standard[1]) / clock

        phaMax = np.array(data_mask_channel["phaMax"], np.int32)
        phaMax = phaMax - 2048
        phaMaxTime = np.array(data_mask_channel["phaMaxTime"], np.int32)
        phaMin = np.array(data_mask_channel["phaMin"], np.int32)
        phaMin = phaMin - 2048
        phaFirst = np.array(data_mask_channel["phaFirst"], np.int32)
        phaFirst = phaFirst - 2048
        phaLast = np.array(data_mask_channel["phaLast"], np.int32)
        phaLast = phaLast - 2048
        maxDerivative = np.array(data_mask_channel["maxDerivative"], np.int32)
        baseline = np.array(data_mask_channel["baseline"], np.int32)
        triggerCount = np.array(data_mask_channel["triggerCount"], np.int32)
        triggerCount = check_loop(triggerCount, 2**16)
        triggerCountShift = np.roll(triggerCount, 1)
        deadCount = triggerCount - triggerCountShift - 1
        deadCount[0] = 0
        data = np.stack([unixTime, phaMax, phaMaxTime, phaMin, phaFirst, phaLast, maxDerivative, baseline, deadCount])
        df = pd.DataFrame(data, columns=['Time', 'Max', 'MaxTime', 'Min', 'First', 'Last', 'maxDerivative', 'Baseline', 'DeadCount'])
        print(df)
        return df
    else:
        return np.array([0])

def check_loop(array, maximum):
    if (array[0] > array[array.size-1]):
        noloop = array[array>=array[0]]
        loop = array[array<array[0]]
        loop += maximum
        narray = np.concatenate([noloop, loop])
        return narray
    else:
        return array

def gps_base_time(gps):
    time_tag_base = int(gps[0][0])&0xFFFFFFFFFF
    unixtime_base = float(gps[0][1])
    gps_string_base = gps[0][2][8:14]
    time_obj_unixtime = datetime.datetime.fromtimestamp(unixtime_base, datetime.timezone.utc)
    time_str_gps = time_obj_unixtime.strftime("%Y%m%d")+" "+gps_string_base+"+00:00"
    time_obj_precise = datetime.datetime.strptime(time_str_gps, "%Y%m%d %H%M%S%z")+datetime.timedelta(seconds=1.0)
    delta_time_gps = time_obj_unixtime - time_obj_precise
    if (delta_time_gps.total_seconds() > 12.0 * 3600.0):
        time_obj_precise = time_obj_precise+datetime.timedelta(days=1.0)
    elif (delta_time_gps.total_seconds() < -12.0 * 3600.0):
        time_obj_precise = time_obj_precise - datetime.timedelta(days=1.0)
    unixtime_precise = time_obj_precise.timestamp()
    clock = clock_verification(gps)
    time_standard = [unixtime_precise, time_tag_base, clock]
    return time_standard

def non_gps_base_time(filename, event, gps):
    time_str_file = (os.path.basename(filename).split('.', 1)[0])+"+00:00"
    time_obj_file = datetime.datetime.strptime(time_str_file, "%Y%m%d_%H%M%S%z")
    time_obj_gps = datetime.datetime.fromtimestamp(float(gps[0][1]), datetime.timezone.utc)
    delta_time_obj = time_obj_gps - time_obj_file
    time_lag_hour = round((delta_time_obj.total_seconds()) / 3600.0)
    time_obj_file = time_obj_file + datetime.timedelta(hours=time_lag_hour)
    time_standard = [time_obj_file.timestamp(), int(event[0][1]), 1.0e8]
    return time_standard

def gps_status_test(string):
    if (string=="GP") or (string=="NULL"):
        gps_status = False
    else:
        gps_status = True
    return gps_status
