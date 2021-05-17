import numpy as np
import math
import datetime
import pandas as pd
#from datetime import datetime, timedelta
import h5py
import json
import glob
import os
from scipy import signal
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import pywt
from skimage.filters import hessian
import matplotlib.colors as colors

# GROWTH class
class GROWTH:
    def __init__(self, GR):
        ''' Constructor '''
        
        # GR1 = Musala, GR2 = Jungfraujoch, GR3 = Zugspitze
        # Musala has two active channels 0 and 2, Jungfraujoch has one active channel 0, and Zugspitze has one active channel 0
        channel_dict = {'GR1':[0, 2], 'GR2':[0], 'GR3':[0]} # Dictionary with active channels
        self.channels = channel_dict[GR]
        
        path = '/storage/stations' # Path that leads to data files
        self.path = os.path.join(path, GR)
        
        print('GROWTH ' + GR + ' was initialized. Path to data: ' + self.path)

    def __del__(self):
        ''' Destructor '''
        print('Object has been terminated. "I\'ll be back."')
        
    def go_through_files(self, time, path=None):
        '''
        This function goes through files in the path and return only the files of interest
        It has two arguments - time and path. Time can be given as one string or tuple of two strings. 
        
        '''
        if (path != None):
            self.path = path

        residual = None
        one_hour = datetime.timedelta(hours=1)
        if isinstance(time, str) or time == None:
            dt = str2datetime(time)
            self.time_range = dt
            Ym = str(dt.year) + f"{dt:%m}"
            directory = os.path.join(self.path, Ym)
            if (len(time) == 13):
                temp = os.path.join(directory, Ym + f"{dt:%d}" + '_' + f"{dt:%H}" + '*.fits.gz')
                dt_temp = dt - one_hour
                residual = os.path.join(directory, Ym + f"{dt:%d}" + '_' + f"{dt_temp:%H}" + '*.fits.gz')
            elif (len(time) == 11):
                temp = os.path.join(directory, Ym + f"{dt:%d}" + '_' + f"{dt:%H}" + '*.fits.gz')
                dt_temp = dt - one_hour
                residual = os.path.join(directory, Ym + f"{dt:%d}" + '_' + f"{dt_temp:%H}" + '*.fits.gz')
            elif (len(time) == 8):
                temp = os.path.join(directory, Ym + f"{dt:%d}" + '*.fits.gz')
            elif (len(time) == 6):
                temp = os.path.join(directory, '*.fits.gz')
            else:
                print('Wrong format. YYYYMMdd_hhmm - upper case is a must have, lower case letters are optional.')
            files = glob.glob(temp)
            if residual is not None:
                files.extend(glob.glob(residual))
            if files == []:
                print('No data for given datetime. Please check your datetime.')
            files = sort_files(files)
            
        elif isinstance(time, tuple):
            (t1, t2) = time
            dt1 = str2datetime(t1)
            dt2 = str2datetime(t2)
            self.time_range = [dt1, dt2]
            Ym1 = str(dt1.year) + f"{dt1:%m}"
            Ym2 = str(dt2.year) + f"{dt1:%m}"
            if Ym1 == Ym2:
                directory = os.path.join(self.path, Ym1)
                temp = os.path.join(directory, Ym1 + '*.fits.gz')
                files = glob.glob(temp)
                tup = list(map(file2datetime, files))
                df_files = pd.DataFrame(tup, columns=['datetime', 'file'])
                df_files.index = df_files['datetime']
                mask = (df_files.index > dt1) & (df_files.index <= dt2)
                df_files = df_files.loc[mask]
                files = df_files.iloc[:, 1].to_numpy()
            else:
                files = []

        else:
            print('Wrong argument of go_through_files() function')
            files = []
            
        self.files = files  
        print(str(len(files)) + ' .fits.gz files were found.')
        return files

    def read_fits(self, filelist=None, channels=None):
        # Reads .fits file data
        
        if (filelist != None):
            self.files = filelist
        if (channels != None):
            self.channels = channels

        df = pd.DataFrame()
        lst = []
        for file in self.files:
            temp = []
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

                for channel in self.channels:
                    df = extract_data(channel, data, time_standard)
                    temp.append(df)
            for channel in self.channels:
                if lst == []:
                    lst = temp
                else:
                    lst[channel] = lst[channel].append(temp[channel], ignore_index=True)
        data_zip = zip(self.channels, lst)
        data_dict = dict(data_zip)
        self.data = data_dict
        print('Data were loaded into DataFrame.')
        return data_dict
        
    def time_series(self, frequency):
        self.time_series_data = 0
        self.time_series_freq = frequency
        lst = []
        for channel in self.channels:
            df = self.data[channel]
            df.index = df['UTC']
            f = {'Height':[list_height, 'count', 'mean', 'min', 'max', 'sum', 'std'], 'Baseline':['mean', 'min', 'max', 'std']}
            grouped = df.groupby(pd.Grouper(freq=frequency)).agg(f)
            lst.append(grouped)
        data_zip = zip(self.channels, lst)
        data_dict = dict(data_zip)
        self.time_series_data = data_dict
        print('Time series data were created.')
        return data_dict, frequency
    
    def time_series_to_csv(self, filename):
        for channel in self.chennels:
            filename = filename + '_' + str(channel) + '.csv'
            self.time_series_data[channel].to_csv(filename)
            
    def plot_time_series(self, attr=None, stat=None, data=None, file=None):
        title_dict = {'Height': 'Pulse height', 'Baseline':'Pulse baseline'}
        ylabel_dict = {'count': 'Counts [-]', 'mean':'Channel [-]', 'min':'Channel [-]', 'max':'Channel [-]', 'sum':'Channel [-]', 'std':'Channel [-]'}
        attr_types = ['Height', 'Baseline']
        dic = {'Height':['list_height', 'count', 'mean', 'min', 'max', 'sum', 'std'], 'Baseline':['mean', 'min', 'max', 'std']}
        if attr not in attr_types:
            raise ValueError("Invalid attr type. Expected one of: %s" % attr_types)
        if stat not in dic[attr]:
            raise ValueError("Invalid stat type. Expected one of: %s" % dic[attr])   
      
        for channel in self.channels:
            if data == None:
                df = self.time_series_data[channel]
            else:
                df = data[channel]
            smooth = gaussian_filter1d(df['Height']['count'], sigma=2, mode='wrap')
            fig = plt.figure(dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(df.index, df['Height']['count'])
            ax1.plot(df.index, smooth)
            
            ax1.set_title(title_dict[attr])
            ax1.set_xlabel('UTC [-]')
            ax1.set_ylabel(ylabel_dict[stat])
            
            div = int(len(df.index) / 20)
            if (div == 0):
                div = 1
            ax1.set_xticks(df.index[::div])
            ax1.set_xticklabels(df.index[::div], rotation=90)
            plt.tight_layout()
            if file != None:
                file = file + '_' + str(channel) + '.png'
                plt.savefig(file)
            
    def plot_histogram_waterfall(self, data=None, file=None):
        bins = np.linspace(0, 2048, 2049)
        for channel in self.channels:
            if data == None:
                df = self.time_series_data[channel]
            else:
                df = data[channel]
            df_height = df['Height']
            temp = df_height['list_height'].apply(histogram, args=(bins,))
            dt = temp.to_numpy()
            dt = np.stack(dt).T
            c = np.arange(0, len(temp.index))
            fig = plt.figure(dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            im = ax1.pcolormesh(c, bins[:-1], dt, norm=colors.LogNorm())
            #ax1.pcolormesh(c, norm=bins[:-1], dt)
            #ax1.set_xticks(c, temp.index, rotation='vertical')
            div = int(len(c) / 20)
            if (div == 0):
                div = 1
            ax1.set_xticks(c[::div])
            ax1.set_xticklabels(temp.index[::div], rotation=90)
            fig.colorbar(im)
            if file != None:
                file = file + '_' + str(channel) + '.png'
                plt.savefig(file)  
                
    def plot_cwt(self, scales=None, file=None):
        for channel in self.channels:
            df = self.time_series_data[channel]
            df_max = df['Height']
            temp = df_max['count']
            
            fig = plt.figure(dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            scales = np.linspace(1, 5000, 100)
            coeff, freq = pywt.cwt(temp, scales, 'mexh')
            ax1.imshow(abs(coeff), cmap='gray', aspect='auto', vmax=abs(coeff).max(), vmin=-abs(coeff).max())
            
            div = int(len(df.index) / 20)
            print(div)
            if (div == 0):
                div = 1
            xtic = np.arange(0, len(df.index), div)
            ax1.set_xticks(xtic)
            ax1.set_xticklabels(df.index[::div], rotation=90)
            
            if file != None:
                file = file + '_' + str(channel) + '.png'
                plt.savefig(file)
                
            a = hessian(abs(coeff))
            fig = plt.figure(dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.imshow(a, aspect='auto')
            ax1.set_xticks(xtic)
            ax1.set_xticklabels(df.index[::div], rotation=90)
            if file != None:
                file = file + '_hessian_' + str(channel) + '.png'
                plt.savefig(file)

    def plot_histogram(self, file=None):
        bins = np.linspace(0, 2048, 2049)
        for channel in self.channels:
            df = self.data[channel]
            fig = plt.figure(dpi=500)
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.hist(df['Height'], bins=bins, histtype='step')
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.hist(df['Height'], bins=bins, histtype='step', log=True)
            ax1.set_xlabel('Channel [-]')
            ax1.set_ylabel('Counts [-]')
            ax1.set_xlim(0, 500)
            ax2.set_xlim(0, 500)
            ax2.set_xlabel('Channel [-]')
            ax2.set_ylabel('Counts - log scale [-]')
            plt.tight_layout()
            if file != None:
                file = file + '_' + str(channel) + '.png'
                plt.savefig(file)
                
def histogram(x, bins):
    hist, _ = np.histogram(x, bins=bins)
    return hist

def detect_ridges(gray, sigma=3.0):
    hxx, hyy, hxy = hessian_matrix(gray, sigma)
    i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    return i1, i2

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
        data = np.stack([unixTime, phaMax, phaMaxTime, phaMin, phaFirst, phaLast, maxDerivative, baseline, deadCount], axis=1)
        df = pd.DataFrame(data, columns=['Unix', 'Height', 'MaxTime', 'Min', 'First', 'Last', 'maxDerivative', 'Baseline', 'DeadCount'])
        df['UTC'] = pd.to_datetime(df['Unix'] * 1000000000, unit='ns')
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

def clock_verification(gps):
    clock = 1.0e8
    time_tag = np.array([gps[0][0]], dtype="int64")
    for i in range(1, len(gps)):
        if (gps[i][2] != "NULL") and (gps[i][2]!=""):
            time_tag = np.append(time_tag, gps[i][0])
    time_tag = time_tag&0xFFFFFFFFFF
    time_tag = check_loop(time_tag, 2**40)
    if (time_tag.size < 2):
        return clock
    else:
        second=float(time_tag[time_tag.size-1] - time_tag[0])/clock
        second_precise=round(second)
        clock_precise=float(time_tag[time_tag.size-1] - time_tag[0]) / float(second_precise)
        rate=(clock_precise-clock)/clock
        if (abs(rate) > 1.0e-4):
            return clock
        else:
            return clock_precise
        
def sort_files(files):
    out, lst = [], []
    for file in files:
        base, rest = file.split('_')
        date = os.path.basename(base)
        base = os.path.dirname(base)
        time, fits, gz = rest.split('.')
        lst.append((base, date, time, fits, gz))
    lst = sorted(lst, key=lambda element: (element[1], element[2]))
    for base, date, time, fits, gz in lst:
        base = os.path.join(base, date)
        name = base + '_' + time + '.' + fits + '.' + gz
        out.append(name)
    return out

def list_height(x):
    return list(x)

def str2datetime(datetime_str):
    if (len(datetime_str) == 15):
        datetime_object = datetime.datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
    elif (len(datetime_str) == 13):
        datetime_object = datetime.datetime.strptime(datetime_str, '%Y%m%d_%H%M')
    elif (len(datetime_str) == 11):
        datetime_object = datetime.datetime.strptime(datetime_str, '%Y%m%d_%H')
    elif (len(datetime_str) == 8):
        datetime_object = datetime.datetime.strptime(datetime_str, '%Y%m%d')
    elif (len(datetime_str) == 6):
        datetime_object = datetime.datetime.strptime(datetime_str, '%Y%m')
    else:
        print('Wrong format. YYYYMMdd_hhmm - upper case is a must have, lower case is optional.')
    #print(datetime_object)
    return datetime_object

def file2datetime(file):
    basename = os.path.basename(file)
    basename = os.path.splitext(basename)[0]
    basename = os.path.splitext(basename)[0]
    dt = str2datetime(basename)
    return dt, file
        
