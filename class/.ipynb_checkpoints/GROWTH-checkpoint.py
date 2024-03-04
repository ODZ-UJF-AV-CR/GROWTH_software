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
import matplotlib.dates as mdates
import random

# GROWTH class
class GROWTH:
    def __init__(self, GR):
        ''' Constructor '''
        
        # GR1 = Musala, GR2 = Jungfraujoch, GR3 = Zugspitze
        # Musala has two active channels 0 and 2, Jungfraujoch has one active channel 0, and Zugspitze has one active channel 0
        channel_dict = {'GR1':[0], 'GR2':[0], 'GR3':[0]} # Dictionary with active channels
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
        It has two arguments - time and path. Time must be given as as the tuple of two strings. 
        
        '''
        if (path != None):
            self.path = path
  
        if isinstance(time, tuple):
            (t1, t2) = time
            delta = datetime.timedelta(hours = 0.5)
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
                mask = (df_files.index > dt1 - delta) & (df_files.index <= dt2)
                df_files = df_files.loc[mask]
                files = df_files.iloc[:, 1].to_numpy()
            else:
                files = []

        else:
            print('Wrong argument of go_through_files() function')
            files = []
        
        files = list(np.unique(files))
        self.files = files  
        print(str(len(files)) + ' .fits.gz files were found.')
        #print(files)
        return files
           
    def per_event_data(self):
        out = []
        for channel in self.channels:
            gps_ok, gps_not_ok = gps_process(self.files)
            self.gps_ok = gps_ok
            self.gps_not_ok = gps_not_ok
            df, end_list = data_process(gps_ok, gps_not_ok, channel)
            self.gps_not_ok = merge_gps_not_ok(self.gps_not_ok, end_list)
            out.append((channel, (df, gps_ok, gps_not_ok)))
            
            fig = plt.figure(dpi=500)
            ax = fig.add_subplot(1, 1, 1)
            for files, ticks, times, ps in gps_ok:
                ax.scatter(times, ticks)
        self.data = dict(out)
        #a = df['TimeTag']
        #print(a[1])
        return self.data
        
    '''
    def read_fits(self, filelist=None, channels=None):
        # Reads .fits file data
        
        if (filelist != None):
            self.files = filelist
        if (channels != None):
            self.channels = channels

        df = pd.DataFrame()
        lst = []
        for file in self.files:
            print(file)
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
                #print(gps)
                gps_status = gps_status_test(gps)
                #print(gps_status)
                if (gps_status == True):
                    time_standard = gps_base_time(gps)
                else:
                    time_standard = non_gps_base_time(file)
                    #continue

                for channel in self.channels:
                    #print(time_standard)
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
        #print(data_dict)
        return data_dict
    
    
    def export(self, output_file=None, channels=None):
        if (output_file == None):
            tm = str(self.time_range[1])
            _, tm = tm.split()
            #print(tm)
            self.output_file = str(self.time_range[0]) + '_' + tm
        if (channels == None):
            out_channel = self.channels
            
        for channel in out_channel:
            df = self.data[channel]
            filename = self.output_file + '_' + str(channel) + '.csv'
            df.to_csv(filename)
            #print(df.shape)
    
    def read_RT_csv(self, file):
        df = pd.read_csv(file)
        df = df[df['amplitude']>=400]
        #print(df)
        timestamp = list(df['timestamp'].to_numpy())
        timestamp = list(map(str, timestamp))
        microseconds = list(df['microseconds'].to_numpy())
        microseconds = list(map(str, microseconds))
        time = list(map(''.join, zip(timestamp, microseconds)))
        df = df.drop(columns=['microseconds', 'timestamp'])
        df.index = pd.to_datetime(time, unit='us')
        df.columns = ['ll', 'Height']
        df = df.drop(columns=['ll'])
        
        mask = df.index.to_series().between('2020-01-01', '2022-01-10')
        df = df[mask]
        #print(df)
        ten_min = df.groupby(pd.Grouper(freq='10min')).agg(['count', 'sum'])
        ten_min = {0:ten_min}
        one_sec = df.groupby(pd.Grouper(freq='1s')).agg(['count', 'sum'])
        one_sec = {0:one_sec}
        #print(ten_min)
        return one_sec, ten_min
    '''
        
    def time_series(self, frequency, components=None):
        self.time_series_data = 0
        self.time_series_freq = frequency
        frequency_seconds = pd.to_timedelta(frequency).total_seconds()
        #print(frequency_seconds)
        if components == None: 
            self.component_bins = [(0, 500), (500, 1850), (1850, 2048), (0, 2048)]
            self.component_names = ['terrestrial', 'thunderstorm', 'cosmic', 'total']
        else:
            self.component_bins = components[0]
            self.component_names = components[1]

        lst = []
        for channel, value in self.data.items():
            (df, gps_ok, gps_not_ok) = value
            #print(df, gps_ok, gps_not_ok)
            df['utc'] = pd.to_datetime(df['TimeTag'] * 1_000_000_000, unit = 'ns')
            df.index = df['utc']
            dt1 = self.time_range[0]
            dt2 = self.time_range[1]
            mask = (df.index > dt1) & (df.index <= dt2)
            df = df.loc[mask]
            
            components = separate_spectra(df, 'Height', self.component_bins)

            #f = {'Height':[list_height, 'count', 'mean', 'min', 'max', 'sum', 'std'], 'Baseline':['mean', 'min', 'max', 'std']}
            f = {'Height':['count']}
            
            data_components = []
            for idx, component in enumerate(components):
                df_timeseries = component.groupby(pd.Grouper(freq=frequency)).agg(f)
                ''' DO NOT SIMPLIFY - IT WILL STOP WORKING. VALUES IN MULTICOLUMN WILL NOT CHANGE FOR SOME REASON '''
                height = df_timeseries['Height']
                normalized_counts = df_timeseries['Height']['count'].to_numpy() / float(frequency_seconds) # normalize counts
                height['count'] = normalized_counts
                df_timeseries['Height'] = height
                ''' END OF 'DO NOT CHANGE' SECTION '''
                data_components.append((self.component_names[idx], df_timeseries))
            data_components = dict(data_components)
            lst.append(data_components)
        data_zip = zip(self.channels, lst)
        data_dict = dict(data_zip)
        self.time_series_data = data_dict # dictionary with channel keys, value is a dictionary with component keys
        
        print('Time series data were created.')
        return data_dict, frequency
    
    def time_series_to_csv(self, filename):
        for channel in self.channels:
            df_out = pd.DataFrame()
            filename = filename + '_' + str(channel) + '.csv'
            df_list = self.time_series_data[channel]
            for component in self.component_names:
                df = df_list[component]['Height']
                df_out[component] = df['count']
                df['count'] = df['count'].round(2)
            print(df_out)
            df_out.to_csv(filename)
            
    def plot_gps_time_series(self, attr=None, stat=None, data=None, freq=None, component=None, file=None):
        title_dict = {'Height': 'Pulse height', 'Baseline':'Pulse baseline'}
        ylabel_dict = {'count': 'Counts ($s^{-1}$)', 'mean':'Channel (-)', 'min':'Channel (-)', 'max':'Channel (-)', 'sum':'Channel (-)', 'std':'Channel (-)'}
        attr_types = ['Height', 'Baseline']
        dic = {'Height':['list_height', 'count', 'mean', 'min', 'max', 'sum', 'std'], 'Baseline':['mean', 'min', 'max', 'std']}
        components_lst = ['terrestrial', 'cosmic', 'thunderstorm', 'total']
        
        if attr not in attr_types:
            raise ValueError("Invalid attr type. Expected one of: %s" % attr_types)
        if stat not in dic[attr]:
            raise ValueError("Invalid stat type. Expected one of: %s" % dic[attr]) 
        if component==None:
            component = components_lst
        
        if freq == None:
            frequency = self.time_series_freq
        else:
            frequency = freq
      
        for channel in self.channels:
            if data == None:
                dictn = self.time_series_data[channel]
            else:
                dictn = data[channel]
            
            colors = ['r', 'g', 'b', 'k']
            #smooth = gaussian_filter1d(df['Height']['count'], sigma=2, mode='wrap')
            fig = plt.figure(dpi=500)
            ax1 = fig.add_subplot(2, 1, 2)
            ax2 = ax1.twinx()
            ax3 = fig.add_subplot(2, 1, 1)
            maximum_1, maximum_2 = 0, 0
            minimum_1, minimum_2 = 1e+100, 1e+100
            #print(dictn)
            
            for files, tick, time, p in self.gps_ok:
                #print(time)
                time_utc = [datetime.datetime.fromtimestamp(int(ts)).strftime('%c') for ts in time]
                #time_utc = datetime.datetime.fromtimestamp(np.array(time))
                ax3.scatter(time, tick, s=1)
                
            for file, time in self.gps_not_ok:
                time_utc_0 = datetime.datetime.fromtimestamp(time[0]).strftime('%c')
                time_utc_1 = datetime.datetime.fromtimestamp(time[1]).strftime('%c')
                ax3.plot([time[0], time[1]], [0, 0])
                #print(self.gps_not_ok)
            
            for idx, key in enumerate(dictn.keys()):
                #print(dictn[key])
                if key in component:
                    df = dictn[key]
                    if key=='total' or key=='terrestrial':
                        if maximum_1 < max(df[attr][stat]):
                            maximum_1 = max(df[attr][stat])
                        if minimum_1 > min(df[attr][stat]):
                            minimum_1 = min(df[attr][stat])
                        style = {'total':'-', 'terrestrial':'-'}
                        colorpie = {'total':'k', 'terrestrial':'crimson'}
                        label = {'total':'total', 'terrestrial':'terrestrial < 3 MeV'}
                        #ax1.plot(df.index, df[attr][stat], linestyle=style[key], color='k', label=key)
                        ax1.step(df.index, df[attr][stat], linestyle=style[key], color=colorpie[key], label=label[key])                        
                    else:
                        if maximum_2 < max(df[attr][stat]):
                            maximum_2 = max(df[attr][stat])
                        if minimum_2 > min(df[attr][stat]):
                            minimum_2 = min(df[attr][stat])
                        style = {'thunderstorm':'-', 'cosmic':'-'}
                        colorpie = {'thunderstorm':'lime', 'cosmic':'g'}
                        label = {'thunderstorm':'thunderstorm 3-9 MeV', 'cosmic':'cosmic > 9 MeV'}
                        #ax2.plot(df.index, df[attr][stat], linestyle=style[key], color='gray', label=key)
                        #ax2.step(df.index, df[attr][stat], linestyle=style[key], color='gray', label=key)
                        ax2.step(df.index, df[attr][stat], linestyle=style[key], color=colorpie[key], label=label[key])

            ax1.set_ylim(bottom=0.7*minimum_1, top=1.05*maximum_1)
            ax2.set_ylim(bottom=0.9*minimum_2, top=1.4*maximum_2)
            ax1.set_title(title_dict[attr] + ' - ' + frequency + ' integration time')
            ax1.set_xlabel('UTC (YY/MM/DD hh:mm)')
            ax1.set_ylabel(ylabel_dict[stat])
            ax2.set_ylabel(ylabel_dict[stat])
            ax1.tick_params(axis='y', labelcolor='crimson')
            ax2.tick_params(axis='y', labelcolor='g')
            ax1.yaxis.label.set_color('crimson')
            ax2.yaxis.label.set_color('g') 
            ax1.legend(loc=2, fontsize=7)
            ax2.legend(loc=1,fontsize=7)
            
            
            div = int(len(df.index) / 20)
            if (div == 0):
                div = 1
            ax1.set_xticks(df.index[::div])
            ax1.set_xticklabels(df.index[::div], rotation=90)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d %H:%M"))
            plt.tight_layout()
            if file != None:
                file = file + '_' + str(channel) + '.png'
                #plt.savefig(file)
                
    def plot_time_series(self, attr=None, stat=None, data=None, freq=None, component=None, file=None):
        title_dict = {'Height': 'Pulse height', 'Baseline':'Pulse baseline'}
        ylabel_dict = {'count': 'Counts ($s^{-1}$)', 'mean':'Channel (-)', 'min':'Channel (-)', 'max':'Channel (-)', 'sum':'Channel (-)', 'std':'Channel (-)'}
        attr_types = ['Height', 'Baseline']
        dic = {'Height':['list_height', 'count', 'mean', 'min', 'max', 'sum', 'std'], 'Baseline':['mean', 'min', 'max', 'std']}
        components_lst = ['terrestrial', 'cosmic', 'thunderstorm', 'total']
        
        if attr not in attr_types:
            raise ValueError("Invalid attr type. Expected one of: %s" % attr_types)
        if stat not in dic[attr]:
            raise ValueError("Invalid stat type. Expected one of: %s" % dic[attr]) 
        if component==None:
            component = components_lst
        
        if freq == None:
            frequency = self.time_series_freq
        else:
            frequency = freq
      
        for channel in self.channels:
            if data == None:
                dictn = self.time_series_data[channel]
            else:
                dictn = data[channel]
            
            colors = ['r', 'g', 'b', 'k']
            #smooth = gaussian_filter1d(df['Height']['count'], sigma=2, mode='wrap')
            fig = plt.figure(dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            ax2 = ax1.twinx()
            maximum_1, maximum_2 = 0, 0
            minimum_1, minimum_2 = 1e+100, 1e+100
            
            for idx, key in enumerate(dictn.keys()):
                #print(dictn[key])
                if key in component:
                    df = dictn[key]
                    if key=='total' or key=='terrestrial':
                        if maximum_1 < max(df[attr][stat]):
                            maximum_1 = max(df[attr][stat])
                        if minimum_1 > min(df[attr][stat]):
                            minimum_1 = min(df[attr][stat])
                        style = {'total':'-', 'terrestrial':'-'}
                        colorpie = {'total':'k', 'terrestrial':'crimson'}
                        label = {'total':'total', 'terrestrial':'terrestrial < 3 MeV'}
                        #ax1.plot(df.index, df[attr][stat], linestyle=style[key], color='k', label=key)
                        ax1.step(df.index, df[attr][stat], linestyle=style[key], color=colorpie[key], label=label[key])                        
                    else:
                        if maximum_2 < max(df[attr][stat]):
                            maximum_2 = max(df[attr][stat])
                        if minimum_2 > min(df[attr][stat]):
                            minimum_2 = min(df[attr][stat])
                        style = {'thunderstorm':'-', 'cosmic':'-'}
                        colorpie = {'thunderstorm':'lime', 'cosmic':'g'}
                        label = {'thunderstorm':'thunderstorm 3-9 MeV', 'cosmic':'cosmic > 9 MeV'}
                        #ax2.plot(df.index, df[attr][stat], linestyle=style[key], color='gray', label=key)
                        #ax2.step(df.index, df[attr][stat], linestyle=style[key], color='gray', label=key)
                        ax2.step(df.index, df[attr][stat], linestyle=style[key], color=colorpie[key], label=label[key])

            ax1.set_ylim(bottom=0.7*minimum_1, top=1.05*maximum_1)
            ax2.set_ylim(bottom=0.9*minimum_2, top=1.4*maximum_2)
            ax1.set_title(title_dict[attr] + ' - ' + frequency + ' integration time')
            ax1.set_xlabel('UTC (YY/MM/DD hh:mm)')
            ax1.set_ylabel(ylabel_dict[stat])
            ax2.set_ylabel(ylabel_dict[stat])
            ax1.tick_params(axis='y', labelcolor='crimson')
            ax2.tick_params(axis='y', labelcolor='g')
            ax1.yaxis.label.set_color('crimson')
            ax2.yaxis.label.set_color('g') 
            ax1.legend(loc=2, fontsize=7)
            ax2.legend(loc=1,fontsize=7)
            
            
            div = int(len(df.index) / 20)
            if (div == 0):
                div = 1
            ax1.set_xticks(df.index[::div])
            ax1.set_xticklabels(df.index[::div], rotation=90)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d %H:%M"))
            plt.tight_layout()
            if file != None:
                file = file + '_' + str(channel) + '.png'
                plt.savefig(file)
            
    def plot_histogram_waterfall(self, data=None, file=None):
        for channel in self.channels:
            if data == None:
                dic = self.time_series_data[channel]
            else:
                dic = data[channel]

            for idx, comp in enumerate(self.component_names):
                (a, b) = self.component_bins[idx]
                bins = np.linspace(a, b, b-a+1)
                df = dic[comp]
                df_height = df['Height']
                temp = df_height['list_height'].apply(histogram, args=(bins,))
                dt = temp.to_numpy()
                dt = np.stack(dt)
                c = np.arange(0, len(temp.index))
                fig = plt.figure(dpi=500)
                ax1 = fig.add_subplot(1, 1, 1)
                im = ax1.pcolormesh(bins[:-1], c, dt, norm=colors.LogNorm())
                #ax1.pcolormesh(c, norm=bins[:-1], dt)
                #ax1.set_xticks(c, temp.index, rotation='vertical')
                div = int(len(c) / 20)
                if (div == 0):
                    div = 1
                ax1.set_title(comp)
                ax1.set_yticks(c[::div])
                ax1.set_yticklabels(temp.index[::div]) # rotation=90
                ax1.set_xlabel('Channel [-]')
                fig.colorbar(im)
                if file != None:
                    figname = file + '_' + comp + '_' + str(channel) + '.png'
                    plt.savefig(figname)
    
    def plot_histogram_difference(self, data=None, file=None):
        limit_1 = 500
        limit_2 = 1800
        bins = np.linspace(0, 2048, 2049)
        for channel in self.channels:
            if data == None:
                df = self.time_series_data[channel]
            else:
                df = data[channel]
            print(df)
            df_height = df['Height']
            temp = df_height['list_height'].apply(histogram, args=(bins,))
            hist = temp.to_numpy()
            hist_stack = np.stack(hist).T
            total = np.sum(hist_stack, axis=1)
            background = np.apply_along_axis(lambda x:total - x, 0, hist_stack)
            background_normed = np.apply_along_axis(norm, 0, background)
            histogram_normed = np.apply_along_axis(norm, 0, hist_stack)
            #print(background_normed, histogram_normed)
            result = histogram_normed - background_normed
            result = result[limit_1:limit_2, :]
            
            cosmic = result[limit_2:, :]
            thunderstorm = result[limit_1:limit_2, :]
            terrestrial = result[:limit_1, :]
            bins_cosmic = bins[limit_2:]
            bins_thunderstorm = bins[limit_1:limit_2, :]
            bins_terrestrial = bins[:limit_1, :]

            c = np.arange(0, len(temp.index))
            fig = plt.figure(dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            std = np.std(result)
            mean = np.mean(result)
            alpha = 3
            im1 = ax1.pcolormesh(c, bins[limit_1:limit_2], result, norm=colors.Normalize(vmin=mean-alpha*std, vmax=mean+alpha*std))
            #print(c, bins[limit:-1], result.shape)
            #im1 = ax1.pcolormesh(c, bins[limit:-1], result, norm=colors.LogNorm())
            #ax1.pcolormesh(c, norm=bins[:-1], dt)
            #ax1.set_xticks(c, temp.index, rotation='vertical')
            div = int(len(c) / 20)
            if (div == 0):
                div = 1
            ax1.set_xticks(c[::div])
            ax1.set_xticklabels(temp.index[::div], rotation=90)
            fig.colorbar(im1)
            if file != None:
                file = file + '_' + str(channel) + '.png'
                plt.savefig(file) 
                
    def plot_cwt(self, data=None, scales=None, file=None):
        for channel in self.channels:
            if data == None:
                dic = self.time_series_data[channel]
            else:
                dic = data[channel]
                
            for comp in self.component_names:
                df = dic[comp]
                df_max = df['Height']
                temp = df_max['count']

                fig = plt.figure(dpi=500)
                ax1 = fig.add_subplot(1, 1, 1)
                scales = np.linspace(1, 2500, 100)
                coeff, freq = pywt.cwt(temp, scales, 'mexh')
                print(coeff)
                print(len(coeff))
                coeff =  coeff[:, 9000:-9000]
                print(coeff)
                print(len(coeff))
                df = df.iloc[9000:-9000, :]
                print(df)
                ax1.imshow(abs(coeff), cmap='gray', aspect='auto', vmax=abs(coeff).max(), vmin=-abs(coeff).max())

                div = int(len(df.index) / 20)
                print(div)
                if (div == 0):
                    div = 1
                xtic = np.arange(0, len(df.index), div)
                ax1.set_xticks(xtic)
                ax1.set_xticklabels(df.index[::div], rotation=90)

                if file != None:
                    figname = file + '_' + comp + '_' + str(channel) + '.png'
                    plt.savefig(figname)

                a = hessian(abs(coeff))
                fig = plt.figure(dpi=500)
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.imshow(a, aspect='auto')
                ax1.set_xticks(xtic)
                ax1.set_xticklabels(df.index[::div], rotation=90)
                if file != None:
                    figname = file + '_hessian_' + comp + '_' + str(channel) + '.png'
                    plt.savefig(figname)

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
                
def separate_spectra(df, attr, bins):
    output = []
    for left, right in bins:
        mask = (df[attr] >= left) & (df[attr] < right)
        temp = df.loc[mask]
        output.append(temp)
    return output
                
def histogram(x, bins, normed=False):
    hist, _ = np.histogram(x, bins=bins, density=normed)
    return hist

def norm(x):
    return x / np.sum(x)

def hist_diff(hist, total):
    tot = total - hist
    a = norm(tot)
    b = norm(hist)
    return a - b

def detect_ridges(gray, sigma=3.0):
    hxx, hyy, hxy = hessian_matrix(gray, sigma)
    i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    return i1, i2

def extract_data(data, adc_channel):
    data_mask_channel = data[(data["boardIndexAndChannel"]==adc_channel)]
    if (len(data_mask_channel) != 0):
        #print(unixTime)
        #unixTime = check_loop(unixTime, 2.0**40)
        #print(time_standard[0], time_standard[1], clock, (unixTime - time_standard[1]) / clock)
        #unixTime = time_standard[0] + (unixTime - time_standard[1]) / clock
        #print(unixTime)
        #print(pd.to_datetime(unixTime * 1000000000, unit='ns'))
        
        timeTag = np.array(data_mask_channel["timeTag"], np.float64)
        diff = np.diff(timeTag)
        a = np.where(diff < 0)[0]
        '''
        fig = plt.figure(dpi=500)
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(timeTag, label=a)
        ax.legend()
        '''
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
        triggerCountShift = np.roll(triggerCount, 1)
        deadCount = triggerCount - triggerCountShift - 1
        deadCount[0] = 0
        data = np.stack([timeTag, phaMax, phaMaxTime, phaMin, phaFirst, phaLast, maxDerivative, baseline, deadCount], axis=1)
        df = pd.DataFrame(data, columns=['TimeTag', 'Height', 'MaxTime', 'Min', 'First', 'Last', 'maxDerivative', 'Baseline', 'DeadCount'])
        #print(df)
        if len(a) == 1:
            df1 = df.iloc[:a[0]+1,:]
            df2 = df.iloc[a[0]+1:,:]
            #print(df1, df2)
            df = [df1, df2]
            #print(df1, df2)
        elif len(a) == 0:
            df = [df]
        else:
            print('Data ticks have more then 1 negative derivation.')
            
        #df['UTC'] = pd.to_datetime(df['Unix'] * 1000000000, unit='ns')
        return df
    else:
        return np.array([0])

'''
def extract_data(adc_channel, event, time_standard):
    clock = time_standard[2]
    data_mask_channel = event[(event["boardIndexAndChannel"]==adc_channel)]
    #print(data_mask_channel)
    if (len(data_mask_channel) != 0):
        unixTime = np.array(data_mask_channel["timeTag"], np.float64)
        #print(unixTime)
        unixTime = check_loop(unixTime, 2.0**40)
        #print(time_standard[0], time_standard[1], clock, (unixTime - time_standard[1]) / clock)
        unixTime = time_standard[0] + (unixTime - time_standard[1]) / clock
        #print(unixTime)
        #print(pd.to_datetime(unixTime * 1000000000, unit='ns'))

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
        #triggerCount = check_loop(triggerCount, 2**16)
        triggerCountShift = np.roll(triggerCount, 1)
        deadCount = triggerCount - triggerCountShift - 1
        deadCount[0] = 0
        data = np.stack([unixTime, phaMax, phaMaxTime, phaMin, phaFirst, phaLast, maxDerivative, baseline, deadCount], axis=1)
        df = pd.DataFrame(data, columns=['Unix', 'Height', 'MaxTime', 'Min', 'First', 'Last', 'maxDerivative', 'Baseline', 'DeadCount'])
        df['UTC'] = pd.to_datetime(df['Unix'] * 1000000000, unit='ns')
        return df
    else:
        return np.array([0])
'''

def check_loop(array, maximum):
    if (array[0] > array[array.size-1]):
        noloop = array[array>=array[0]]
        loop = array[array<array[0]]
        loop += maximum
        narray = np.concatenate([noloop, loop])
        #print('true')
        return narray
    else:
        #print('false')
        return array

def gps_base_time(gps):
    #print(int(gps[0][0])&0xFFFFFFFFFF)
    time_tag_base = int(gps[0][0])&0xFFFFFFFFFF
    #print('time_tag_base:', time_tag_base)
    unixtime_base = float(gps[0][1])
    gps_string_base = gps[0][2][8:14]
    #print('gps_base_time: ', time_tag_base, unixtime_base, gps_string_base)
    time_obj_unixtime = datetime.datetime.fromtimestamp(unixtime_base, datetime.timezone.utc)
    time_str_gps = time_obj_unixtime.strftime("%Y%m%d")+" "+gps_string_base+"+00:00"
    time_obj_precise = datetime.datetime.strptime(time_str_gps, "%Y%m%d %H%M%S%z")+datetime.timedelta(seconds=1.0)
    delta_time_gps = time_obj_unixtime - time_obj_precise
    if (delta_time_gps.total_seconds() > 12.0 * 3600.0):
        time_obj_precise = time_obj_precise+datetime.timedelta(days=1.0)
    elif (delta_time_gps.total_seconds() < -12.0 * 3600.0):
        time_obj_precise = time_obj_precise - datetime.timedelta(days=1.0)
        
    #unixtime_precise = time_obj_precise.timestamp() # original Yuuki solution - I have no idea what is time_obj_precise
    unixtime_precise = time_obj_unixtime.timestamp()
    clock = clock_verification(gps)
    time_standard = [unixtime_base, time_tag_base, clock]
    return time_standard

def non_gps_base_time(filename):
    time_str_file = (os.path.basename(filename).split('.', 1)[0])+"+00:00"
    time_obj_file = datetime.datetime.strptime(time_str_file, "%Y%m%d_%H%M%S%z")
    #print(time_obj_file, time_obj_file.timestamp())
    #time_obj_gps = datetime.datetime.fromtimestamp(float(gps[0][1]), datetime.timezone.utc)
    #delta_time_obj = time_obj_gps - time_obj_file
    
    #time_lag_hour = round((delta_time_obj.total_seconds()) / 3600.0)
    #time_obj_file = time_obj_file + datetime.timedelta(hours=time_lag_hour)
    #time_standard = [time_obj_file.timestamp(), int(event[1]), 1.0e8]
    return time_obj_file.timestamp()

def gps_status_test(gps):
    string = gps[0][2]
    if (string=="GP") or (string=="NULL"):
        gps_status = False
    else:
        gps_status = True
        
    #print(gps[:][0])
    timetags = list(zip(*gps))[0]
    gps_set = set(timetags)
    #len(gps_set)
    if len(gps_set) <= 1 or len(gps) < 4:
        gps_status = False
    return gps_status

def clock_verification(gps):
    clock = 1.0e8
    time_tag = np.array([gps[0][0]], dtype="int64")
    for i in range(1, len(gps)):
        if (gps[i][2] != "NULL") and (gps[i][2]!=""):
            time_tag = np.append(time_tag, gps[i][0])
    time_tag = time_tag&0xFFFFFFFFFF
    #time_tag = check_loop(time_tag, 2**40)
    if (time_tag.size < 2):
        return clock
    else:
        return clock
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

def remove_untuned_ticks(ticks, time):
    # remove out of tune ticks - ticks that do not fit in the tick frequency
    tick_frequency = 100000000

    diff_ticks = np.diff(ticks)
    diff_time = np.diff(time)
    diff_division = diff_ticks / diff_time
    
    outlayers = np.where(abs(diff_division - tick_frequency) >= 1000)[0]
    diff_division = np.delete(diff_division, outlayers)
    gps_ticks = np.delete(ticks, outlayers)
    gps_time = np.delete(time, outlayers)
    return gps_ticks, gps_time

def remove_outlayer_ticks(data):
    out = []
    for block in data:
        (files, y, x, p) = block
        #print(files, y, x, p)
        y_pred = p[0] * x + p[1]
        residue = np.abs(y - y_pred) / y_pred
        
        a = np.where(residue > 0.01)[0]
        if len(a) > 0:
            y = np.delete(y, a)
            x = np.delete(x, a)
        out.append((files, y, x, p))
    return out

def data_process(gps_ok, gps_not_ok, channel):
    df_data = pd.DataFrame()
    time_delay, aux_lst = [], []
    for files, ticks, time, p in gps_ok:
        print(files)
        for idx, file in enumerate(files):
            #print(idx, file)
            df_lst = read_fits(file, channel)
            if len(df_lst) == 2: # discontinuity in data timetags
                #print(file, idx, len(df_lst))
                if idx == 0:
                    df = recalculate_time(df_lst[1], p) # the beggining of the data block - range of the counter
                    aux_lst.append([file, 1])
                else:
                    df = recalculate_time(df_lst[0], p) # the end of the data block - range of the counter
                    aux_lst.append([file, 0])
                #print(aux_lst)
            elif len(df_lst) == 1:
                df = recalculate_time(df_lst[0], p)
            else:
                print('Shit happens...')
            first_time = datetime.datetime.fromtimestamp(df.iloc[0, 0]).strftime('%c')
            last_time = datetime.datetime.fromtimestamp(df.iloc[0-1, 0]).strftime('%c')
            print('good gps', file, first_time, last_time, idx, len(df_lst), len(files))
            df_data = pd.concat([df_data, df], ignore_index=True)
            
            time, _ = file2datetime(file)
            time_file = time.timestamp()
            time_delay.append(int(df.iloc[0, 0] - time_file))
            
    #print('time_delay ', time_delay)
    if len(time_delay) == 0:
        time_delay = 0
    else:
        time_delay = max(set(time_delay), key = time_delay.count)
    #print('the most frequent value ', time_delay)
        
    end_list = []
    for file, time in gps_not_ok:
        df_lst = read_fits(file, channel)
        if len(df_lst) == 2: # discontinuity in data timetags
            #print(df_lst)
            df = pd.concat([df_lst[0], df_lst[1]], ignore_index=True)
            len_1 = df_lst[0].shape[0]
            df = assign_time(df, file, time_delay) # the end of the data block - range of the counter
            df_lst[0] = df.iloc[:len_1, :]
            df_lst[1] = df.iloc[len_1+1:, :]
            print(df_lst[0].iloc[-1, 0], df_lst[1].iloc[0, 0])
            if [file, 1] in aux_lst:
                df = df_lst[1]
            elif [file, 0] in aux_lst:
                df = df_lst[0]
        elif len(df_lst) == 1: # no discontinuity in data timetags
            df = assign_time(df_lst[0], file, time_delay)
        else:
            print('Shit happens...')
        
        end_list.append(df.iloc[-1, 0])
        #df_data = pd.concat([df_data, df], ignore_index=True)
        
    df_data.sort_values('TimeTag')
    # df_data = sort
    return df_data, end_list

def gps_process(filelist):
    gps_ok, gps_not_ok, gps_missing = read_gps_from_fits(filelist)
    #print('before gps process', len(gps_ok), len(gps_not_ok))
    gps_ok, not_validated_gps = process_ok_gps(gps_ok) # gps_ok is a list of tuples ([files], [ticks], [time], [p])
    #print(not_validated_gps)
    plot_gps(gps_ok)
    for file, ticks, time, p in gps_ok:
        start_time = datetime.datetime.fromtimestamp(time[0]).strftime('%c')
        end_time = datetime.datetime.fromtimestamp(time[-1]).strftime('%c')
        print(file, start_time, end_time, p)
    gps_not_ok.extend(not_validated_gps)
    #print('gps ok', gps_ok)
    print('gps not ok', gps_not_ok)
    #print(gps_not_ok, not_validated_gps)
    #print(type(not_validated_gps))
    #print(gps_not_ok)
    gps_not_ok = process_not_ok_gps(gps_not_ok) # gps_not_ok is a list of tuples (file, epoch_time)
    #print(gps_ok, gps_not_ok)
    for file, time in gps_not_ok:
        start_time = datetime.datetime.fromtimestamp(time).strftime('%c')
        print(file, start_time)
    #print(gps_not_ok)
    return gps_ok, gps_not_ok

def read_gps_from_fits(filelist):
    # Reads GPS data from .fits files
    gps_ok, gps_not_ok, gps_missing  = [], [], []
    for file in filelist:
        fits_file = fitsio.open(file)
        
        try:
            gps = fits_file[2].data
        except:
            gps_missing.append(file)
            continue
        gps = list(filter(lambda i: i[0] != 0, gps))

        gps_status = gps_status_test(gps)
        if gps_status == False:
            #print(file, "true")
            gps_not_ok.append(file)
        else:
            #print(file, "false")
            gps = np.array(gps)
            gps_ticks = gps[:, 0].astype(int)&0xFFFFFFFFFF
            gps_time = gps[:, 1].astype(int)
            #print(gps_time)
            gps_ok.append((file, gps_ticks, gps_time))
    return gps_ok, gps_not_ok, gps_missing

def find_gps_function(data):
    out = []
    for files, ticks, time, k in data:
        #print(files, ticks, time)
        p = np.polyfit(time, ticks, 1)
        #print(p)
        out.append((files, ticks, time, p))
    return out

def separate_gps_data(data):
    out, tup = [], []
    last_tick = 0
    for file, gps_ticks, gps_time in data:
        first_tick = gps_ticks[0]
        #print(file, gps_ticks, gps_time)
        #print(gps_time[0] - last_gps)
        diff = np.diff(gps_ticks)
        a = np.where(diff < 0)[0]
        if len(a) == 1:
            ticks = [gps_ticks[0:a[0]+1], gps_ticks[a[0]+1:]]
            time = [gps_time[0:a[0]+1], gps_time[a[0]+1:]]
            tup.append((file, ticks[0], time[0]))
            out.append(tup)
            tup = [(file, ticks[1], time[1])]
        elif len(a) == 0 and first_tick < last_tick:
            print(first_tick, last_tick)
            out.append(tup)
            tup = [(file, gps_ticks, gps_time)]
        elif len(a) == 0:
            tup.append((file, gps_ticks, gps_time))
        else:
            print('GPS ticks have more then 1 negative derivation.')
        last_tick = gps_ticks[-1]
    #print(len(out))
    
    fig = plt.figure(dpi=500)
    ax1 = fig.add_subplot(1, 1, 1)
    for file, ticks, time in out[0]:
        ax1.scatter(time, ticks, s=1)
    
    return out

def correct_gps_data(data):
    out = []
    for block in data:
        x, y, files = [], [], []
        for file, ticks, time in block:
            files.append(file)
            x.extend(time)
            y.extend(ticks)
        #print(x, y)
        ticks, time = remove_untuned_ticks(y, x)
        out.append((files, ticks, time, []))
    return out

def process_ok_gps(data):
    gps_ok = separate_gps_data(data)
    gps_ok = correct_gps_data(gps_ok)
    gps_ok = find_gps_function(gps_ok)
    gps_ok = remove_outlayer_ticks(gps_ok)
    gps_ok = find_gps_function(gps_ok)
    #files, _, _, _ = list(zip(*gps_ok))
    gps_ok, gps_not_ok = check_validity_of_gps(gps_ok)

    if gps_not_ok != []:
        gps_not_ok = gps_not_ok[0]
    return gps_ok, gps_not_ok

def process_not_ok_gps(data):
    out = []
    for file in data:
        unix = non_gps_base_time(file)
        out.append((file, unix))
        #print(unix)
    return out

def plot_gps(data):
    for files, ticks, time, p in data:
        fig = plt.figure(dpi=500)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.scatter(time, ticks, s=2)
        x_min = np.min(time)
        x_max = np.max(time)
        x = np.linspace(x_min, x_max, 3)
        ax1.plot(x, p[0] * x + p[1], 'r', linewidth=1)

def file2datetime(file):
    basename = os.path.basename(file)
    basename = os.path.splitext(basename)[0]
    basename = os.path.splitext(basename)[0]
    dt = str2datetime(basename)
    return dt, file

def read_fits(file, channel):
    fits_file = fitsio.open(file)
    data = fits_file[1].data
    df_lst = extract_data(data, channel)
    return df_lst

def recalculate_time(df, func):
    #print(func)
    df['TimeTag'] = (df['TimeTag'] - func[1]) / func[0]
    return df

def assign_time(df, file, delay):
    time, _ = file2datetime(file)
    #print(file)
    epoch = time.timestamp() + delay
    initial_tick = df['TimeTag'][0]
    #print(df['TimeTag'])
    df['TimeTag'] = (df['TimeTag'] - initial_tick) * 1E-08
    df['TimeTag'] = df['TimeTag'] + epoch
    
    first_time = df.iloc[0, 0]
    last_time = df.iloc[-1, 0]
    
    #print('bad gps ', file, last_time - first_time, datetime.datetime.fromtimestamp(first_time).strftime('%c'), datetime.datetime.fromtimestamp(last_time).strftime('%c'))
    return df

def merge_gps_not_ok(gps_not_ok, end_list):
    out = []
    for idx, (file, time) in enumerate(gps_not_ok):
        out.append((file, [time, end_list[idx]]))
    return out

def check_validity_of_gps(gps):
    #print(gps)
    out, out_comp = [], []
    for file, ticks, time, p in gps:
        
        if len(ticks) >= 5:
            #print('gps ok', file)
            out.append((file, ticks, time, p))
            #print(file, len(ticks))
        else:
            #print('gps not ok', file)
            out_comp.append(file)
            
    #print(len(out), len(out_comp))
    return out, out_comp