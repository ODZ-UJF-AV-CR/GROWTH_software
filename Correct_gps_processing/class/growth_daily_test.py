import GROWTH as gr
import importlib
import sys
import argparse
import os
from datetime import date, datetime, timedelta

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-g", "--growth", dest="growth", help='Select growth: GR1 (Musala), GR2 (Jungfraujoch), GR3 (Zugspitze)')
parser.add_argument("-o", "--output", dest="output", help='Path for output figures')
args = parser.parse_args()

today = date.today()
three_before = today - timedelta(days=3)
two_before = today - timedelta(days=2)
one_before = today - timedelta(days=1)
t1 = three_before.strftime("%Y%m%d")
day = two_before.strftime("%Y-%m-%d")
t2 = one_before.strftime("%Y%m%d")
appendix = '_120000'
time = (t1 + appendix, t2 + appendix) # tuple (YYYYMMDD_HHMMSS, YYYYMMDD_HHMMSS), to open more than 72 hours is not recommanded (it is going to take ages and it might crush)
print(args.growth, args.output, day, t1, t2)
path = os.path.join(args.output, day)
os.mkdir(path)

GR = args.growth
components = ([(0, 500), (500, 1800), (1800, 2048), (0, 2048)], ['terrestrial', 'thunderstorm', 'cosmic', 'total']) # separates the energy spectrum into components list(tuple(left, right)), names[string])

foo = gr.GROWTH(GR) # Initializes an instance of GROWTH class
_ = foo.go_through_files(time) # Selects the data files of interest - please read manual to this function (function is "overloaded")
_ = foo.read_fits()

data_coarse, coarse_freq = foo.time_series('10min', components=components)
figname = os.path.join(path, day + '_' + coarse_freq + '_timeseries')
foo.plot_time_series(attr='Height', stat='count', data=data_coarse, freq=coarse_freq, file=figname)
figname = os.path.join(path, day + '_' + coarse_freq + '_hist')
foo.plot_histogram_waterfall(data=data_coarse, file=figname)

data_fine, fine_freq = foo.time_series('1min', components=components)
figname = os.path.join(path, day + '_' + fine_freq + '_timeseries')
foo.plot_time_series(attr='Height', stat='count', data=data_fine, freq=fine_freq, component=['total', 'thunderstorm'], file=figname)

