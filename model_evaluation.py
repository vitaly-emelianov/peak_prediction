import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import seaborn

data = pd.read_csv('./features.csv')
data.REPORT_DATE = pd.to_datetime(data.REPORT_DATE)

months = data.REPORT_DATE.map(lambda x: x.month)
years = data.REPORT_DATE.map(lambda x: x.year)

# Splitting data to train and test part
month, year = 4, 2015
train_mask = ((months < month) & (years == year)) | (years < year)
train_data = data.loc[train_mask, :]
test_data = data.loc[~train_mask, :]

# Saving all peaks in a dictionary
all_peaks = {}
for item in data.itertuples():
    date_in_train = (item.REPORT_DATE.year < year) or (item.REPORT_DATE.year == year and item.REPORT_DATE.month < month)
    if date_in_train and item.IS_PEAK:
        if item.BRANCH_ID not in all_peaks:
            all_peaks[item.BRANCH_ID] = {item.REPORT_DATE}
        else:
            all_peaks[item.BRANCH_ID].add(item.REPORT_DATE)
