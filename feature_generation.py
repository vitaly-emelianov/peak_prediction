import pandas as pd
import datetime as dt


def is_weekend(date):
    """Check whether it's a weekend."""
    return date.weekday() in {5, 6}


def is_holiday(date):
    """Check whether this day is a holiday in Russia."""
    return (date.day, date.month) in {(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1),
                                      (23, 2), (8, 3), (1, 5), (9, 5), (12, 6), (1, 9), (4, 11)
                                      }


# Reading data
data = pd.read_csv('./data.csv')
data.REPORT_DATE = pd.to_datetime(data.REPORT_DATE, format='%d.%m.%Y')

# Generating features
branch_ids = data.BRANCH_ID.unique()
all_months = data.REPORT_DATE.map(lambda x: x.month)
all_years = data.REPORT_DATE.map(lambda x: x.year)

for i, branch_id in enumerate(branch_ids):
    if i % 10 == 0:
        print i,
    branch_mask = (data.BRANCH_ID == branch_id)
    data_branch = data.loc[branch_mask, :]
    months = data_branch.REPORT_DATE.map(lambda x: x.month)
    years = data_branch.REPORT_DATE.map(lambda x: x.year)
    for month, year in set(zip(months, years)):
        try:
            month_mask = (months == month) & (years == year)
            data_branch_month = data_branch.loc[month_mask, :]
            data_branch_month = data_branch_month.sort_values(by=['CLIENTS_REGGED'], ascending=False)
            peaks = set(data_branch_month.REPORT_DATE[:6])
            min_peak_value = data_branch_month.CLIENTS_REGGED.values[5]
            soft_peaks = data_branch_month.CLIENTS_REGGED.map(lambda x: int(x >= min_peak_value * 0.9))
            data.loc[(branch_mask) & (all_years == year) & (all_months == month), 'SOFT_PEAK'] = soft_peaks
            is_peak = data_branch_month.REPORT_DATE.map(lambda x: int(x in peaks))
            data.loc[(branch_mask) & (all_years == year) & (all_months == month), 'IS_PEAK'] = is_peak
        except:
            print month, year, branch_id

data['DAY'] = data.REPORT_DATE.map(lambda x: x.day)
data['WEEKDAY'] = data.REPORT_DATE.map(lambda x: x.weekday())
data['IS_WEEKEND'] = data.REPORT_DATE.map(lambda x: int(is_weekend(x)))

data['MONTH'] = all_months
data['IS_HOLIDAY'] = data.REPORT_DATE.map(lambda x: int(is_holiday(x)))
data['ONE_BEFORE_HOLIDAY'] = data.REPORT_DATE.map(lambda date: int(is_holiday(date + dt.timedelta(days=1))))
data['TWO_BEFORE_HOLIDAY'] = data.REPORT_DATE.map(lambda date: int(is_holiday(date + dt.timedelta(days=2))))
data['THREE_BEFORE_HOLIDAY'] = data.REPORT_DATE.map(lambda date: int(is_holiday(date + dt.timedelta(days=3))))
data['ONE_AFTER_HOLIDAY'] = data.REPORT_DATE.map(lambda date: int(is_holiday(date - dt.timedelta(days=1))))
data['TWO_AFTER_HOLIDAY'] = data.REPORT_DATE.map(lambda date: int(is_holiday(date - dt.timedelta(days=2))))
data['THREE_AFTER_HOLIDAY'] = data.REPORT_DATE.map(lambda date: int(is_holiday(date - dt.timedelta(days=3))))

# Saving features to file
data.to_csv("features.csv", index=False)
