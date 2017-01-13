import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def get_labels_from_probs(prob_pred, number_of_peaks=6):
    """Return labels given probabilities."""
    y_pred = [0] * prob_pred.shape[0]
    indices = set(prob_pred.argsort()[::-1][0:number_of_peaks])
    for i, val in enumerate(y_pred):
        if i in indices:
            y_pred[i] = 1
    return y_pred


def get_number_of_peak_match(y_pred, y_test):
    counter = 0
    for predicted, real in zip(y_pred, y_test):
        if predicted == 1 and real == 1:
            counter += 1
    return counter


def train_test_split(data, month, year):
    """Divide data to train and test."""
    months = data.REPORT_DATE.map(lambda x: x.month)
    years = data.REPORT_DATE.map(lambda x: x.year)
    test_mask = (months == month) & (years == year)
    train_mask = (months < month) & (years == year) | (years < year)
    return data.loc[train_mask], data.loc[test_mask]


def test_model(model, data, month, year):
    """Test classifier on last month."""
    train, test = train_test_split(data, month, year)
    branch_ids = test.BRANCH_ID.unique()
    X_train = train.drop(axis=1, labels=['REPORT_DATE', 'BRANCH_ID', 'IS_PEAK', 'SOFT_PEAK'])
    y_train = train['SOFT_PEAK']
    model.fit(X_train, y_train)
    month_counts = []
    for i, branch_id in enumerate(branch_ids):
        test_for_branch = test.loc[test.BRANCH_ID == branch_id, :]
        X_test = test_for_branch.drop(axis=1, labels=['REPORT_DATE', 'BRANCH_ID', 'IS_PEAK', 'SOFT_PEAK'])
        y_test = test_for_branch['SOFT_PEAK']
        probs_pred = model.predict_proba(X_test)[:, 1]
        y_pred = get_labels_from_probs(probs_pred)
        month_counts.append(get_number_of_peak_match(y_pred, y_test))
    return np.mean(month_counts)


def get_average_of_correct_guessed_peaks(model, data, start=12):
    """Get average of predicted peaks for month by starting one."""
    months = data.REPORT_DATE.map(lambda x: x.month)
    years = data.REPORT_DATE.map(lambda x: x.year)
    chunks = pd.unique(zip(years, months))
    chunks.sort()
    averages = []

    for year, month in chunks[start:]:
        avgs = test_model(model, data, month, year)
        averages.append(avgs)
        print month, year, avgs
    return pd.Series(data=averages, index=chunks[start:])


# Reading generated features
data = pd.read_csv('./features.csv')
data.REPORT_DATE = pd.to_datetime(data.REPORT_DATE)

months = data.REPORT_DATE.map(lambda x: x.month)
years = data.REPORT_DATE.map(lambda x: x.year)

# Splitting data to train and test part
month, year = 1, 2015
train_mask = ((months < month) & (years == year)) | (years < year)

# Saving all peaks in a dictionary
all_peaks = {}
for item in data.itertuples():
    date_in_train = (item.REPORT_DATE.year < year) or \
                    (item.REPORT_DATE.year == year and item.REPORT_DATE.month < month)
    if date_in_train and item.IS_PEAK:
        if item.BRANCH_ID not in all_peaks:
            all_peaks[item.BRANCH_ID] = {item.REPORT_DATE}
        else:
            all_peaks[item.BRANCH_ID].add(item.REPORT_DATE)

# Calculating conditional probabilities
weekdays = data.REPORT_DATE.map(lambda x: x.weekday())
branch_ids = data.BRANCH_ID.unique()

for weekday in weekdays.unique():
    print weekday,
    weekday_mask = (weekdays == weekday)
    counter = 0

    for branch_id in branch_ids:
        branch_mask = (data.BRANCH_ID == branch_id)
        peaks = all_peaks[branch_id]
        branch_counter = 0
        for date in peaks:
            if date.weekday() == weekday:
                counter += 1
                branch_counter += 1
        branch_norm_factor = (weekday_mask & branch_mask & train_mask).sum()
        if branch_norm_factor > 0 and weekday != 6:
            data.loc[branch_mask & weekday_mask, 'WB_PEAK'] = branch_counter / float(branch_norm_factor)
        else:
            data.loc[branch_mask & weekday_mask, 'WB_PEAK'] = 0
    all_norm_factor = (train_mask & weekday_mask).sum()

weekdays = data.REPORT_DATE.map(lambda x: x.weekday())
months = data.REPORT_DATE.map(lambda x: x.month)
branch_ids = data.BRANCH_ID.unique()

for weekday, month in set(zip(weekdays, months)):
    print weekday, ':', month,
    weekday_month_mask = (weekdays == weekday) & (months == month)
    counter = 0
    for branch_id in branch_ids:
        peaks = all_peaks[branch_id]
        for date in peaks:
            if date.weekday() == weekday and date.month == month:
                counter += 1
    all_norm_factor = (train_mask & weekday_month_mask).sum()
    if all_norm_factor > 0 and weekday != 6:
        data.loc[weekday_month_mask, 'WM_PEAK'] = counter / float(all_norm_factor)
    else:
        data.loc[weekday_month_mask, 'WM_PEAK'] = 0

# Using one-hot encoding to represent day, month and weekday column values.
data = pd.get_dummies(data, columns=['DAY', 'MONTH', 'WEEKDAY'])

# Evaluating logistic regression model
params = {'C': 1.0, 'penalty': 'l2'}
clf = LogisticRegression(**params)
avgs = get_average_of_correct_guessed_peaks(clf, data, start=11)
