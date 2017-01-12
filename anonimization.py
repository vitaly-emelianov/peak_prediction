import pandas as pd

data = pd.read_csv('./data.csv')
data.REPORT_DATE = pd.to_datetime(data.REPORT_DATE, format='%d.%m.%Y')

fake_ids = {}
counter = 0
for branch_id in data.BRANCH_ID.unique():
    fake_ids[branch_id] = counter * 1000
    counter += 1

# Anonimising branch ids
data['FAKE_ID'] = data["BRANCH_ID"].apply(lambda x: fake_ids[x])
data = data.drop(axis=1, labels=["BRANCH_ID", ])
data = data.rename(index=str, columns={"FAKE_ID": "BRANCH_ID", })

# Anonimising number of clients registered
data["CLIENTS_REGGED"] = data["CLIENTS_REGGED"].apply(lambda x: x * 3)
data.to_csv("data.csv", index=False)
