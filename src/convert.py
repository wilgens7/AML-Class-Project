
""" This file adds labels to the two csv files, training and testing and combines them into one
dataframe, then outputs them in a csv file."""

import pandas as pd

# Dataset field names
datacols = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]

# Load NSL_KDD train dataset
dfkdd_train = pd.read_table('train_data.txt', sep=",", names=datacols) # change path to where the dataset is located.


# Load NSL_KDD test dataset
dfkdd_test = pd.read_table('test_data.txt', sep=",", names=datacols)

a=len(dfkdd_train)
b=len(dfkdd_test)

print((a/(a+b))) #percentage of train data
print(a)
print(b)

print(len(dfkdd_train)+len(dfkdd_test))
print(len(pd.concat([dfkdd_train,dfkdd_test],axis = 'rows')))

combined = pd.concat([dfkdd_train,dfkdd_test],axis = 'rows')



#combined.to_csv(r'data.csv',index = False)
#print(len(pd.read_csv('data.csv')))

data = pd.read_csv('data.csv')

print(data[data.isna()].count())

