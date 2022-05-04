import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import pickle
pd.set_option('display.max_columns', None)

df_insulin = pd.read_csv('InsulinData.csv', low_memory=False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
df_CGM = pd.read_csv('CGMData.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
# print(df_insulin)

df_insulin_p2 = pd.read_csv('Insulin_patient2.csv', low_memory=False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
df_CGM_p2 = pd.read_csv('CGM_patient2.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])


df_insulin['timestamp']=pd.to_datetime(df_insulin['Date'] + ' ' + df_insulin['Time'])
df_CGM['timestamp']=pd.to_datetime(df_CGM['Date'] + ' ' + df_CGM['Time'])
# print(df_insulin)

df_insulin_p2['timestamp']=pd.to_datetime(df_insulin_p2['Date'] + ' ' + df_insulin_p2['Time'])
df_CGM_p2['timestamp']=pd.to_datetime(df_CGM_p2['Date'] + ' ' + df_CGM_p2['Time'])
# print(df_insulin_p2)


def find_meal_data(df_insulin, df_CGM, set_num):
    insulin = df_insulin.copy()
    insulin = insulin.set_index('timestamp')
    insulin = insulin.sort_values(by='timestamp', ascending=True).dropna().reset_index()
    insulin['BWZ Carb Input (grams)'].replace(0.0, np.nan, inplace=True)
    insulin = insulin.dropna().reset_index()
    valid_list = []
    # print(insulin)
    for idx, i in enumerate(insulin['timestamp']):
        if idx == insulin.shape[0] - 1:
            valid_list.append(i)
            break
        gap = (insulin['timestamp'][idx + 1] - i).seconds / 60.0
        if gap > 120:
            valid_list.append(i)
    # print(valid_list)
    CGM_list = []
    for idx, i in enumerate(valid_list):
        start = pd.to_datetime(i - timedelta(minutes=30))
        end = pd.to_datetime(i + timedelta(minutes=120))
        if set_num == 1:
            curr_date = i.date().strftime('%#m/%#d/%Y')
            CGM_list.append(df_CGM.loc[df_CGM['Date'] == curr_date].set_index('timestamp').between_time(
                start_time=start.strftime('%#H:%#M:%#S'), end_time=end.strftime('%#H:%#M:%#S'))[
                            'Sensor Glucose (mg/dL)'].values.tolist())
        else:
            curr_date = i.date().strftime('%Y-%m-%d')
            CGM_list.append(df_CGM.loc[df_CGM['Date'] == curr_date].set_index('timestamp').between_time(
                start_time=start.strftime('%H:%M:%S'), end_time=end.strftime('%H:%M:%S'))[
                                'Sensor Glucose (mg/dL)'].values.tolist())
    return pd.DataFrame(CGM_list)


meal_data = find_meal_data(df_insulin, df_CGM, 1)
meal_data= meal_data.iloc[:, 0:30]
# print(meal_data)

meal_data_p2 = find_meal_data(df_insulin_p2, df_CGM_p2, 2)
meal_data_p2 = meal_data_p2.iloc[:, 0:30]
# print(meal_data_p2)


def find_no_meal_data(df_insulin, df_CGM, set_num):
    insulin = df_insulin.copy()
    insulin = insulin.set_index('timestamp')
    insulin = insulin.sort_values(by='timestamp', ascending=True).dropna().reset_index()
    insulin['BWZ Carb Input (grams)'].replace(0.0, np.nan, inplace=True)
    insulin = insulin.dropna().reset_index()
    valid_list = []
    for idx, i in enumerate(insulin['timestamp']):
        if idx == insulin.shape[0] - 1:
            valid_list.append(i)
            break
        gap = (insulin['timestamp'][idx + 1] - i).seconds / 3600
        i += pd.Timedelta(hours=2)
        if gap > 4:
            while (i + pd.Timedelta(hours=2)) < insulin['timestamp'][idx + 1]:
                valid_list.append(i)
                i += pd.Timedelta(hours=2)

    # print(valid_list)
    CGM_list = []
    for idx, i in enumerate(valid_list):
        start = pd.to_datetime(i)
        end = pd.to_datetime(i + timedelta(minutes=120))
        if set_num == 1:
            curr_date = i.date().strftime('%#m/%#d/%Y')
            CGM_list.append(df_CGM.loc[df_CGM['Date'] == curr_date].set_index('timestamp').between_time(
                start_time=start.strftime('%#H:%#M:%#S'), end_time=end.strftime('%#H:%#M:%#S'))[
                            'Sensor Glucose (mg/dL)'].values.tolist())
        else:
            curr_date = i.date().strftime('%Y-%m-%d')
            CGM_list.append(df_CGM.loc[df_CGM['Date'] == curr_date].set_index('timestamp').between_time(
                start_time=start.strftime('%H:%M:%S'), end_time=end.strftime('%H:%M:%S'))[
                                'Sensor Glucose (mg/dL)'].values.tolist())
    return pd.DataFrame(CGM_list)


no_meal_data = find_no_meal_data(df_insulin, df_CGM, 1)
no_meal_data = no_meal_data.iloc[:, 0:24]
# print(no_meal_data)
# no_meal_data.to_csv('test.csv', header=False, index=False)

no_meal_data_p2 = find_no_meal_data(df_insulin_p2, df_CGM_p2, 2)
no_meal_data_p2 = no_meal_data_p2.iloc[:, 0:24]
# print(no_meal_data_p2)



def meal_feature_matrix(meal_data):
    index=meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>6).dropna().index
    meal_data_cleaned=meal_data.drop(meal_data.index[index]).reset_index().drop(columns='index')
    meal_data_cleaned=meal_data_cleaned.interpolate(method='linear',axis=1)
    index_to_drop_again=meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    meal_data_cleaned=meal_data_cleaned.drop(meal_data.index[index_to_drop_again]).reset_index().drop(columns='index')
    meal_data_cleaned['tau_time']=(meal_data_cleaned.iloc[:,22:25].idxmin(axis=1)-meal_data_cleaned.iloc[:,5:19].idxmax(axis=1))
    meal_data_cleaned['difference_in_glucose_normalized']=(meal_data_cleaned.iloc[:,5:19].max(axis=1)-meal_data_cleaned.iloc[:,22:25].min(axis=1))/(meal_data_cleaned.iloc[:,22:25].min(axis=1))
    meal_data_cleaned=meal_data_cleaned.dropna().reset_index().drop(columns='index')
    power_first_max=[]
    index_first_max=[]
    power_second_max=[]
    index_second_max=[]
    for i in range(len(meal_data_cleaned)):
        array=abs(rfft(meal_data_cleaned.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(meal_data_cleaned.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    meal_feature_matrix=pd.DataFrame()
    meal_feature_matrix['tau_time']=meal_data_cleaned['tau_time']
    meal_feature_matrix['difference_in_glucose_normalized']=meal_data_cleaned['difference_in_glucose_normalized']
    meal_feature_matrix['power_first_max']=power_first_max
    meal_feature_matrix['power_second_max']=power_second_max
    meal_feature_matrix['index_first_max']=index_first_max
    meal_feature_matrix['index_second_max']=index_second_max
    tm=meal_data_cleaned.iloc[:,22:25].idxmin(axis=1)
    maximum=meal_data_cleaned.iloc[:,5:19].idxmax(axis=1)
    list1=[]
    second_differential_data=[]
    standard_deviation=[]
    for i in range(len(meal_data_cleaned)):
        list1.append(np.diff(meal_data_cleaned.iloc[:,maximum[i]:tm[i]].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(meal_data_cleaned.iloc[:,maximum[i]:tm[i]].iloc[i].tolist())).max())
        standard_deviation.append(np.std(meal_data_cleaned.iloc[i]))
    meal_feature_matrix['1stDifferential']=list1
    meal_feature_matrix['2ndDifferential']=second_differential_data
    return meal_feature_matrix


meal_matrix=meal_feature_matrix(meal_data)
meal_matrix_p2=meal_feature_matrix(meal_data_p2)
meal_matrix_combo=pd.concat([meal_matrix,meal_matrix_p2]).reset_index().drop(columns='index')
# print(meal_matrix_combo)


def no_meal_feature_matrix(non_meal_data):
    index_to_remove_non_meal=non_meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    non_meal_data_cleaned=non_meal_data.drop(non_meal_data.index[index_to_remove_non_meal]).reset_index().drop(columns='index')
    non_meal_data_cleaned=non_meal_data_cleaned.interpolate(method='linear',axis=1)
    index_to_drop_again=non_meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    non_meal_data_cleaned=non_meal_data_cleaned.drop(non_meal_data_cleaned.index[index_to_drop_again]).reset_index().drop(columns='index')
    non_meal_feature_matrix=pd.DataFrame()
    non_meal_data_cleaned['tau_time']=(24-non_meal_data_cleaned.iloc[:,0:19].idxmax(axis=1))
    non_meal_data_cleaned['difference_in_glucose_normalized']=(non_meal_data_cleaned.iloc[:,0:19].max(axis=1)-non_meal_data_cleaned.iloc[:,24])/(non_meal_data_cleaned.iloc[:,24])
    power_first_max,index_first_max,power_second_max,index_second_max=[],[],[],[]
    for i in range(len(non_meal_data_cleaned)):
        array=abs(rfft(non_meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(non_meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    non_meal_feature_matrix['tau_time']=non_meal_data_cleaned['tau_time']
    non_meal_feature_matrix['difference_in_glucose_normalized']=non_meal_data_cleaned['difference_in_glucose_normalized']
    non_meal_feature_matrix['power_first_max']=power_first_max
    non_meal_feature_matrix['power_second_max']=power_second_max
    non_meal_feature_matrix['index_first_max']=index_first_max
    non_meal_feature_matrix['index_second_max']=index_second_max
    first_differential_data=[]
    second_differential_data=[]
    for i in range(len(non_meal_data_cleaned)):
        first_differential_data.append(np.diff(non_meal_data_cleaned.iloc[:,0:24].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(non_meal_data_cleaned.iloc[:,0:24].iloc[i].tolist())).max())
    non_meal_feature_matrix['1stDifferential']=first_differential_data
    non_meal_feature_matrix['2ndDifferential']=second_differential_data
    return non_meal_feature_matrix


no_meal_matrix=no_meal_feature_matrix(no_meal_data)
no_meal_matrix_p2=no_meal_feature_matrix(no_meal_data_p2)
no_meal_matrix_combo=pd.concat([no_meal_matrix,no_meal_matrix_p2]).reset_index().drop(columns='index')
# print(no_meal_matrix_combo)

meal_matrix_combo['label']=1
no_meal_matrix_combo['label']=0
total_data=pd.concat([meal_matrix_combo,no_meal_matrix_combo]).reset_index().drop(columns='index')
# print(total_data)
dataset=shuffle(total_data,random_state=1).reset_index().drop(columns='index')
kfold = KFold(n_splits=10,shuffle=True,random_state=1)
principaldata=dataset.drop(columns='label')
scores_rf = []
model=DecisionTreeClassifier(criterion="entropy")
for train_index, test_index in kfold.split(principaldata):
    X_train,X_test,y_train,y_test = principaldata.loc[train_index],principaldata.loc[test_index],\
        dataset.label.loc[train_index],dataset.label.loc[test_index]
    model.fit(X_train,y_train)
    scores_rf.append(model.score(X_test,y_test))
print(np.mean(scores_rf)*100)

classifier=DecisionTreeClassifier(criterion='entropy')
X, Y= principaldata, dataset['label']
classifier.fit(X,Y)
pickle.dump(classifier, open("DecisionTreeClassifier.pickle", "wb"))