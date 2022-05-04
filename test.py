import pandas as pd
import numpy as np
from scipy.fftpack import rfft
import pickle

test_data = pd.read_csv('test.csv', header=None)

def meal_feature_matrix(meal_data):
    index=meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>6).dropna().index
    meal_data_cleaned=meal_data.drop(meal_data.index[index]).reset_index().drop(columns='index')
    meal_data_cleaned=meal_data_cleaned.interpolate(method='linear',axis=1)
    index_to_drop_again=meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    meal_data_cleaned=meal_data_cleaned.drop(meal_data.index[index_to_drop_again]).reset_index().drop(columns='index')
    meal_data_cleaned['tau_time']=(meal_data_cleaned.iloc[:,22:24].idxmin(axis=1)-meal_data_cleaned.iloc[:,5:19].idxmax(axis=1))*5
    meal_data_cleaned['difference_in_glucose_normalized']=(meal_data_cleaned.iloc[:,5:19].max(axis=1)-meal_data_cleaned.iloc[:,22:24].min(axis=1))/(meal_data_cleaned.iloc[:,22:24].min(axis=1))
    meal_data_cleaned=meal_data_cleaned.dropna().reset_index().drop(columns='index')
    power_first_max=[]
    index_first_max=[]
    power_second_max=[]
    index_second_max=[]
    for i in range(len(meal_data_cleaned)):
        array=abs(rfft(meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
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
    tm=meal_data_cleaned.iloc[:,22:24].idxmin(axis=1)
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



# def test_feature_matrix(test_data):
#     index_to_remove_test=test_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
#     test_data_cleaned=test_data.drop(test_data.index[index_to_remove_test]).reset_index().drop(columns='index')
#     test_data_cleaned=test_data_cleaned.interpolate(method='linear',axis=1)
#     index_to_drop_again=test_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
#     test_data_cleaned=test_data_cleaned.drop(test_data_cleaned.index[index_to_drop_again]).reset_index().drop(columns='index')
#     test_feature_matrix=pd.DataFrame()
#     test_data_cleaned['tau_time']=(24-test_data_cleaned.iloc[:,0:19].idxmax(axis=1))*5
#     test_data_cleaned['difference_in_glucose_normalized']=(test_data_cleaned.iloc[:,0:19].max(axis=1)-test_data_cleaned.iloc[:,24])/(test_data_cleaned.iloc[:,24])
#     power_first_max=[]
#     index_first_max=[]
#     power_second_max=[]
#     index_second_max=[]
#     for i in range(len(test_data_cleaned)):
#         array=abs(rfft(test_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
#         sorted_array=abs(rfft(test_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
#         sorted_array.sort()
#         power_first_max.append(sorted_array[-2])
#         power_second_max.append(sorted_array[-3])
#         index_first_max.append(array.index(sorted_array[-2]))
#         index_second_max.append(array.index(sorted_array[-3]))
#     test_feature_matrix['tau_time']=test_data_cleaned['tau_time']
#     test_feature_matrix['difference_in_glucose_normalized']=test_data_cleaned['difference_in_glucose_normalized']
#     test_feature_matrix['power_first_max']=power_first_max
#     test_feature_matrix['power_second_max']=power_second_max
#     test_feature_matrix['index_first_max']=index_first_max
#     test_feature_matrix['index_second_max']=index_second_max
#     first_differential_data=[]
#     second_differential_data=[]
#     for i in range(len(test_data_cleaned)):
#         first_differential_data.append(np.diff(test_data_cleaned.iloc[:,0:24].iloc[i].tolist()).max())
#         second_differential_data.append(np.diff(np.diff(test_data_cleaned.iloc[:,0:24].iloc[i].tolist())).max())
#     test_feature_matrix['1stDifferential']=first_differential_data
#     test_feature_matrix['2ndDifferential']=second_differential_data
#     return test_feature_matrix


dataset = meal_feature_matrix(test_data)
# print(dataset)
pickle_file = pickle.load(open("DecisionTreeClassifier.pickle", "rb"))
predict = pickle_file.predict(dataset)
pd.DataFrame(predict).to_csv('Result.csv',index=False,header=False)