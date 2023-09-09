import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
class decoding_list:
        def __init__(self):
                             
                train_ds = pd.read_csv('fraudTrain.csv')
                test_ds = pd.read_csv('fraudTest.csv')

                data=pd.concat([train_ds,test_ds],axis=0)
                data= data.drop(['Unnamed: 0'], axis=1)

                self.x =  data.drop(['is_fraud'],axis=1)

        def decoder(self, new_input_data):
                columns = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last',
                'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
                new_data=pd.DataFrame([new_input_data],columns=self.x.columns)
                appending = pd.concat([new_data,self.x], axis=0)
                encoder = OrdinalEncoder()
                appending[columns]= encoder.fit_transform(appending[columns])
                from sklearn.preprocessing import MinMaxScaler
                scaled= MinMaxScaler()
                user_x = scaled.fit_transform(appending)
                return user_x[0]


