import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
class CreditCardFraud:
    def __init__(self):
        train_ds = pd.read_csv('fraudTrain.csv')
        test_ds = pd.read_csv('fraudTest.csv')

        data=pd.concat([train_ds,test_ds],axis=0)
        data= data.drop(['Unnamed: 0'], axis=1)

        x =  data.drop(['is_fraud'],axis=1)
        y= data['is_fraud']

        columns = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last',
                'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']


        encoder = OrdinalEncoder()
        x[columns]= encoder.fit_transform(x[columns])


        scaled= MinMaxScaler()
        x = scaled.fit_transform(x)
        y = data[['is_fraud']].values


        from imblearn.under_sampling import NearMiss
        nm_sampler = NearMiss()
        x_sampled, y_sampled = nm_sampler.fit_resample(x,y)


        x_train, x_test, y_train, y_test= train_test_split(x_sampled,y_sampled, test_size=0.2, random_state=2)
        #logistic regression
        # lr = LogisticRegression()
        # fitting_data= lr.fit(x_train,y_train)
        # pred_train= lr.predict(x_train)
        # pred_test= lr.predict(x_test)

        # #checking accuracy

        # lr_accuracy  =  accuracy_score(y_train,pred_train)
        # f1= f1_score(y_train,pred_train)
        # rec= recall_score(y_train,pred_train)
        # preci=precision_score(y_train,pred_train)

        # print(lr_accuracy,f1,rec,preci)


        self.rfc = RandomForestClassifier()
        fitting_data= self.rfc.fit(x_train,y_train)
        pred_train= self.rfc.predict(x_train)
        pred_test= self.rfc.predict(x_test)

        #checking accuracy

        lr_accuracy  =  accuracy_score(y_train,pred_train)
        # f1= f1_score(y_train,pred_train)
        # rec= recall_score(y_train,pred_train)
        # preci=precision_score(y_train,pred_train)

        # # print(lr_accuracy,f1,rec,preci)

        # # print('testing data training')
        # lr_accuracy  =  accuracy_score(y_test,pred_test)
        # f1= f1_score(y_test,pred_test)
        # rec= recall_score(y_test,pred_test)
        # preci=precision_score(y_test,pred_test)

        # print(lr_accuracy,f1,rec,preci)


        # new_input=['01-01-2019 00:00:18',	2703186189652095,	'fraud_Rippin, Kub and Mann',	'misc_net',	4.97,	'Jennifer',	'Banks'	,'F'	,'561 Perry Cove',	'Moravian Falls',	'NC',	28654,	36.0788,	-81.1781,	3495	,'Psychologist, counselling'	,'09/03/1988',	'0b242abb623afc578575680df30655b9',	1325376018,	36.011293,	-82.048315]
        
    def new_prediction(self,new_input):
        from testing import decoding_list
        insa=decoding_list()
        newX= insa.decoder(new_input)
        let = self.rfc.predict([newX])
        return let[0]

# new_input=['2019-01-03 16:54:53',  4922710831011201,'fraud_Rau and Sons','grocery_pos',337.05,'Heather','Chase','F','6888 Hicks Stream Suite 954','Manor','PA',15665,40.3359,-79.6607,1472,'Public affairs consultant','1941-03-07','7301679c460c5f2a464b0ecb5c610b47',1325609693,41.174382,-79.809888,]
# import datetime
# t1 = datetime.datetime.now()
# obj = CreditCardFraud()
# obj1= obj.new_prediction(new_input)
# print(obj1)
# t2=datetime.datetime.now()
# print(t2-t1)
