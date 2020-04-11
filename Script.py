import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


##############################
local="D:\\Artigo\\Random Florest\\Dados\\dados_limpos.csv"
dados=pd.read_csv(local)
dados=dados.drop(columns=["Unnamed: 0"])
print("iniciou")
########Random Florest
rmse={}
#####Resultado 2
print("Floresta Aleatória")

X=dados.drop(columns=['Resultado 2','Resultado 3','Resultado 4'])
y=dados['Resultado 2']
X_train, X_test, y_train, y_test =train_test_split(X,y,
                                                   test_size=0.3,
                                                   random_state=1)
dt1=RandomForestClassifier(n_estimators=10000,
                         min_samples_leaf=0.1,
                         random_state=1,
                         oob_score=True)

mse_cv1=cross_val_score(dt1,X_train,y_train,cv=10,scoring='neg_mean_squared_error',n_jobs=-1)
dt1.fit(X_train,y_train)
dt1.pred=dt1.predict(X_test)
mse_dt=MSE(y_test,dt1.pred)
rmse_dt=mse_dt**(1/2)
rmse['Resultado 2']=rmse_dt
importancia=pd.Series(dt1.feature_importances_,index=X.columns)
importancia = importancia.sort_values()
importancia.plot(kind='barh', color='green')
plt.title("Resultado 2")
plt.show()


X=dados.drop(columns=['Resultado 2','Resultado 3','Resultado 4'])
y=dados['Resultado 4']
X_train, X_test, y_train, y_test =train_test_split(X,y,
                                                   test_size=0.3,
                                                   random_state=1)
dt2=RandomForestClassifier(n_estimators=1000,
                         min_samples_leaf=0.1,
                         random_state=1)
mse_cv2=cross_val_score(dt2,X_train,y_train,cv=10,scoring='neg_mean_squared_error',n_jobs=-1)
dt2.fit(X_train,y_train)
dt2.pred=dt2.predict(X_test)
mse_dt=MSE(y_test,dt2.pred)
rmse_dt=mse_dt**(1/2)
rmse['Resultado 4']=rmse_dt
importancia=pd.Series(dt2.feature_importances_,index=X.columns)
importancia = importancia.sort_values()
importancia.plot(kind='barh', color='green')
plt.title("Resultado 4")
plt.show()

X=dados.drop(columns=['Resultado 2','Resultado 3','Resultado 4'])
y=dados['Resultado 3']
X_train, X_test, y_train, y_test =train_test_split(X,y,
                                                   test_size=0.3,
                                                   random_state=1)
dt3=RandomForestClassifier(n_estimators=1000,
                         min_samples_leaf=0.1,
                         random_state=1)
mse_cv3=cross_val_score(dt3,X_train,y_train,cv=10,scoring='neg_mean_squared_error',n_jobs=-1)
dt3.fit(X_train,y_train)
dt3.pred=dt3.predict(X_test)
mse_dt=MSE(y_test,dt3.pred)
rmse_dt=mse_dt**(1/2)
rmse['Resultdodo 3']=rmse_dt
importancia=pd.Series(dt3.feature_importances_,index=X.columns)
importancia = importancia.sort_values()
importancia.plot(kind='barh', color='green')
plt.title("Resultado 3")
plt.show()


#################### Melhor modelo para classificação
print("modelo para classificação")
SEED=1
X_train, X_test, y_train, y_test =train_test_split(X,y,
                                                   test_size=0.3,
                                                   random_state=SEED)
lr=LogisticRegression(random_state=SEED)
knn=KNN()
dt=RandomForestClassifier(random_state=SEED)

classifier=[('Logistic Regression',lr),
            ('K Nearest Neighbours',knn),
            ('Classification Tree',dt1),
            ('Classification Tree',dt2),
            ('Classification Tree',dt3)]

for i, j in classifier:
    j.fit(X_train,y_train)
    y_pred=j.predict(X_test)
    print('{:s}:{:.3f}'.format(i,accuracy_score(y_test,y_pred)))
#################Bagging
"""print("Bagging")
bc1=BaggingClassifier(base_estimator=dt1, n_estimators=100, n_jobs=-1)
bc1.fit(X_train, y_train)
y_pred1 = bc1.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)

bc2=BaggingClassifier(base_estimator=dt2, n_estimators=100, n_jobs=-1)
bc2.fit(X_train, y_train)
y_pred2 = bc2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)

bc3=BaggingClassifier(base_estimator=dt3, n_estimators=100, n_jobs=-1)
bc3.fit(X_train, y_train)
y_pred3 = bc3.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred3)
"""
#####################Boosting

#######Adaboost
print("Adaboost")
adb_clf1=AdaBoostClassifier(base_estimator=dt1, n_estimators=100)
adb_clf1.fit(X_train, y_train)
y_pred_proba1 = adb_clf1.predict_proba(X_test)[:,1]
adb_clf1_roc_auc_score = roc_auc_score(y_test, y_pred_proba1)

adb_clf2=AdaBoostClassifier(base_estimator=dt2, n_estimators=100)
adb_clf2.fit(X_train, y_train)
y_pred_proba2 = adb_clf2.predict_proba(X_test)[:,1]
adb_clf2_roc_auc_score = roc_auc_score(y_test, y_pred_proba2)

adb_clf3=AdaBoostClassifier(base_estimator=dt3, n_estimators=100)
adb_clf3.fit(X_train, y_train)
y_pred_proba3 = adb_clf3.predict_proba(X_test)[:,1]
adb_clf3_roc_auc_score = roc_auc_score(y_test, y_pred_proba3)

#####################Podando a Arvore
print("podando")
params_dt={'max_depth':[9,10,11,12,13,14,15],
           'min_samples_leaf':[0.02,0.04,0.06,0.08],
           'max_features':[0.2,0.4,0.6,0.8]
           }
grid_dt1=GridSearchCV(estimator=dt1,
                     param_grid=params_dt,
                     scoring='accuracy',
                     cv=10,
                     n_jobs=-1)
grid_dt1.fit(X_train,y_train)
best_hyperparams1=grid_dt1.best_params_
best_CV_score1=grid_dt1.best_score_


grid_dt2=GridSearchCV(estimator=dt2,
                     param_grid=params_dt,
                     scoring='accuracy',
                     cv=10,
                     n_jobs=-1)
grid_dt2.fit(X_train,y_train)
best_hyperparams2=grid_dt2.best_params_
best_CV_score2=grid_dt2.best_score_

grid_dt3=GridSearchCV(estimator=dt3,
                     param_grid=params_dt,
                     scoring='accuracy',
                     cv=10,
                     n_jobs=-1)
grid_dt3.fit(X_train,y_train)
best_hyperparams3=grid_dt3.best_params_
best_CV_score3=grid_dt3.best_score_
##################### Fim
print("acabou")