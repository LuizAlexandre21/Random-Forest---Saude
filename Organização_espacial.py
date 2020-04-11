import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.metrics import mean_squared_error as MSE
from sklearn.neighbors import KNeighborsClassifier as KNN
######################Importando os dados

local="/home/alexandre/Documentos/Artigo/Random Florest/Dados/dados_limpos.csv"
dados=pd.read_csv(local)
dados=dados.drop(columns=["Unnamed: 0","Resultado 2","Resultado 3","Resultado 4"])
#######e####### Floresta Aleatória

X=dados.drop(columns=['Motricidade_grosa','Motricidade_fina','Equilíbrio','Esq_corp_mãos','Esq_corp_braços','Organização_espacial'])
y=dados['Organização_espacial']
X_train, X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size=0.3,
                                                  random_state=1)
dt=RandomForestClassifier(n_estimators=1000,
                          min_samples_leaf=0.1,
                          random_state=1,
                          oob_score=True)

mse_cv=cross_val_score(dt,X_train,y_train,cv=10,scoring='neg_mean_squared_error',n_jobs=-1)
dt.fit(X_train,y_train)
mse_dt=MSE(y_test,dt.predict(X_test))
rmse_dt=mse_dt**(1/2)
importancia=pd.Series(dt.feature_importances_,index=X.columns)
importancia=importancia.sort_values()
importancia.plot(kind='barh',color='green')
plt.title("Organização")
plt.show()


############ Escolhendo a Modelagem

lr=LogisticRegression(random_state=1)
knn=KNN()
dt=RandomForestClassifier(random_state=1)

classifier=[("Logistic Regression",lr),
            ("K Nearest Neighbours",knn),
            ("Classification Tree",dt)]

for i,j in classifier:
    j.fit(X_train,y_train)
    y_pred=j.predict(X_test)
    print('{:s}:{:.3f}'.format(i,accuracy_score(y_test,y_pred)))


############### Bagging
#bc=BaggingClassifier(base_estimator=dt,n_estimator=100,n_jobs=-1)
#bc.fit(X_train, y_train)
#y_pred=bc.predict(X_test)
#accuracy=accuracy_score(y_test,y_pred1)

############### Adaboot
adb_clf=AdaBoostClassifier(base_estimator=dt,n_estimators=100)
adb_clf.fit(X_train,y_train)
y_pred_proba=adb_clf.predict_proba(X_test)[:1]
#adb_clf_roc_auc_score=roc_auc_score(y_test,y_pred_proba)

############### Podando a Arvore

params_dt={'max_depth':[9,10,11,12,13,14,15],
           'min_samples_leaf':[0.02,0.04,0.06,0.08],
           'max_features':[0.2,0.4,0.6,0.8]
           }

grid_dt=GridSearchCV(estimator=dt,
                     param_grid=params_dt,
                     scoring='accuracy',
                     cv=10,
                     n_jobs=-1)
grid_dt.fit(X_train,y_train)

best_hyperparams=grid_dt.best_params_
best_CV_score=grid_dt.best_score_


################Árvore Aleatória
dt=RandomForestClassifier(n_estimators=1000,random_state=1)
dt.fit(X_train,y_train)
export_graphviz(dt.estimators_[5],out_file="tree_nonlimited.dot",
                feature_names=X.columns,
                class_names=["0","1"],
                rounded=True,proportion=False,
                precision=2,filled=True)


call(['dot', '-Tpng', 'tree_nonlimited.dot', '-o', 'tree_nonlimited.png', '-Gdpi=60'])
