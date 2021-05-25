import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


data = pd.read_csv('total_train', sep=',', header=0)

data['Date'] = data['Date'].apply(lambda x: x.split('-')[0]+x.split('-')[1]+x.split('-')[2])

encoder1 = ce.TargetEncoder(min_samples_leaf=4, smoothing=50).fit(data['Street'], data['AddressAccuracy'])
data['Street'] = encoder1.transform(data['Street'])

encoder2 = ce.TargetEncoder(min_samples_leaf=4, smoothing=50).fit(data['Species'], data['AddressAccuracy'])
data['Species'] = encoder2.transform(data['Species'])


def for_trap(x): # избавление от излишних букв в колвнке 'Trap'
    null_str = ''
    for i in range(1, len(x)):
        if x[i] == 'B':
            null_str += '8'
        elif x[i] == 'C':
            null_str += '9'
        else:
            null_str += x[i]
    x = null_str
    return x


data['Trap'] = data['Trap'].apply(for_trap)

array1 = np.array(data['WetBulb'].values[data['WetBulb'] != 'M'])
array1 = array1.astype(float)
data['WetBulb'] = np.where(data['WetBulb'] == 'M', np.mean(array1), data['WetBulb']) # замена пропущеных на среднее

array2 = np.array(data['StnPressure'].values[data['StnPressure'] != 'M'])
array2 = array2.astype(float)
data['StnPressure'] = np.where(data['StnPressure'] == 'M', np.mean(array2), data['StnPressure']) # замена пропущеных на среднее

X = data[['Date', 'Species', 'Block', 'Street', 'Trap', 'Latitude', 'Longitude', 'AddressAccuracy', 'Tmax', 'Tmin',
          'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'StnPressure', 'SeaLevel',
          'ResultSpeed', 'ResultDir', 'AvgSpeed']]
y = data['NumMosquitos'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test = X_train.copy(), X_test.copy()
# построение модели Линейной регресии
model = LinearRegression().fit(X_train, y_train)
prediction = model.predict(X_test)
print(mean_squared_error(y_test, prediction))

importance = permutation_importance(model,X_test, y_test, n_repeats=40)  # проверяю важность переменных
for i, column in enumerate(X_test.columns):
    print(column, round(importance.importances_mean[i],5), round(importance.importances_std[i],5))

# удаляю переменные, что не несут смысловой важности
X_train.drop(['AvgSpeed', 'ResultDir', 'ResultSpeed', 'Sunset', 'Cool', 'WetBulb', 'DewPoint', 'Tavg', 'AddressAccuracy',
              'Species', 'Date', 'StnPressure', 'SeaLevel'], axis=1, inplace=True)
X_test.drop(['AvgSpeed', 'ResultDir', 'ResultSpeed', 'Sunset', 'Cool', 'WetBulb', 'DewPoint', 'Tavg', 'AddressAccuracy',
             'Species', 'Date', 'StnPressure', 'SeaLevel'], axis=1, inplace=True)
model = LinearRegression().fit(X_train, y_train)
prediction = model.predict(X_test)
print(mean_squared_error(y_test, prediction)) # показатель улучшился
# удаление выбросов и регуляризация ухудшают модель

plt.subplot(3,4,1)
plt.hist(X['Block'])
plt.subplot(3,4,2)
plt.hist(X['Street'])
plt.subplot(3,4,3)
plt.hist(X['Trap'])
plt.subplot(3,4,4)
plt.hist(X['Latitude'])
plt.subplot(3,4,5)
plt.hist(X['Longitude'])
plt.subplot(3,4,6)
plt.hist(X['Tmax'])
plt.subplot(3,4,7)
plt.hist(X['Tmin'])
plt.subplot(3,4,8)
plt.hist(X['Depart'])
plt.subplot(3,4,9)
plt.hist(X['Heat'])
plt.subplot(3,4,10)
plt.hist(X['Sunrise'])
plt.show()

# улучшаю показатели оставшихся переменных
X_train['Block'] = np.log1p(X_train['Block'])  # логарифмирую
X_test['Block'] = np.log1p(X_test['Block'])
model = LinearRegression().fit(X_train, y_train)
prediction = model.predict(X_test)
print(mean_squared_error(y_test, prediction)) # показатель улучшился

# разделяю переменные на две части
X_train['Low_Latitude'] = np.where(X_train['Latitude'] < 41.9, X_train['Latitude'], 0)
X_train['High_Latitude'] = np.where(X_train['Latitude'] >= 41.9, X_train['Latitude'], 0)
X_test['Low_Latitude'] = np.where(X_test['Latitude'] < 41.9, X_test['Latitude'], 0)
X_test['High_Latitude'] = np.where(X_test['Latitude'] >= 41.9, X_test['Latitude'], 0)
X_train.drop(['Latitude'], axis=1, inplace=True)
X_test.drop(['Latitude'], axis=1, inplace=True)
model = LinearRegression().fit(X_train, y_train)
prediction = model.predict(X_test)
print(mean_squared_error(y_test, prediction)) # показатель улучшился

X_train['Low_Sunrise'] = np.where(X_train['Sunrise'] < 472, X_train['Sunrise'], 0)
X_train['High_Sunrise'] = np.where(X_train['Sunrise'] >= 486, X_train['Sunrise'], 0)
X_test['Low_Sunrise'] = np.where(X_test['Sunrise'] < 472, X_test['Sunrise'], 0)
X_test['High_Sunrise'] = np.where(X_test['Sunrise'] >= 486, X_test['Sunrise'], 0)
X_train.drop(['Sunrise'], axis=1, inplace=True)
X_test.drop(['Sunrise'], axis=1, inplace=True)
model = LinearRegression().fit(X_train, y_train)
prediction = model.predict(X_test)
print(mean_squared_error(y_test, prediction))
print(np.round(prediction[0:8],0))# показатель улучшился, максимально выжатый результат на линейной ригресии

# построение дерева на тех же данных
tree1 = DecisionTreeRegressor(criterion='mse',  max_depth=9,  min_samples_split=10,  min_samples_leaf=5).fit(X_train, y_train)
prediction = tree1.predict(X_test)
print(mean_squared_error(y_test, prediction))
print(np.round(prediction[0:8],0))# показатель улучшился

# построение градиентноого бустинга на тех же данных
X_train['Trap'] = pd.to_numeric(X_train['Trap'])
X_test['Trap'] = pd.to_numeric(X_test['Trap'])
dtrain = xgb.DMatrix(X_train, y_train, )
dtest = xgb.DMatrix(X_test, y_test, )
params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'gamma': 1.,
        'max_depth': 9,
        'min_child_weight': 5,
        'seed': 32  }
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=50,
    evals=[(dtrain, 'Train'), (dtest, 'Test')]
)
prediction = model.predict(dtest)
print(mean_squared_error(y_test, prediction)) # показатель не улучшился

# создание непосредственно данных на которых будет тренероваться основная модель
X = data[['Block', 'Street', 'Trap', 'Longitude', 'Tmax', 'Tmin',  'Depart', 'Heat', 'Latitude','Sunrise']].copy()
y = data['NumMosquitos'].values

X['Block'] = np.log1p(X['Block'])
X['Low_Latitude'] = np.where(X['Latitude'] < 41.9, X['Latitude'], 0)
X['High_Latitude'] = np.where(X['Latitude'] >= 41.9, X['Latitude'], 0)
X.drop(['Latitude'], axis=1, inplace=True)

X['Low_Sunrise'] = np.where(X['Sunrise'] < 472, X['Sunrise'], 0)
X['High_Sunrise'] = np.where(X['Sunrise'] >= 486, X['Sunrise'], 0)
X.drop(['Sunrise'], axis=1, inplace=True)
print(X.head())

# создание данных с помощью которых будет делаться предикт
data1 = pd.read_csv('test_truncated.csv', sep=';', header=0)
data2 = pd.read_csv('weather.csv', sep=',', header=0)

data2.drop(['CodeSum', 'SnowFall', 'PrecipTotal','Depth', 'Water1'], inplace=True, axis=1)
data2 = data2[['Date', 'Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset',
              'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed']][data2['Station'] == 1]
data1 = pd.merge(data1, data2, how='left', on=['Date'])
encoder2 = ce.TargetEncoder(min_samples_leaf=4, smoothing=50).fit(data1['Street'], data1['AddressAccuracy'])
data1['Street'] = encoder2.transform(data1['Street'])
data1['Trap'] = data1['Trap'].apply(for_trap)

X_topredict = data1[['Block', 'Street', 'Trap', 'Longitude', 'Tmax', 'Tmin',  'Depart', 'Heat', 'Latitude',
                    'Sunrise']].copy()
X_topredict['Block'] = np.log1p(X_topredict['Block'])

X_topredict['Low_Latitude'] = np.where(X_topredict['Latitude'] < 41.9, X_topredict['Latitude'], 0)
X_topredict['High_Latitude'] = np.where(X_topredict['Latitude'] >= 41.9, X_topredict['Latitude'], 0)
X_topredict.drop(['Latitude'], axis=1, inplace=True)

X_topredict['Sunrise'] = pd.to_numeric(X_topredict['Sunrise'])
X_topredict['Low_Sunrise'] = np.where(X_topredict['Sunrise'] < 472, X_topredict['Sunrise'], 0)
X_topredict['High_Sunrise'] = np.where(X_topredict['Sunrise'] >= 486, X_topredict['Sunrise'], 0)
X_topredict.drop(['Sunrise'], axis=1, inplace=True)

X_topredict.head()

# непосредственно построение лучшей модели и предсказание колонки NumMosquitos
write_data = pd.read_csv('test_truncated.csv', sep=';', header=0)
my_tree = DecisionTreeRegressor(criterion='mse',  max_depth=9,  min_samples_split=10,  min_samples_leaf=5).fit(X, y)
write_data['NumMosquitos'] = np.round(my_tree.predict(X_topredict),0)
write_data.to_csv('test_truncated.csv', sep=',')

# Переохожу к прогнозированию к прогнозированию WnvPresent
# Проделываю все те же первоначальные манипулции с данными
new_data = pd.read_csv('total_train', sep=',', header=0)
new_data['Date'] = new_data['Date'].apply(lambda x: x.split('-')[0]+x.split('-')[1]+x.split('-')[2])

encoder1 = ce.TargetEncoder(min_samples_leaf=4, smoothing=50).fit(new_data['Street'], new_data['AddressAccuracy'])
new_data['Street'] = encoder1.transform(new_data['Street'])

encoder2 = ce.TargetEncoder(min_samples_leaf=4, smoothing=50).fit(new_data['Species'], new_data['AddressAccuracy'])
new_data['Species'] = encoder2.transform(new_data['Species'])


def for_trap(x):
    null_str = ''
    for i in range(1, len(x)):
        if x[i] == 'B':
            null_str += '8'
        elif x[i] == 'C':
            null_str += '9'
        else:
            null_str += x[i]
    x = null_str
    return x


new_data['Trap'] = new_data['Trap'].apply(for_trap)

array1 = np.array(new_data['WetBulb'].values[new_data['WetBulb'] != 'M'])
array1 = array1.astype(float)
new_data['WetBulb'] = np.where(new_data['WetBulb'] == 'M', np.mean(array1), new_data['WetBulb'])

array2 = np.array(new_data['StnPressure'].values[new_data['StnPressure'] != 'M'])
array2 = array2.astype(float)
new_data['StnPressure'] = np.where(new_data['StnPressure'] == 'M', np.mean(array2), new_data['StnPressure'])
new_data.drop('Unnamed: 0', inplace=True, axis=1)
new_data.to_csv('total_train', sep=',')

data = pd.read_csv('total_train', sep=',', header=0)

X = data[['Date', 'Species', 'Block', 'Street', 'Trap', 'Latitude', 'Longitude', 'AddressAccuracy', 'Tmax', 'Tmin',
          'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'StnPressure', 'SeaLevel',
          'ResultSpeed', 'ResultDir', 'AvgSpeed', 'NumMosquitos']].copy()
y = data['WnvPresent'].values
print(X.head())
# Строим модель Логистической регресии
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = X_train.copy(), X_test.copy()
model = LogisticRegression().fit(X_train, y_train)
prediction = model.predict(X_test)
print(roc_auc_score(y_test, prediction))

# Откидываю переменные, что мешают модели
X_train.drop(['AvgSpeed', 'ResultDir', 'ResultSpeed', 'Sunset', 'Cool', 'WetBulb', 'DewPoint', 'Tavg', 'AddressAccuracy',
               'Date',  'Tmin', 'Tmax', 'Block',  'Trap','SeaLevel', 'Heat',  'Depart','Sunrise'], axis=1, inplace=True)
X_test.drop(['AvgSpeed', 'ResultDir', 'ResultSpeed', 'Sunset', 'Cool', 'WetBulb', 'DewPoint', 'Tavg', 'AddressAccuracy',
              'Date',  'Tmin', 'Tmax', 'Block',  'Trap','SeaLevel', 'Heat',  'Depart','Sunrise'], axis=1, inplace=True)
model = LogisticRegression().fit(X_train, y_train)
prediction = model.predict(X_test)
print(roc_auc_score(y_test, prediction))
print(X_train.head()) # показатель вырос

plt.subplot(231)
plt.hist(data['Species'])
plt.subplot(232)
plt.hist(data['Street'])
plt.subplot(233)
plt.hist(data['Latitude'])
plt.subplot(234)
plt.hist(data['Longitude'])
plt.subplot(235)
plt.hist(data['StnPressure'])
plt.subplot(236)
plt.hist(data['NumMosquitos'])
plt.show() # логарифмирование, разделение только усугубили показатели


X.drop(['AvgSpeed', 'ResultDir', 'ResultSpeed', 'Sunset', 'Cool', 'WetBulb', 'DewPoint', 'Tavg', 'AddressAccuracy',
               'Date',  'Tmin', 'Tmax', 'Block',  'Trap','SeaLevel', 'Heat',  'Depart','Sunrise'], axis=1, inplace=True)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def estimate_cv(reg_param):
    test_res = []
    for train_idx, test_idx in kf.split(X, y):
        X_train, y_train = X.loc[train_idx].copy(), y[train_idx]
        X_test, y_test = X.loc[test_idx].copy(), y[test_idx]
        mdl = LogisticRegression(penalty='l1',  C=1 / reg_param,  solver='liblinear', )
        mdl = mdl.fit(X_train, y_train)
        y_test_preds = mdl.predict_proba(X_test)[:, 1]
        test_res.append(np.round(roc_auc_score(y_test, y_test_preds), 4))
    return np.round(np.mean(test_res), 4), np.round(np.std(test_res), 4)


for reg_param in [2.1, 2.3, 2.7, 3]:
    test_error, test_error_std = estimate_cv(reg_param)
    print(f'Regularizatoin: {reg_param} -  test error: {test_error} +- {test_error_std}')

model = LogisticRegression(penalty='l1',C=1/2.7,solver='liblinear').fit(X_train, y_train)
prediction = model.predict(X_test)
print(roc_auc_score(y_test, prediction)) # показник знизився

svm = SVC()
params_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],'C': [0.5, 1, 2, 5]}
clf = GridSearchCV(estimator=svm,  param_grid=params_grid, scoring='f1_macro',  cv=4  )
clf = clf.fit(X_train, y_train)
grid_search_results = pd.DataFrame(clf.cv_results_)
print(grid_search_results)

# 'C': 5, 'kernel': 'poly' - найкращий показник
svm = SVC(C=5, kernel='poly').fit(X_train, y_train)
prediction = svm.predict(X_test)
print(roc_auc_score(y_test, prediction)) # показник зменшився

mdl = RandomForestClassifier(n_estimators=80, criterion='gini', max_depth=9, min_samples_split=10, min_samples_leaf=5,
                             max_features=0.8,  bootstrap=True, max_samples=0.8,  random_state=42  )

mdl = mdl.fit(X_train, y_train)
prediction = mdl.predict(X_test)
print(roc_auc_score(y_test, prediction)) # показник зменшився

params = {'objective': 'binary:logistic', 'learning_rate': 0.1, 'subsample': 0.2, 'colsample_bytree': 1,
          'colsample_bylevel': 0.8, 'reg_lambda': 0.5, 'gamma': 1., 'max_depth': 6, 'min_child_weight': 5,
          'eval_metric': 'auc', 'silent': 1, 'seed': 32, 'n_estimators': 30}
xgb_mdl = xgb.XGBClassifier(**params)
params_grid = {'max_depth': [5,6,7], 'min_child_weight': [10,11,12]}
clf = GridSearchCV(estimator=xgb_mdl, param_grid=params_grid,  scoring='roc_auc', cv=4 ).fit(X, y)
results = pd.DataFrame(clf.cv_results_)
best_result = results[results['rank_test_score'] == 1]
print(best_result['params'].values, best_result['mean_test_score'].values) # показало найкращі результати

data = pd.read_csv('total_train', sep=',', header=0)
X = data[['Date', 'Species', 'Block', 'Street', 'Trap', 'Latitude', 'Longitude', 'AddressAccuracy', 'Tmax', 'Tmin',
          'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset', 'StnPressure', 'SeaLevel',
          'ResultSpeed', 'ResultDir', 'AvgSpeed', 'NumMosquitos']].copy()
y = data['WnvPresent'].values

X.drop(['AvgSpeed', 'ResultDir', 'ResultSpeed', 'Sunset', 'Cool', 'WetBulb', 'DewPoint', 'Tavg', 'AddressAccuracy',
               'Date',  'Tmin', 'Tmax', 'Block',  'Trap','SeaLevel', 'Heat',  'Depart','Sunrise'], axis=1, inplace=True)

data1 = pd.read_csv('test_truncated.csv', sep=',', header=0)
data2 = pd.read_csv('weather.csv', sep=',', header=0)

data2.drop(['CodeSum', 'SnowFall', 'PrecipTotal','Depth', 'Water1'], inplace=True, axis=1)
data2 = data2[['Date', 'Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset',
              'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed']][data2['Station'] == 1]
data1 = pd.merge(data1, data2, how='left', on=['Date'])
encoder1 = ce.TargetEncoder(min_samples_leaf=4, smoothing=50).fit(data1['Street'], data1['AddressAccuracy'])
data1['Street'] = encoder1.transform(data1['Street'])
encoder2 = ce.TargetEncoder(min_samples_leaf=4, smoothing=50).fit(data1['Species'], data1['AddressAccuracy'])
data1['Species'] = encoder2.transform(data1['Species'])

X_topredict = data1[['Species', 'Street', 'Latitude', 'Longitude', 'StnPressure', 'NumMosquitos']].copy()
y_topredict = data1['WnvPresent'].values

params = {'objective': 'binary:logistic', 'learning_rate': 0.1, 'subsample': 0.2, 'colsample_bytree': 1,
          'colsample_bylevel': 0.8, 'reg_lambda': 0.5, 'gamma': 1., 'max_depth': 7, 'min_child_weight': 11,
          'eval_metric': 'auc', 'silent': 1, 'seed': 32, 'n_estimators': 30}

X['StnPressure'] = pd.to_numeric(X['StnPressure'])
X_topredict['StnPressure'] = pd.to_numeric(X_topredict['StnPressure'])

dtrain = xgb.DMatrix(X, y)
dtest = xgb.DMatrix(X_topredict, y_topredict)

mdl = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    early_stopping_rounds=20,
    evals=[(dtrain, 'Train'), (dtest, 'Test')]
)
prediction = mdl.predict(dtest)

final_data = pd.read_csv('test_truncated.csv', sep=',', header=0)
final_data['WnvPresent'] = prediction
final_data.to_csv('test_truncated.csv', sep=',')