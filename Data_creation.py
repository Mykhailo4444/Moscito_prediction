import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv', sep=',', header=0)
data.drop(['Address', 'AddressNumberAndStreet'], inplace=True, axis=1)
data.head()

data1 = pd.read_csv('weather.csv', sep=',', header=0)
print(data1.head())

data1.drop(['CodeSum', 'SnowFall', 'PrecipTotal','Depth', 'Water1'],inplace=True, axis=1) # не несуть суттєвого змісту
data1 = data1[['Date', 'Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool', 'Sunrise', 'Sunset',
              'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed']][data1['Station'] == 1]
print(data1.head())

# коректирование даных сo значением 50
start = 50 # самое большое количество 50 : с 1071 по 1121 (то есть 51) достаточно 2^6
for i in range(6):
    del_index = [] # для запоминания индексов, что были использованы
    for i in data.index: # непосредственно сумирование значений
        if i != 0:
            if data.loc[i - 1, 'NumMosquitos'] > start:
                continue
        if data.loc[i, 'NumMosquitos'] == start:
            data.loc[i, 'NumMosquitos'] = data.loc[i, 'NumMosquitos'] + data.loc[i + 1, 'NumMosquitos']
            data.loc[i, 'WnvPresent'] = 1
            del_index.append(i + 1)
    for i in del_index: # удаляю прибавленые строки
        data.drop(i, inplace=True, axis=0)
    new_length = len(data.index)
    new_index = np.arange(new_length)
    data.set_index(new_index, inplace=True)
    start *= 2
print(data.iloc[6360:6377])

new_data = pd.merge(data, data1, how='left', on=['Date']) # обьеденение погоды и табл. с комарами
print(new_data.tail(20))

new_data.to_csv('total_train', sep=',')

