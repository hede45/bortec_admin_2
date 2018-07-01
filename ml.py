import mysql.connector
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import pickle
from config import *
import time
import datetime

# functional programming

dbuser = DB_USER
dbpassword = DB_PASSWORD

conn = mysql.connector.connect(user=dbuser, password=dbpassword, host='localhost',
                               database='bortec_inv_system_db')

# Load available item IDs
cursor = conn.cursor()
# cursor.execute('select items.id, items.product_name, inventory_stocks.stocks, count(sales.item_id) as num_sales from items left join inventory_stocks on items.id = inventory_stocks.item_id left join sales on items.id = sales.item_id where num_sales < 10 group by items.id')
#cursor.execute('select distinct items.id, items.product_name, inventory_stocks.stocks, (SELECT COUNT(*) FROM sales WHERE item_id = items.id) AS memberCount from items left join inventory_stocks on items.id = inventory_stocks.item_id where memberCount = 1')
cursor.execute('SELECT sales.item_id, items.product_name, inventory_stocks.stocks, COUNT(*) as membercount  FROM sales left join items on sales.item_id = items.id left join inventory_stocks on items.id = inventory_stocks.item_id GROUP BY item_id HAVING membercount > 5')
#cursor.execute('select count(sales.item_id) as num_sales from items left join sales on items.id = sales.item_id group by items.id')
data_list = cursor.fetchall()
length = len(data_list)

print('list ',data_list)


def train_models(item_id):
    query = 'SELECT item_id, AVG(weather) AS weather, AVG(temp) AS temp, AVG(temp_min) AS tmin, AVG(temp_max) AS tmax, ' \
            'AVG(pressure) AS pressure, AVG(humidity) AS humidity, AVG(wind_speed) AS wind_speed, ' \
            'AVG(fuel_price) AS fuel, SUM(quantity) AS quantity, date(created_at) AS date, MAX(is_weekend) as weekend, MAX(is_holiday) as holiday FROM `sales`' \
            ' WHERE item_id = \'' + str(item_id) + '\' GROUP BY date(created_at), item_id'

    df = pd.read_sql(query, con=conn)

    df.columns = ['item_id', 'weather', 'temp', 'tmin', 'tmax', 'pressure', 'humidity', 'wind_speed', 'fuel',
                  'quantity',
                  'date','weekend','holiday']

    data_count = len(df)

    if data_count >= 6:
        df['weather'] = df.weather.astype(int)
        df['tmin'] = df.tmin.astype(int)
        df['tmax'] = df.tmax.astype(int)
        df['pressure'] = df.pressure.astype(int)
        df['humidity'] = df.humidity.astype(int)
        df['wind_speed'] = df.wind_speed.astype(int)
        df['fuel'] = df.fuel.astype(int)
        df['quantity'] = df.quantity.astype(int)
        df['weekend'] = df.weekend.astype(int)
        df['holiday'] = df.holiday.astype(int)

        xdf = df.drop(['item_id', 'date', 'quantity'], 1)
        ydf = np.asarray(df['quantity'])

        x = np.asarray(xdf)
        y = ydf

        X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        # Caches the trained data using pickle
        with open('ml_models/' + str(item_id) + '.pickle', 'wb') as f:
            pickle.dump(clf, f)

        np.savetxt('ml_test_data/' + str(item_id) + '_x_test.pickle', X_test)
        np.savetxt('ml_test_data/' + str(item_id) + '_y_test.pickle', y_test)

    else:
        print('Less data to deal with')


# x_lately = np.asarray([1, 298, 298, 298, 1014, 69, 4, 3807]).reshape(1, -1)


def ml_prediction(item_id, item_name, stock, data):
    x_lately = np.asarray(data).reshape(1, -1)
    try:
        datax = np.loadtxt('ml_test_data/' + str(item_id) + '_x_test.pickle')
        datay = np.loadtxt('ml_test_data/' + str(item_id) + '_y_test.pickle')

        X_test = datax
        y_test = datay

        try:
            pickle_in = open('ml_models/' + str(item_id) + '.pickle', 'rb')
            clf = pickle.load(pickle_in)

            accuracy = clf.score(X_test, y_test)
            forecast_set = clf.predict(x_lately)

            # return {'item_id': item_id, 'item_name': item_name, 'stock': stock, 'accuracy': accuracy, 'forecast': forecast_set[0]}
            return {'item_id': item_id, 'item_name': item_name, 'stock': stock, 'accuracy': accuracy, 'forecast': forecast_set[0]}

        except FileNotFoundError:
            print('No trained model for item ', item_id)

    except Exception:
        print('No test data for item ', item_id)


def get_prediction(data):
    x_lately = data
    forecasts = []
    for row, d in enumerate(data_list):
        data = ml_prediction(d[0], d[1], d[2], x_lately)
        forecasts.append(data)
    return forecasts 


def training():
    if length != 0:
        for row_number, d in enumerate(data_list):
            train_models(d[0])
        return "Training process finished"
    else:
        return "no items got"


# data =[1, 298, 298, 298, 1014, 69, 4, 3807,0,1]

# print(get_prediction(data))