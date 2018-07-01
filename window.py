# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!
import http.client
import json

from PyQt5 import QtCore, QtWidgets
import mysql.connector
import bcrypt
from PyQt5.QtWidgets import QMessageBox, QWidget, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import datetime
# from project.ml import * todo Uncomment to use pycharm debugger
# from project.admin_login import GUIForm

from ml import *
from admin_login import GUIForm
from config import *
from helpers import *

style.use('fivethirtyeight')
plt.rcParams['font.size'] = 9.0


class Ui_MainWindow(object):
    # Loads all inventory data from the database
    def load_inventory_data(self):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute('select inventory_stocks.id,items.codes,items.product_name,'
                       'inventory_stocks.received,inventory_stocks.sales,'
                       'inventory_stocks.stocks, inventory_stocks.total_expenditure_cost,'
                       'inventory_stocks.total_sales_cost, inventory_stocks.created_at,'
                       'inventory_stocks.updated_at from inventory_stocks '
                       'left join items on items.id = inventory_stocks.item_id')
        data_list = cursor.fetchall()
        rows = 1
        for row_number, d in enumerate(data_list):
            self.tableWidget.setRowCount(rows)
            self.tableWidget.insertRow(rows)
            for column_number, data in enumerate(d):
                self.tableWidget.setItem(rows, column_number, QtWidgets.QTableWidgetItem(str(data)))
            rows = rows + 1
        conn.close()

    def load_items_data(self):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute('select id, codes, product_name, units, unit_cost, unit_price, remarks, created_at, updated_at from items')
        data_list = cursor.fetchall()
        rows = 1
        for row_number, d in enumerate(data_list):
            self.tableWidget.setRowCount(rows)
            self.tableWidget.insertRow(rows)
            for column_number, data in enumerate(d):
                self.tableWidget.setItem(rows, column_number, QtWidgets.QTableWidgetItem(str(data)))
            rows = rows + 1
        conn.close()

    def load_sales_data(self):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute('select sales.id, items.product_name, CONCAT(operators.first_name,\' \', '
                       'operators.last_name) AS '
                       'name, sales.quantity, sales.total_price, sales.created_at, sales.updated_at '
                       'from sales left join items on items.id = sales.item_id left join operators '
                       'on operators.id = sales.operator_id order by created_at desc')
        data_list = cursor.fetchall()
        rows = 1
        for row_number, d in enumerate(data_list):
            self.tableWidget.setRowCount(rows)
            self.tableWidget.insertRow(rows)
            for column_number, data in enumerate(d):
                self.tableWidget.setItem(rows, column_number, QtWidgets.QTableWidgetItem(str(data)))
            rows = rows + 1
        conn.close()

    def load_received_data(self):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute('select received_products.id, items.product_name, CONCAT(operators.first_name,\' \', '
                       'operators.last_name) AS '
                       'name, received_products.quantity, received_products.total_price, '
                       'received_products.created_at, received_products.updated_at '
                       'from received_products left join items on '
                       'items.id = received_products.item_id left join operators '
                       'on operators.id = received_products.operator_id order by created_at desc')
        data_list = cursor.fetchall()
        rows = 1
        for row_number, d in enumerate(data_list):
            self.tableWidget.setRowCount(rows)
            self.tableWidget.insertRow(rows)
            for column_number, data in enumerate(d):
                self.tableWidget.setItem(rows, column_number, QtWidgets.QTableWidgetItem(str(data)))
            rows = rows + 1
        conn.close()

    def load_operators_data(self):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute('select id, first_name, last_name, auth_id, dob, created_at, updated_at from operators')
        data_list = cursor.fetchall()
        rows = 1
        for row_number, d in enumerate(data_list):
            self.tableWidget.setRowCount(rows)
            self.tableWidget.insertRow(rows)
            for column_number, data in enumerate(d):
                self.tableWidget.setItem(rows, column_number, QtWidgets.QTableWidgetItem(str(data)))
            rows = rows + 1
        conn.close()

    def inventory_table(self):
        MainWindow.setWindowTitle("Inventory")
        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setRowCount(5)
        self.tableWidget.setColumnCount(10)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 9, item)
        _translate = QtCore.QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "Stock ID"))
        item = self.tableWidget.item(0, 1)
        item.setText(_translate("MainWindow", "Codes"))
        item = self.tableWidget.item(0, 2)
        item.setText(_translate("MainWindow", "Product Name"))
        item = self.tableWidget.item(0, 3)
        item.setText(_translate("MainWindow", "Received"))
        item = self.tableWidget.item(0, 4)
        item.setText(_translate("MainWindow", "Sales"))
        item = self.tableWidget.item(0, 5)
        item.setText(_translate("MainWindow", "Stocks"))
        item = self.tableWidget.item(0, 6)
        item.setText(_translate("MainWindow", "Expenditure Cost"))
        item = self.tableWidget.item(0, 7)
        item.setText(_translate("MainWindow", "Sales Cost"))
        item = self.tableWidget.item(0, 8)
        item.setText(_translate("MainWindow", "Created At"))
        item = self.tableWidget.item(0, 9)
        item.setText(_translate("MainWindow", "Updated At"))

        # Loads the inventory data
        self.load_inventory_data()

        self.tableWidget.setSortingEnabled(__sortingEnabled)
        return self.tableWidget

    def items_table(self):
        MainWindow.setWindowTitle("Items list")
        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setRowCount(5)
        self.tableWidget.setColumnCount(9)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 8, item)
        _translate = QtCore.QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "Stock ID"))
        item = self.tableWidget.item(0, 1)
        item.setText(_translate("MainWindow", "Codes"))
        item = self.tableWidget.item(0, 2)
        item.setText(_translate("MainWindow", "Product name"))
        item = self.tableWidget.item(0, 3)
        item.setText(_translate("MainWindow", "Units"))
        item = self.tableWidget.item(0, 4)
        item.setText(_translate("MainWindow", "Unit cost"))
        item = self.tableWidget.item(0, 5)
        item.setText(_translate("MainWindow", "Unit price"))
        item = self.tableWidget.item(0, 6)
        item.setText(_translate("MainWindow", "Remarks"))
        item = self.tableWidget.item(0, 7)
        item.setText(_translate("MainWindow", "Created At"))
        item = self.tableWidget.item(0, 8)
        item.setText(_translate("MainWindow", "Updated At"))

        # Loads the inventory data
        self.load_items_data()

        self.tableWidget.setSortingEnabled(__sortingEnabled)
        return self.tableWidget

    def sales_table(self):
        MainWindow.setWindowTitle("Sales logs")
        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setRowCount(5)
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 6, item)
        _translate = QtCore.QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "Sale ID"))
        item = self.tableWidget.item(0, 1)
        item.setText(_translate("MainWindow", "Product"))
        item = self.tableWidget.item(0, 2)
        item.setText(_translate("MainWindow", "Operator"))
        item = self.tableWidget.item(0, 3)
        item.setText(_translate("MainWindow", "Quantity"))
        item = self.tableWidget.item(0, 4)
        item.setText(_translate("MainWindow", "Total price"))
        item = self.tableWidget.item(0, 5)
        item.setText(_translate("MainWindow", "Created At"))
        item = self.tableWidget.item(0, 6)
        item.setText(_translate("MainWindow", "Updated At"))

        # Loads the inventory data
        self.load_sales_data()

        self.tableWidget.setSortingEnabled(__sortingEnabled)
        return self.tableWidget

    def received_table(self):
        MainWindow.setWindowTitle("Received items logs")
        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setRowCount(5)
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 6, item)
        _translate = QtCore.QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "Received ID"))
        item = self.tableWidget.item(0, 1)
        item.setText(_translate("MainWindow", "Product"))
        item = self.tableWidget.item(0, 2)
        item.setText(_translate("MainWindow", "Operator"))
        item = self.tableWidget.item(0, 3)
        item.setText(_translate("MainWindow", "Quantity"))
        item = self.tableWidget.item(0, 4)
        item.setText(_translate("MainWindow", "Total price"))
        item = self.tableWidget.item(0, 5)
        item.setText(_translate("MainWindow", "Created At"))
        item = self.tableWidget.item(0, 6)
        item.setText(_translate("MainWindow", "Updated At"))

        # Loads the inventory data
        self.load_received_data()

        self.tableWidget.setSortingEnabled(__sortingEnabled)
        return self.tableWidget

    def operators_table(self):
        MainWindow.setWindowTitle("Operators")
        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setRowCount(5)
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 6, item)
        _translate = QtCore.QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "Operator ID"))
        item = self.tableWidget.item(0, 1)
        item.setText(_translate("MainWindow", "First name"))
        item = self.tableWidget.item(0, 2)
        item.setText(_translate("MainWindow", "Last name"))
        item = self.tableWidget.item(0, 3)
        item.setText(_translate("MainWindow", "Auth ID"))
        item = self.tableWidget.item(0, 4)
        item.setText(_translate("MainWindow", "Date of birth"))
        item = self.tableWidget.item(0, 5)
        item.setText(_translate("MainWindow", "Created At"))
        item = self.tableWidget.item(0, 6)
        item.setText(_translate("MainWindow", "Updated At"))

        # Loads the inventory data
        self.load_operators_data()

        self.tableWidget.setSortingEnabled(__sortingEnabled)
        return self.tableWidget

    def analytics_view(self):
        MainWindow.setWindowTitle("Real-time sales analysis")
        # self.figure = plt.figure(figsize=(15, 5))
        # self.canvas = FigureCanvas(self.figure)
        # self.verticalLayout_3.addWidget(self.canvas)
        #
        # plt.cla()
        # ax = self.figure.add_subplot(111)
        # x = [i for i in range(100)]
        # y = [i ** 2 for i in x]
        # ax.plot(x, y, 'b.')
        # ax.set_title('Quadratic Plot')
        # self.canvas.draw()

        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.verticalLayout_3.addWidget(self.canvas)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=1000)
        self.canvas.draw()

    def animate(self, i):
        items = []
        item_names = []
        codes = []
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host,
                                       database=self.database)

        # Load available item IDs
        cursor = conn.cursor()
        cursor.execute('select id, product_name from items')
        names = cursor.fetchall()

        for idx, n in enumerate(names):
            items.append(n[0])
            item_names.append(n[1])


        for i in items:
            cursor = conn.cursor()
            cursor.execute('select id, quantity from sales where item_id = '+str(i))
            data_list = cursor.fetchall()
            db_xs = []
            db_ys = []
            for row_number, d in enumerate(data_list):
                db_xs.append(d[0])
                db_ys.append(int(d[1]))
            icodes = {'x': db_xs, 'y': db_ys}
            codes.append(icodes)
        conn.close()


        self.ax1.clear()
        num_plots = len(codes)
        colormap = plt.cm.gist_ncar
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
        labels = []
        for row, j in enumerate(codes):
            # self.ax1.plot(xs, ys)
            self.ax1.plot(j['x'], j['y'])
            labels.append(item_names[row])

        plt.legend(labels, loc='upper left')
        # plt.legend(labels, loc='upper left')
        # plt.legend(labels, ncol=4, loc='upper center',
        #            bbox_to_anchor=[0.5, 1.1],
        #            columnspacing=1.0, labelspacing=0.0,
        #            handletextpad=0.0, handlelength=1.5,
        #            fancybox=True, shadow=True)

        self.ax1.set_title("Real-time Sales Analysis")
        self.ax1.set_xlabel("Sales")
        self.ax1.set_ylabel("Quantity")

    def static_analytics_view(self):
        MainWindow.setWindowTitle("Static analysis")
        self.pie_charts = QtWidgets.QFrame(self.frame_2)
        self.pie_charts.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.pie_charts.setFrameShadow(QtWidgets.QFrame.Raised)
        self.pie_charts.setObjectName("pie_charts")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.pie_charts)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.sales_chart = QtWidgets.QFrame(self.pie_charts)
        self.sales_chart.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.sales_chart.setFrameShadow(QtWidgets.QFrame.Raised)
        self.sales_chart.setObjectName("sales_chart")
        self.verticalLayout_sales = QtWidgets.QVBoxLayout(self.sales_chart)
        self.verticalLayout_sales.setObjectName("verticalLayout_sales")
        self.sales_pie_vertical_layout = QtWidgets.QVBoxLayout()
        self.sales_pie_vertical_layout.setObjectName("sales_pie_vertical_layout")
        self.verticalLayout_sales.addLayout(self.sales_pie_vertical_layout)
        self.horizontalLayout_2.addWidget(self.sales_chart)
        self.sales_pie()

        self.expenditure_chart = QtWidgets.QFrame(self.pie_charts)
        self.expenditure_chart.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.expenditure_chart.setFrameShadow(QtWidgets.QFrame.Raised)
        self.expenditure_chart.setObjectName("expenditure_chart")
        self.verticalLayout_exp = QtWidgets.QVBoxLayout(self.expenditure_chart)
        self.verticalLayout_exp.setObjectName("verticalLayout_exp")
        self.exp_pie_vertical_layout = QtWidgets.QVBoxLayout()
        self.exp_pie_vertical_layout.setObjectName("exp_pie_vertical_layout")
        self.verticalLayout_exp.addLayout(self.exp_pie_vertical_layout)
        self.horizontalLayout_2.addWidget(self.expenditure_chart)
        self.expenditure_pie()

        self.verticalLayout_3.addWidget(self.pie_charts)

        self.sales_vs_exp_chart = QtWidgets.QFrame(self.frame_2)
        self.sales_vs_exp_chart.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.sales_vs_exp_chart.setFrameShadow(QtWidgets.QFrame.Raised)
        self.sales_vs_exp_chart.setObjectName("sales_vs_exp_chart")
        self.verticalLayout_comp = QtWidgets.QVBoxLayout(self.sales_vs_exp_chart)
        self.verticalLayout_comp.setObjectName("verticalLayout_comp")
        self.comp_vertical_layout = QtWidgets.QVBoxLayout()
        self.comp_vertical_layout.setObjectName("exp_pie_vertical_layout")
        self.verticalLayout_comp.addLayout(self.comp_vertical_layout)
        self.verticalLayout_3.addWidget(self.sales_vs_exp_chart)

        self.sales_exp()

    def sales_pie(self):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute('select items.product_name, inventory_stocks.received, inventory_stocks.sales '
                       'from items left join inventory_stocks on items.id = inventory_stocks.item_id')
        data_list = cursor.fetchall()
        names = []
        sales_percent = []
        for row_number, d in enumerate(data_list):
            names.append(d[0])
            # Calculate sales percentage
            try:
                sales_percent.append(int((((d[1] - d[2]) / d[1]) * 100)))
            except ZeroDivisionError:
                sales_percent.append(0)
        conn.close()

        # labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'  # todo Load sales vs item_name from inventory table
        labels = names
        # sizes = [15, 30, 45, 10]
        sizes = sales_percent
        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        self.fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        self.canvas_sales_pie = FigureCanvas(self.fig1)
        self.sales_pie_vertical_layout.addWidget(self.canvas_sales_pie)
        ax1.set_title("Sales")
        self.canvas_sales_pie.draw()

    def expenditure_pie(self):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute('select items.product_name, inventory_stocks.received, inventory_stocks.stocks '
                       'from items left join inventory_stocks on items.id = inventory_stocks.item_id')
        data_list = cursor.fetchall()
        names = []
        stocks_percent = []
        for row_number, d in enumerate(data_list):
            names.append(d[0])
            # Calculate sales percentage
            try:
                stocks_percent.append(int((((d[1] - d[2]) / d[1]) * 100)))
            except ZeroDivisionError:
                stocks_percent.append(0)
        conn.close()

        # labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'  # todo Load expenditure vs item_name from inventory table
        labels = names
        # sizes = [15, 30, 45, 10]
        sizes = stocks_percent

        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        self.fig2, ax1 = plt.subplots()
        # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        #         shadow=True, startangle=90)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        self.canvas_exp_pie = FigureCanvas(self.fig2)
        self.exp_pie_vertical_layout.addWidget(self.canvas_exp_pie)
        # ax1.title('Raining Hogs and Dogs', bbox={'facecolor': '0.8', 'pad': 5})
        ax1.set_title("Stock")
        self.canvas_exp_pie.draw()

    def sales_exp(self):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute('select items.product_name, inventory_stocks.total_expenditure_cost, '
                       'inventory_stocks.total_sales_cost '
                       'from items left join inventory_stocks on items.id = inventory_stocks.item_id')
        data_list = cursor.fetchall()
        names = []
        total_exp = []
        total_sales = []
        for row_number, d in enumerate(data_list):
            names.append(d[0])
            total_exp.append(int(d[1]))
            total_sales.append(int(d[2]))
        conn.close()
        # data to plot
        n_groups = len(data_list)
        # means_frank = (90, 55, 40, 65)
        # means_guido = (85, 62, 54, 20)
        means_frank = total_exp
        means_guido = total_sales

        # create plot
        self.fig3, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, means_frank, bar_width,
                         alpha=opacity,
                         color='b',
                         label='Expenditure')

        rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                         alpha=opacity,
                         color='g',
                         label='Sales')

        plt.xlabel('Products')
        plt.ylabel('Price(UGX)')
        plt.title('Sales vs Expenditure')
        # plt.xticks(index + bar_width, ('Ug', 'Empire', 'Zed', 'London'))
        plt.xticks(index + bar_width, tuple(names))
        plt.legend()

        plt.tight_layout()
        self.canvas_comp = FigureCanvas(self.fig3)
        self.comp_vertical_layout.addWidget(self.canvas_comp)
        self.canvas_comp.draw()
        # plt.show()
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowState(QtCore.Qt.WindowMaximized)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("QFrame.SidebarFrame {\n"
                                         "border-top-left-radius: 10px;\n"
                                         "border-top-right-radius: 10px;\n"
                                         "border: 1px solid black;\n"
                                         "background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
                                         "stop: 0 #56a, stop: 0.1 #016);\n"
                                         "}")
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMaximumSize(QtCore.QSize(200, 16777215))
        # self.frame.setStyleSheet("background:rgb(178, 208, 255)")
        self.frame.setStyleSheet("background:rgb(85, 113, 156)")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setProperty("class", "")
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.frame_3)
        self.label.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.frame_4 = QtWidgets.QFrame(self.frame_3)
        self.frame_4.setStyleSheet("border-color:green;")
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.frame_4)
        # self.pushButton.setStyleSheet("color:red;\n"
        #                               "background-color:blue;")
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        self.pushButton_operators = QtWidgets.QPushButton(self.frame_4)
        self.pushButton_operators.setObjectName("pushButton_operators")
        self.verticalLayout.addWidget(self.pushButton_operators)
        self.verticalLayout_2.addWidget(self.frame_4)
        self.verticalLayout_4.addWidget(self.frame_3)

        # Analytics buttons
        self.frame_5 = QtWidgets.QFrame(self.frame)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_6 = QtWidgets.QFrame(self.frame_5)
        self.frame_6.setStyleSheet("border-color:green;")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.frame_6)
        # self.pushButton_6.setStyleSheet("color:red;\n"
        #                                 "background-color:blue;")
        # self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_5.addWidget(self.pushButton_6)
        self.pushButton_5 = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout_5.addWidget(self.pushButton_5)
        self.pushButton_7 = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_5.addWidget(self.pushButton_7)
        self.pushButton_8 = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_8.setObjectName("pushButton_8")
        self.verticalLayout_5.addWidget(self.pushButton_8)
        '''self.pushButton_9 = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_9.setObjectName("pushButton_9")
        self.verticalLayout_5.addWidget(self.pushButton_9)'''
        self.operator_analysis_btn = QtWidgets.QPushButton(self.frame_6)
        self.operator_analysis_btn.setObjectName("operator_analysis_btn") #todo Operator analytics button object
        self.verticalLayout_5.addWidget(self.operator_analysis_btn)
        self.pushButton_10 = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_10.setObjectName("pushButton_10")
        self.verticalLayout_5.addWidget(self.pushButton_10)
        self.verticalLayout_6.addWidget(self.frame_6)
        self.verticalLayout_4.addWidget(self.frame_5)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem)
        self.horizontalLayout.addWidget(self.frame)

        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setStyleSheet("background-color:rgb(207, 255, 180)")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        # Loads first page, items page

        self.btn_analytics_click()

        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.horizontalLayout.addWidget(self.frame_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuHelp.addAction(self.actionAbout)
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.actionAbout.setText("About")
        self.actionExit.setText("Exit")

        # Assigns button click callbacks
        self.pushButton.clicked.connect(self.btn_items_click)
        self.pushButton_2.clicked.connect(self.btn_sales_click)
        self.pushButton_3.clicked.connect(self.btn_received_click)
        self.pushButton_4.clicked.connect(self.btn_inventory_click)
        self.pushButton_5.clicked.connect(self.btn_analytics_click)
        self.pushButton_6.clicked.connect(self.btn_static_analytics_click)
        # self.pushButton_7.clicked.connect(self.btn_predictive_analysis)
        self.pushButton_7.clicked.connect(self.view_predictive_analysis)
        self.pushButton_8.clicked.connect(self.btn_operator_performance_click)
        self.operator_analysis_btn.clicked.connect(self.btn_operator_analytics_click)
        # self.pushButton_9.clicked.connect(self.btn_admins)
        self.pushButton_operators.clicked.connect(self.btn_operators_click)
        self.pushButton_10.clicked.connect(self.btn_logout_click)

        # Menu bar action events
        self.actionAbout.triggered.connect(self.menu_about_click) #todo Added menu bar action triggers
        self.actionExit.triggered.connect(self.menu_exit_click)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    #todo Menu bar action trigger methods
    def menu_exit_click(self):
        exit(0)

    def menu_about_click(self):
        self.dialog = QtWidgets.QDialog()
        self.dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.aboutSystemDialogUi(self.dialog)
        self.dialog.show()

    def aboutSystemDialogUi(self, Dialog): #todo About system dialog
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(140, 70, 111, 31))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(30, 100, 351, 51))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(170, 180, 71, 20))
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(160, 240, 80, 23))
        self.pushButton.setObjectName("pushButton")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(66, 150, 251, 20))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")

        self.aboutRetranslateUi(Dialog)

        self.pushButton.clicked.connect(self.btn_close_about)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def aboutRetranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "About"))
        self.label.setText(_translate("Dialog", "<h1>BORTEC</h1>"))
        self.label_2.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:x-large; font-weight:600;\">Inventory management system</span></p></body></html>"))
        self.label_3.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:small; font-weight:600;\">Version </span><span style=\" font-size:small; font-weight:600; color:#ff0000;\">1.0</span></p></body></html>"))
        self.pushButton.setText(_translate("Dialog", "Cancel"))
        self.label_4.setText(_translate("Dialog", "<h3>With Predictive Sales Analysis</h3>"))

    def btn_close_about(self):
        self.dialog.close()

    '''
    Side menu button events handlers
    '''

    #todo Operator analytis code
    def btn_operator_analytics_click(self):
        for i in reversed(range(self.verticalLayout_3.count())):
            self.verticalLayout_3.itemAt(i).widget().setParent(None)
        self.operator_analytics_view()

    def operator_analytics_view(self): #todo Operator analytics 1
        MainWindow.setWindowTitle("Operator analysis")
        self.frame_6 = QtWidgets.QFrame(self.frame_2)
        self.frame_6.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.comboBox = QtWidgets.QComboBox(self.frame_6)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout_2.addWidget(self.comboBox)
        self.item_combo = QtWidgets.QComboBox(self.frame_6)
        self.item_combo.setObjectName("item_combo")
        self.horizontalLayout_2.addWidget(self.item_combo)
        self.dateRange = QtWidgets.QLabel(self.frame_6)
        self.dateRange.setObjectName("dateRange")
        self.horizontalLayout_2.addWidget(self.dateRange)
        self.dateFrom = QtWidgets.QLineEdit(self.frame_6)
        self.dateFrom.setObjectName("dateFrom")
        self.horizontalLayout_2.addWidget(self.dateFrom)
        self.label_3 = QtWidgets.QLabel(self.frame_6)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.dateTo = QtWidgets.QLineEdit(self.frame_6)
        self.dateTo.setObjectName("dateTo")
        self.horizontalLayout_2.addWidget(self.dateTo)
        self.runFilter = QtWidgets.QPushButton(self.frame_6)
        self.runFilter.setObjectName("runFilter")
        self.horizontalLayout_2.addWidget(self.runFilter)
        spacerItem = QtWidgets.QSpacerItem(500, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout_3.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(self.frame_2)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_3.addWidget(self.frame_7)
        self.horizontalLayout.addWidget(self.frame_2)

        self.dateFrom.setPlaceholderText("e.g. 2018-5-1")
        self.dateTo.setPlaceholderText("e.g 2018-12-1")
        self.dateRange.setText("Date Range:")
        self.label_3.setText("To")
        self.runFilter.setText("Filter")

        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host,
                                       database=self.database)
        cursor = conn.cursor()
        cursor.execute('select id, first_name, last_name from operators')
        data_list = cursor.fetchall()

        cursor.execute('select id, first_name, last_name from operators')
        data_list = cursor.fetchall()
        print(data_list)

        self.combo_items = {}
        self.comboBox.addItem("All operators")
        for row_number, d in enumerate(data_list):
            self.combo_items[str(d[1])] = str(d[0])
            self.comboBox.addItem(str(d[1]))

        cursor.execute('select id, product_name from items')
        data_list_items = cursor.fetchall()
        print(data_list_items)

        self.item_combo_items = {}
        self.item_combo.addItem("All items")
        for row_number, d in enumerate(data_list_items):
            self.item_combo_items[str(d[1])] = str(d[0])
            self.item_combo.addItem(str(d[1]))

        self.runFilter.clicked.connect(self.filter_operator_result)

        query = 'SELECT SUM(quantity) as total, DATE (created_at) FROM sales GROUP BY DATE (created_at);'
        self.operator_an_view(query)

    def filter_operator_result(self): #todo Operator analytics 2
        combo_item = ''
        comboItemName = str(self.comboBox.currentText())
        if comboItemName != 'All operators':
            combo_item = self.combo_items[comboItemName]

        item_combo_item = ''
        itemComboItemName = str(self.item_combo.currentText())
        if itemComboItemName != 'All items':
            item_combo_item = self.item_combo_items[itemComboItemName]

        date_from = str(self.dateFrom.text()).strip()
        date_to = str(self.dateTo.text()).strip()

        if combo_item or item_combo_item or date_from or date_to:
            dic = {}

            if combo_item:
                dic["operator"] = 'operator_id = \'' + combo_item.strip() + '\''

            if item_combo_item:
                dic["item"] = 'item_id = \'' + item_combo_item.strip() + '\''


            if date_from:
                dic["date_from"] = 'DATE(created_at) >=\'' + date_from.strip() + '\''

            if date_to:
                dic["date_to"] = 'DATE(created_at) <=\'' + date_to.strip() + '\''


            like_list = []
            for key in dic:
                like_list.append(dic[key])

            like_query = ' AND '.join(like_list)

            print("Query ", like_query)
            query = 'SELECT SUM(quantity) as total, DATE (created_at) FROM sales WHERE '+like_query+' GROUP BY DATE (created_at);'

        else:
            query = 'SELECT SUM(quantity) as total, DATE (created_at) FROM sales GROUP BY DATE (created_at);'

        self.operator_an_view(query)

    def operator_an_view(self, query): #todo Operator analytics 3
        for i in reversed(range(self.verticalLayout.count())):
            self.verticalLayout.itemAt(i).widget().setParent(None)

        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()


        cursor.execute(query)

        data_list = cursor.fetchall()
        print(len(data_list))

        if len(data_list) > 0:
            date = []
            total_sales = []
            for row_number, d in enumerate(data_list):
                total_sales.append(int(d[0]))
                date.append(str(d[1]))
            conn.close()

            x = np.arange(len(data_list))
            money = total_sales
            plt.clf()
            self.fig4, ax4 = plt.subplots()
            plt.bar(x, money)
            plt.legend('Sales', loc='upper right')
            plt.xticks(x, tuple(date))
            plt.tick_params(axis='both', which='major', labelsize=6)
            plt.tick_params(axis='both', which='minor', labelsize=6)
            plt.xlabel('Date', fontsize=8)
            plt.ylabel('Total sales', fontsize=8)
            plt.title('Operator Sales Analysis', fontsize=12)
            self.canvas_operators = FigureCanvas(self.fig4)
            self.verticalLayout.addWidget(self.canvas_operators)
            self.canvas_operators.draw()
        else:
            notification('No data available')


    def btn_logout_click(self):
        self.show_login_dialog()

    def btn_inventory_click(self):
        for i in reversed(range(self.verticalLayout_3.count())):
            self.verticalLayout_3.itemAt(i).widget().setParent(None)
        self.verticalLayout_3.addWidget(self.inventory_table())

    def btn_items_click(self):
        for i in reversed(range(self.verticalLayout_3.count())):
            self.verticalLayout_3.itemAt(i).widget().setParent(None)
        # self.verticalLayout_3.addWidget(self.items_table())
        self.frame_6 = QtWidgets.QFrame(self.frame_2)
        self.frame_6.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.addItem = QtWidgets.QPushButton(self.frame_6)
        self.addItem.setText("Add item")
        self.addItem.setObjectName("addItem")
        self.deleteItem = QtWidgets.QPushButton(self.frame_6)
        self.deleteItem.setText("Delete item")
        self.deleteItem.setObjectName("deleteItem")
        self.editItem = QtWidgets.QPushButton(self.frame_6)
        self.editItem.setText("Edit item")
        self.editItem.setObjectName("editItem")
        self.horizontalLayout_2.addWidget(self.addItem)
        self.horizontalLayout_2.addWidget(self.deleteItem)
        self.horizontalLayout_2.addWidget(self.editItem)
        spacerItem = QtWidgets.QSpacerItem(652, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout_3.addWidget(self.frame_6)

        ## Lower frame
        self.frame_7 = QtWidgets.QFrame(self.frame_2)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")

        # loads operators data within table
        self.verticalLayout_5.addWidget(self.items_table())
        self.verticalLayout_6.addLayout(self.verticalLayout_5)
        self.verticalLayout_3.addWidget(self.frame_7)
        self.addItem.clicked.connect(self.btn_open_add_item_dialog)
        self.deleteItem.clicked.connect(self.btn_open_delete_item_dialog)
        self.editItem.clicked.connect(self.btn_open_edit_item_dialog)

    def btn_open_edit_item_dialog(self):
        self.edit_dialog = QtWidgets.QDialog()
        self.edit_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setupEditUi(self.edit_dialog)
        self.edit_dialog.show()

    def setupEditUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.listWidget = QtWidgets.QListWidget(self.frame)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_2.addWidget(self.listWidget)
        self.editItem = QtWidgets.QPushButton(self.frame)
        self.editItem.setObjectName("editItem")
        self.verticalLayout_2.addWidget(self.editItem)
        self.verticalLayout.addWidget(self.frame)

        self.load_items_list_dialog_data()

        self.retranslateEditUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.editItem.clicked.connect(self.edit_item)

    def retranslateEditUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Select item to edit"))
        self.editItem.setText(_translate("Dialog", "Edit Item"))

    def edit_item(self):
        self.item_name = self.listWidget.currentItem().text()
        self.save_edit_dialog = QtWidgets.QDialog()
        self.save_edit_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.EditItemDetailsDialogUi(self.save_edit_dialog, self.item_name)
        self.save_edit_dialog.show()

    def EditItemDetailsDialogUi(self, Dialog, item_name):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.itemName = QtWidgets.QLineEdit(self.frame)
        self.itemName.setObjectName("itemName")
        self.horizontalLayout.addWidget(self.itemName)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.UnitOfMeasurement = QtWidgets.QLineEdit(self.frame)
        self.UnitOfMeasurement.setObjectName("UnitOfMeasurement")
        self.horizontalLayout_2.addWidget(self.UnitOfMeasurement)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.unitPrice = QtWidgets.QLineEdit(self.frame)
        self.unitPrice.setText("")
        self.unitPrice.setObjectName("unitPrice")
        self.horizontalLayout_3.addWidget(self.unitPrice)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.codes = QtWidgets.QLineEdit(self.frame)
        self.codes.setText("")
        self.codes.setObjectName("codes")
        self.horizontalLayout_4.addWidget(self.codes)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.remarks = QtWidgets.QTextEdit(self.frame)
        self.remarks.setObjectName("remarks")
        self.horizontalLayout_5.addWidget(self.remarks)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addWidget(self.frame)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateEditItemDetailsUi(Dialog, item_name)
        self.buttonBox.accepted.connect(self.dialog_accepted_save_edit)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateEditItemDetailsUi(self, Dialog, item_name):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Add Item"))
        self.label.setText(_translate("Dialog", "Item name:"))
        self.label_2.setText(_translate("Dialog", "Units of measurement:"))
        self.label_3.setText(_translate("Dialog", "Unit price: "))
        self.label_4.setText(_translate("Dialog", "Codes: "))
        self.label_5.setText(_translate("Dialog", "Remarks: "))

        self.item_edit_name = item_name

        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        query = 'select id, codes, product_name, units, unit_price, remarks, created_at, updated_at from items where ' \
                'product_name = \'' + item_name + '\''
        cursor.execute(query)
        data_list = cursor.fetchall()
        print(data_list)
        for row_number, d in enumerate(data_list):
            self.itemName.setText(_translate("Dialog", d[2]))
            self.UnitOfMeasurement.setText(_translate("Dialog", d[3]))
            self.unitPrice.setText(_translate("Dialog", str(d[4])))
            self.codes.setText(_translate("Dialog", d[1]))
            self.remarks.setText(_translate("Dialog", d[5]))
        conn.close()

    def dialog_accepted_save_edit(self):
        self.item_name = str(self.itemName.text()).strip()
        self.units_of_measurement = str(self.UnitOfMeasurement.text()).strip()
        self.unit_price = str(self.unitPrice.text()).strip()
        self.code = str(self.codes.text()).strip()
        self.remark = str(self.remarks.toPlainText()).strip()

        self.save_edit_details(self.item_name, self.units_of_measurement, self.unit_price, self.code, self.remark)
        self.save_edit_dialog.close()
        # Reloads the items table list
        self.listWidget.clear()
        self.load_items_list_dialog_data()
        self.btn_items_click()

    def save_edit_details(self, item_name, units_of_measurement, unit_price, code, remarks):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        query = 'update items set codes=\'' + code + '\',product_name=\'' + \
                item_name + '\',units=\'' + units_of_measurement + '\',unit_price=\'' + unit_price + '\'' \
                                                                                                     ',remarks=\'' + remarks + '\' where product_name=\'' + self.item_edit_name + '\''

        print(query)
        try:
            cursor.execute(query)
            conn.commit()
        except mysql.connector.Error as err:
            print("Something went wrong: {}".format(err))
            conn.rollback()

        conn.close()

    # def delete_item(self):
    #     # print("Delete item", self.listWidget.currentItem().text())
    #     # print("Delete item", self.listWidget.item(1).text())
    #     self.del_item_name = self.listWidget.currentItem().text()
    #     self.delete_from_db(self.del_item_name)

    def btn_open_delete_item_dialog(self):
        self.del_dialog = QtWidgets.QDialog()
        self.del_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setupDeleteUi(self.del_dialog)
        self.del_dialog.show()

    def setupDeleteUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.listWidget = QtWidgets.QListWidget(self.frame)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_2.addWidget(self.listWidget)
        self.deleteItem = QtWidgets.QPushButton(self.frame)
        self.deleteItem.setObjectName("deleteItem")
        self.verticalLayout_2.addWidget(self.deleteItem)
        self.verticalLayout.addWidget(self.frame)

        self.load_items_list_dialog_data()

        self.retranslateDeleteUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.deleteItem.clicked.connect(self.delete_item)

    def delete_item(self):
        # print("Delete item", self.listWidget.currentItem().text())
        # print("Delete item", self.listWidget.item(1).text())
        self.del_item_name = self.listWidget.currentItem().text()
        self.delete_from_db(self.del_item_name)

    def delete_from_db(self, item_name):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        query = 'delete from items where product_name=\'' + item_name + '\''
        try:
            cursor.execute(query)
            conn.commit()
            self.del_dialog.close()
            self.btn_items_click()
        except mysql.connector.Error as err:
            print("Something went wrong: {}".format(err))
            conn.rollback()

        conn.close()

    def retranslateDeleteUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Select item to delete"))
        self.deleteItem.setText(_translate("Dialog", "Delete Item"))

    def load_items_list_dialog_data(self):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute('select id, product_name from items')
        data_list = cursor.fetchall()
        # item = QtWidgets.QListWidgetItem()
        for row_number, d in enumerate(data_list):
            self.listWidget.addItem(QtWidgets.QListWidgetItem(d[1]))
        conn.close()

    def btn_sales_click(self):
        for i in reversed(range(self.verticalLayout_3.count())):
            self.verticalLayout_3.itemAt(i).widget().setParent(None)
        self.verticalLayout_3.addWidget(self.sales_table())

    def btn_received_click(self):
        for i in reversed(range(self.verticalLayout_3.count())):
            self.verticalLayout_3.itemAt(i).widget().setParent(None)
        self.verticalLayout_3.addWidget(self.received_table())

    def btn_analytics_click(self):
        for i in reversed(range(self.verticalLayout_3.count())):
            self.verticalLayout_3.itemAt(i).widget().setParent(None)
        self.analytics_view()

    def btn_static_analytics_click(self):
        for i in reversed(range(self.verticalLayout_3.count())):
            self.verticalLayout_3.itemAt(i).widget().setParent(None)
        self.static_analytics_view()

    def view_predictive_analysis(self):
        now = datetime.datetime.now()

        for i in reversed(range(self.verticalLayout_3.count())):
            self.verticalLayout_3.itemAt(i).widget().setParent(None)
        MainWindow.setWindowTitle("Predictive analytics")
        self.frame_6 = QtWidgets.QFrame(self.frame_2)
        self.frame_6.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.trainModel = QtWidgets.QPushButton(self.frame_6)
        self.trainModel.setObjectName("trainModel")
        self.horizontalLayout_2.addWidget(self.trainModel)
        self.runPredictions = QtWidgets.QPushButton(self.frame_6)
        self.runPredictions.setObjectName("runPredictions")
        self.horizontalLayout_2.addWidget(self.runPredictions)
        self.showChart = QtWidgets.QPushButton(self.frame_6)
        self.showChart.setObjectName("showChart")
        self.horizontalLayout_2.addWidget(self.showChart)
        spacerItem = QtWidgets.QSpacerItem(652, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout_3.addWidget(self.frame_6)
        self.frame = QtWidgets.QFrame(self.frame_2)
        self.frame.setMaximumSize(QtCore.QSize(16777215, 150))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)

        self.verticalLayout.addWidget(self.external_conditions_view())

        self.label.raise_()
        self.tableWidget.raise_()
        self.verticalLayout_3.addWidget(self.frame)
        self.frame_7 = QtWidgets.QFrame(self.frame_2)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.frame_7)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)


        self.verticalLayout_2.addWidget(self.predictions_table())

        self.verticalLayout_3.addWidget(self.frame_7)

        self.runPredictions.setText("Run prediction")
        self.trainModel.setText("Train model")
        self.showChart.setText("Show chart")
        self.label.setText("Current factors for "+now.strftime("%Y-%m-%d"))
        self.label_2.setText("Predictions for "+now.strftime("%Y-%m-%d"))

        self.trainModel.clicked.connect(self.btn_train_predictive_model)
        self.runPredictions.clicked.connect(self.btn_predictive_analysis)
        self.showChart.clicked.connect(self.btn_show_chart)

        # Run predictions
        self.btn_predictive_analysis()

    def btn_train_predictive_model(self):
        notification(training())        

    def btn_show_chart(self):
        plt.clf()
        now = datetime.datetime.now()
        objects = tuple(self.names)
        y_pos = np.arange(len(objects))
        performance = self.sales

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.legend('Sales', loc='upper left')
        plt.xticks(y_pos, objects)
        plt.ylabel('Sales')
        plt.xlabel('Items')
        plt.title('Sales predictions as of '+now.strftime("%Y-%m-%d"))

        plt.show()

    def predictions_table(self):
        self.tableWidget_2 = QtWidgets.QTableWidget(self.frame_7)
        self.tableWidget_2.setRowCount(5)
        self.tableWidget_2.setColumnCount(5)
        # self.tableWidget_2.verticalHeader().setVisible(False)
        # self.tableWidget_2.horizontalHeader().setVisible(False)
        self.tableWidget_2.setObjectName("tableWidget_2")
        self.verticalLayout_2.addWidget(self.tableWidget_2)

        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setItem(0, 4, item)

        _translate = QtCore.QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        __sortingEnabled = self.tableWidget_2.isSortingEnabled()
        self.tableWidget_2.setSortingEnabled(False)
        item = self.tableWidget_2.item(0, 0)
        item.setText(_translate("MainWindow", "Item"))
        item = self.tableWidget_2.item(0, 1)
        item.setText(_translate("MainWindow", "Name"))
        item = self.tableWidget_2.item(0, 2)
        item.setText(_translate("MainWindow", "Sales"))
        item = self.tableWidget_2.item(0, 3)
        item.setText(_translate("MainWindow", "Current stock"))
        item = self.tableWidget_2.item(0, 4)
        item.setText(_translate("MainWindow", "Status"))

        return self.tableWidget_2

    def external_conditions_view(self):
        self.tableWidget = QtWidgets.QTableWidget(self.frame)
        self.tableWidget.setRowCount(2)
        self.tableWidget.setColumnCount(10)
        # self.tableWidget.verticalHeader().setVisible(False)
        # self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.setObjectName("tableWidget")

        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(0, 9, item)

        _translate = QtCore.QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "Weather"))
        item = self.tableWidget.item(0, 1)
        item.setText(_translate("MainWindow", "Temperature (K)"))
        item = self.tableWidget.item(0, 2)
        item.setText(_translate("MainWindow", "Minimum temperature (K)"))
        item = self.tableWidget.item(0, 3)
        item.setText(_translate("MainWindow", "Maximum temeprature (K)"))
        item = self.tableWidget.item(0, 4)
        item.setText(_translate("MainWindow", "Pressure"))
        item = self.tableWidget.item(0, 5)
        item.setText(_translate("MainWindow", "Humidity"))
        item = self.tableWidget.item(0, 6)
        item.setText(_translate("MainWindow", "Wind_speed"))
        item = self.tableWidget.item(0, 7)
        item.setText(_translate("MainWindow", "Fuel price"))
        item = self.tableWidget.item(0, 8)
        item.setText(_translate("MainWindow", "is weekend"))
        item = self.tableWidget.item(0, 9)
        item.setText(_translate("MainWindow", "is holiday"))

        self.load_external_conditions()

        return self.tableWidget

    def load_external_conditions(self):
        weather = None
        temp = 0
        pressure = 0
        humidity = 0
        temp_min = 0
        temp_max = 0
        wind_speed = 0
        fuel = 0
        w = 0

        try:
            connection = http.client.HTTPConnection(BORTEC_API)
            connection.request('GET', '/api/external-apis')
            response = connection.getresponse()
            data = response.read().decode()
            json_data = json.loads(data)
            print(json_data['weather'])

            weather = json_data['weather']
            

            if weather == '':
                weather = None

            x = weather
            
            if x == 'Clouds':
                w = 1
            elif x == 'Thunderstorm':
                w = 2
            elif x == 'Rain':
                w =3
            elif x == 'Drizzle':
                w = 4
            elif x == 'Sun':
                w =5
            else:
                w = 0

            temp = json_data['temp']
            pressure = float(json_data['pressure'])
            humidity = float(json_data['humidity'])
            temp_min = json_data['temp_min']
            temp_max = json_data['temp_max']
            wind_speed = float(json_data['wind_speed'])
            fuel = float(json_data['fuel'])
            w = float(w)
        except Exception:
            notification('Unable to reach api servers')


        current_date = datetime.datetime.now().strftime("%d%m")
        if current_date == "0101" or current_date == "0101" or current_date == "2601" or current_date == "1602" or current_date == "0803" or current_date == "0105" or current_date == "0306" or current_date == "0906" or current_date == "0910" or current_date == "2512" or current_date == "2612":
            is_holiday = 1;
            holiday = True
        else:
            is_holiday = 0;
            holiday = False

        day_of_week =  datetime.date.today().strftime("%A")

        if day_of_week == 'Saturday' or day_of_week == 'Sunday':
            is_weekend = 1
            wkn = True
        else:
            is_weekend = 0
            wkn = False
                    

        self.tableWidget.setItem(1, 0, QtWidgets.QTableWidgetItem(str(weather)))
        self.tableWidget.setItem(1, 1, QtWidgets.QTableWidgetItem(str(temp)))
        self.tableWidget.setItem(1, 2, QtWidgets.QTableWidgetItem(str(temp_min)))
        self.tableWidget.setItem(1, 3, QtWidgets.QTableWidgetItem(str(temp_max)))
        self.tableWidget.setItem(1, 4, QtWidgets.QTableWidgetItem(str(pressure)))
        self.tableWidget.setItem(1, 5, QtWidgets.QTableWidgetItem(str(humidity)))
        self.tableWidget.setItem(1, 6, QtWidgets.QTableWidgetItem(str(wind_speed)))
        self.tableWidget.setItem(1, 7, QtWidgets.QTableWidgetItem(str(fuel)))
        self.tableWidget.setItem(1, 8, QtWidgets.QTableWidgetItem(str(wkn)))
        self.tableWidget.setItem(1, 9, QtWidgets.QTableWidgetItem(str(holiday)))

        self.d_apis = [w, temp, temp_min, temp_max, pressure, humidity, wind_speed, fuel, is_weekend, is_holiday]

    
    def btn_predictive_analysis(self):
        forecasts = get_prediction(self.d_apis)
        rows = 1
        self.names = []
        self.sales = []
        for row_number, d in enumerate(forecasts):
            self.tableWidget_2.setRowCount(rows)
            self.tableWidget_2.insertRow(rows)
            self.tableWidget_2.setItem(rows, 0, QtWidgets.QTableWidgetItem(str(d['item_id'])))
            self.tableWidget_2.setItem(rows, 1, QtWidgets.QTableWidgetItem(str(d['item_name'])))
            self.tableWidget_2.setItem(rows, 2, QtWidgets.QTableWidgetItem(str(int(d['forecast']))))
            self.tableWidget_2.setItem(rows, 3, QtWidgets.QTableWidgetItem(str(int(d['stock']))))
            print("Accuracy for {0} is {1}".format(d['item_name'],d['accuracy']))

            if d['forecast'] > d['stock']:
                status = 'Low stock'
            elif d['forecast'] < d['stock']:
                status = 'In stock'
            elif d['forecast'] == d['stock']:
                status = 'Restock'
            else:
                status = 'Indeterminable'

            self.tableWidget_2.setItem(rows, 4, QtWidgets.QTableWidgetItem(status))
            self.names.append(str(d['item_name']))
            self.sales.append(int(d['forecast']))
            rows = rows + 1


    def btn_operators_click(self):
        for i in reversed(range(self.verticalLayout_3.count())):
            self.verticalLayout_3.itemAt(i).widget().setParent(None)

        # self.verticalLayout_3.addWidget(self.operators_table())
        self.frame_6 = QtWidgets.QFrame(self.frame_2)
        self.frame_6.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.addOperator = QtWidgets.QPushButton(self.frame_6)
        self.addOperator.setText("Add Operator")
        self.addOperator.setObjectName("addOperator")
        self.deleteOperator = QtWidgets.QPushButton(self.frame_6)
        self.deleteOperator.setText("Delete operator")
        self.deleteOperator.setObjectName("deleteOperator")
        self.editOperator = QtWidgets.QPushButton(self.frame_6)
        self.editOperator.setText("Edit operator")
        self.editOperator.setObjectName("editOperator")
        self.horizontalLayout_2.addWidget(self.addOperator)
        self.horizontalLayout_2.addWidget(self.deleteOperator)
        self.horizontalLayout_2.addWidget(self.editOperator)
        spacerItem = QtWidgets.QSpacerItem(652, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout_3.addWidget(self.frame_6)

        ## Lower frame
        self.frame_7 = QtWidgets.QFrame(self.frame_2)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")

        # loads operators data within table
        self.verticalLayout_5.addWidget(self.operators_table())
        self.verticalLayout_6.addLayout(self.verticalLayout_5)
        self.verticalLayout_3.addWidget(self.frame_7)
        self.addOperator.clicked.connect(self.btn_open_add_operator_dialog)
        self.deleteOperator.clicked.connect(self.btn_open_delete_operator_dialog)
        self.editOperator.clicked.connect(self.btn_open_edit_operator_dialog)

    def btn_open_edit_operator_dialog(self):
        self.edit_operator_dialog = QtWidgets.QDialog()
        self.edit_operator_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setupEditOperatorUi(self.edit_operator_dialog)
        self.edit_operator_dialog.show()

    def setupEditOperatorUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.listWidget = QtWidgets.QListWidget(self.frame)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_2.addWidget(self.listWidget)
        self.editItem = QtWidgets.QPushButton(self.frame)
        self.editItem.setObjectName("editItem")
        self.verticalLayout_2.addWidget(self.editItem)
        self.verticalLayout.addWidget(self.frame)

        self.load_operators_list_dialog_data()

        self.retranslateEditOperatorUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.editItem.clicked.connect(self.edit_operator)

    def btn_open_delete_operator_dialog(self):
        self.del_operator_dialog = QtWidgets.QDialog()
        self.del_operator_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setupDeleteOperatorUi(self.del_operator_dialog)
        self.del_operator_dialog.show()

    def retranslateEditOperatorUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Select operator to edit"))
        self.editItem.setText(_translate("Dialog", "Edit Operator"))

    def edit_operator(self):
        self.operator_auth_id = self.listWidget.currentItem().text()
        self.save_operator_edit_dialog = QtWidgets.QDialog()
        self.save_operator_edit_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.EditOperatorDetailsDialogUi(self.save_operator_edit_dialog, self.operator_auth_id)
        self.save_operator_edit_dialog.show()

    def EditOperatorDetailsDialogUi(self, Dialog, operator_auth_id):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.widget = QtWidgets.QWidget(self.frame)
        self.widget.setGeometry(QtCore.QRect(53, 20, 301, 211))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.setEnabled(False)
        self.horizontalLayout_3.addWidget(self.lineEdit_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_4.addWidget(self.lineEdit_4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_5.setEchoMode(QtWidgets.QLineEdit.Password)
        self.horizontalLayout_5.addWidget(self.lineEdit_5)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addWidget(self.frame)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateDialogEditOperatorUi(Dialog, operator_auth_id)
        self.buttonBox.accepted.connect(self.dialog_operator_edit_accepted)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def dialog_operator_edit_accepted(self):
        self.first_name = str(self.lineEdit.text()).strip()
        self.last_name = str(self.lineEdit_2.text()).strip()
        self.auth_id = str(self.lineEdit_3.text()).strip()
        self.date_of_birth = str(self.lineEdit_4.text()).strip()
        self.password = str(self.lineEdit_5.text()).strip()

        if self.password != "":
            self.hashed_password = bcrypt.hashpw(self.password.encode('utf-8'), bcrypt.gensalt())
            self.save_operator_edit_details(self.first_name, self.last_name, self.date_of_birth,
                                            password=str(self.hashed_password, 'utf-8'))
        else:
            self.save_operator_edit_details(self.first_name, self.last_name, self.date_of_birth)

        self.save_operator_edit_dialog.hide()
        # Reloads the operators table list
        self.btn_operators_click()

    def retranslateDialogEditOperatorUi(self, Dialog, auth_id):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Add Operator"))
        self.label.setText(_translate("Dialog", "First name:"))
        self.label_2.setText(_translate("Dialog", "Last name:"))
        self.label_3.setText(_translate("Dialog", "Auth ID:"))
        self.label_4.setText(_translate("Dialog", "Date of birth:"))
        self.label_5.setText(_translate("Dialog", "New password:"))

        self.operator_edit_auth_id = auth_id
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute(
            'select id, first_name, last_name, auth_id, dob, created_at, updated_at from operators where auth_id=\'' + self.operator_edit_auth_id + '\'')
        data_list = cursor.fetchall()
        for row_number, d in enumerate(data_list):
            self.lineEdit.setText(_translate("Dialog", d[1]))
            self.lineEdit_2.setText(_translate("Dialog", d[2]))
            self.lineEdit_3.setText(_translate("Dialog", d[3]))
            self.lineEdit_4.setText(_translate("Dialog", str(d[4])))
        conn.close()

    def save_operator_edit_details(self, first_name, last_name, date_of_birth, password=''):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        query = ''
        if password != '':
            query = 'update operators set first_name=\'' + first_name + '\',last_name=\'' + last_name + '\',dob=\'' + date_of_birth + '\'' \
                                                                                                                                      ',password=\'' + password + '\' where auth_id=\'' + self.operator_edit_auth_id + '\''

        else:
            query = 'update operators set first_name=\'' + first_name + '\',last_name=\'' + last_name + '\',dob=\'' + date_of_birth + '\' ' \
                                                                                                                                      'where auth_id=\'' + self.operator_edit_auth_id + '\''

        print("Edit query: ", query)
        try:
            cursor.execute(query)
            conn.commit()
        except mysql.connector.Error as err:
            print("Something went wrong: {}".format(err))
            conn.rollback()

        conn.close()

    def setupDeleteOperatorUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.listWidget = QtWidgets.QListWidget(self.frame)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_2.addWidget(self.listWidget)
        self.deleteItem = QtWidgets.QPushButton(self.frame)
        self.deleteItem.setObjectName("deleteItem")
        self.verticalLayout_2.addWidget(self.deleteItem)
        self.verticalLayout.addWidget(self.frame)

        self.load_operators_list_dialog_data()

        self.retranslateDeleteOperatorUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.deleteItem.clicked.connect(self.delete_operator)

    def retranslateDeleteOperatorUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Select item to delete"))
        self.deleteItem.setText(_translate("Dialog", "Delete Item"))

    def load_operators_list_dialog_data(self):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        cursor.execute('select id, auth_id from operators')
        data_list = cursor.fetchall()
        # item = QtWidgets.QListWidgetItem()
        for row_number, d in enumerate(data_list):
            self.listWidget.addItem(QtWidgets.QListWidgetItem(d[1]))
        conn.close()

    def delete_operator(self):
        self.del_auth_id = self.listWidget.currentItem().text()
        self.delete_operator_from_db(self.del_auth_id)

    def delete_operator_from_db(self, auth_id):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        query = 'delete from operators where auth_id=\'' + auth_id + '\''
        print(query)
        try:
            cursor.execute(query)
            conn.commit()
            self.del_operator_dialog.close()
            self.btn_operators_click()
        except mysql.connector.Error as err:
            print("Something went wrong: {}".format(err))
            conn.rollback()

        conn.close()

    def btn_open_add_operator_dialog(self):
        # Dialog of modal type, not dismissible by click to other windows
        self.dialog = QtWidgets.QDialog()
        self.dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        # self.nd = Ui_Dialog()
        # self.nd.setupUi(self.dialog)
        self.setupDialogUi(self.dialog)
        self.dialog.show()

    def btn_open_add_item_dialog(self):
        # Dialog of modal type, not dismissible by click to other windows
        self.dialog = QtWidgets.QDialog()
        self.dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        # self.nd = Ui_Dialog()
        # self.nd.setupUi(self.dialog)
        self.addItemDialogUi(self.dialog)
        self.dialog.show()

    def btn_operator_performance_click(self):
        for i in reversed(range(self.verticalLayout_3.count())):
            self.verticalLayout_3.itemAt(i).widget().setParent(None)
        #self.operator_performance_view()
        self.newSalesView()


    ##New sales layout

    def newSalesView(self): #todo Modified 3
        MainWindow.setWindowTitle("Sales analysis")
        self.frame_6 = QtWidgets.QFrame(self.frame_2)
        self.frame_6.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.comboBox = QtWidgets.QComboBox(self.frame_6)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout_2.addWidget(self.comboBox)
        self.dateRange = QtWidgets.QLabel(self.frame_6)
        self.dateRange.setObjectName("dateRange")
        self.horizontalLayout_2.addWidget(self.dateRange)
        self.dateFrom = QtWidgets.QLineEdit(self.frame_6)
        self.dateFrom.setObjectName("dateFrom")
        self.horizontalLayout_2.addWidget(self.dateFrom)
        self.label_3 = QtWidgets.QLabel(self.frame_6)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.dateTo = QtWidgets.QLineEdit(self.frame_6)
        self.dateTo.setObjectName("dateTo")
        self.horizontalLayout_2.addWidget(self.dateTo)
        self.runFilter = QtWidgets.QPushButton(self.frame_6)
        self.runFilter.setObjectName("runFilter")
        self.horizontalLayout_2.addWidget(self.runFilter)
        spacerItem = QtWidgets.QSpacerItem(500, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout_3.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(self.frame_2)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_3.addWidget(self.frame_7)
        self.horizontalLayout.addWidget(self.frame_2)

        self.dateFrom.setPlaceholderText("e.g. 2018-5-1")
        self.dateTo.setPlaceholderText("e.g 2018-12-1")
        self.dateRange.setText("Date Range:")
        self.label_3.setText("To")
        self.runFilter.setText("Filter")

        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host,
                                       database=self.database)
        cursor = conn.cursor()
        cursor.execute('select id, product_name from items')
        data_list = cursor.fetchall()
        print(data_list)

        self.combo_items = {}
        self.comboBox.addItem("All")
        for row_number, d in enumerate(data_list):
            self.combo_items[str(d[1])] = str(d[0])
            self.comboBox.addItem(str(d[1]))

        self.runFilter.clicked.connect(self.filter_result)

        query = 'SELECT SUM(quantity) as total, DATE (created_at) FROM sales GROUP BY DATE (created_at);'
        self.sales_analytics_graph_view(query)

    def sales_analytics_graph_view(self, query): #todo Modified
        for i in reversed(range(self.verticalLayout.count())):
            self.verticalLayout.itemAt(i).widget().setParent(None)

        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()

        cursor.execute(query)

        data_list = cursor.fetchall()
        print(len(data_list))

        if len(data_list) > 0:
            date = []
            total_sales = []
            for row_number, d in enumerate(data_list):
                total_sales.append(int(d[0]))
                date.append(str(d[1]))
            conn.close()

            x = np.arange(len(data_list))
            money = total_sales
            plt.clf()
            self.fig4, ax4 = plt.subplots()
            plt.bar(x, money)
            plt.legend('Sales', loc='upper right')
            plt.xticks(x, tuple(date))
            plt.tick_params(axis='both', which='major', labelsize=6)
            plt.tick_params(axis='both', which='minor', labelsize=6)
            plt.xlabel('Date', fontsize=8)
            plt.ylabel('Total sales', fontsize=8)
            plt.title('Sales analysis', fontsize=12)
            self.canvas_operators = FigureCanvas(self.fig4)
            self.verticalLayout.addWidget(self.canvas_operators)
            self.canvas_operators.draw()
        else:
            notification('No data available') #todo Count check, effect this asap

    def filter_result(self): #todo modied 2
        combo_item = ''
        comboItemName = str(self.comboBox.currentText())
        if comboItemName != 'All':
            combo_item = self.combo_items[comboItemName]
        date_from = str(self.dateFrom.text()).strip()
        date_to = str(self.dateTo.text()).strip()

        if combo_item or date_from or date_to:
            dic = {}

            if combo_item:
                dic["item"] = 'item_id = \'' + combo_item.strip() + '\''

            if date_from:
                dic["date_from"] = 'DATE(created_at) >=\'' + date_from.strip() + '\''

            if date_to:
                dic["date_to"] = 'DATE(created_at) <=\'' + date_to.strip() + '\''


            like_list = []
            for key in dic:
                like_list.append(dic[key])

            like_query = ' AND '.join(like_list)

            print("Query ", like_query)
            query = 'SELECT SUM(quantity) as total, DATE (created_at) FROM sales WHERE '+like_query+' GROUP BY DATE (created_at);'

        else:
            query = 'SELECT SUM(quantity) as total, DATE (created_at) FROM sales GROUP BY DATE (created_at);'

        self.sales_analytics_graph_view(query)

    ##End of new sales layout

    '''
        Ends button events handlers
    '''

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Bortec Inventory Analytics System"))
        self.label.setText(_translate("MainWindow", "<h2><i><b>BORTEC</b></i></h2>"))
        self.pushButton.setText(_translate("MainWindow", "Items"))
        self.pushButton_2.setText(_translate("MainWindow", "Sales"))
        self.pushButton_3.setText(_translate("MainWindow", "Received"))
        self.pushButton_4.setText(_translate("MainWindow", "Inventory"))
        self.pushButton_operators.setText(_translate("MainWindow", "Operators"))
        self.pushButton_5.setText(_translate("MainWindow", "Real-time Analytics"))
        self.pushButton_6.setText(_translate("MainWindow", "Static Analytics"))
        self.pushButton_7.setText(_translate("MainWindow", "Predictive Analysis"))
        self.pushButton_8.setText(_translate("MainWindow", "Sales Analytics"))
        self.operator_analysis_btn.setText(_translate("MainWindow", "Operator analysis"))
        # self.pushButton_9.setText(_translate("MainWindow", "Admins"))
        self.pushButton_10.setText(_translate("MainWindow", "Logout"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))

    # class Ui_Dialog(object):
    def setupDialogUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.widget = QtWidgets.QWidget(self.frame)
        self.widget.setGeometry(QtCore.QRect(53, 20, 301, 211))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_3.addWidget(self.lineEdit_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_4.addWidget(self.lineEdit_4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_5.setEchoMode(QtWidgets.QLineEdit.Password)
        self.horizontalLayout_5.addWidget(self.lineEdit_5)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addWidget(self.frame)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateDialogUii(Dialog)
        self.buttonBox.accepted.connect(self.dialog_accepted)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def dialog_accepted(self):
        self.first_name = str(self.lineEdit.text()).strip()
        self.last_name = str(self.lineEdit_2.text()).strip()
        self.auth_id = str(self.lineEdit_3.text()).strip()
        self.date_of_birth = str(self.lineEdit_4.text()).strip()
        self.password = str(self.lineEdit_5.text()).strip()

        self.hashed_password = bcrypt.hashpw(self.password.encode('utf-8'), bcrypt.gensalt())

        self.save_operator_details(self.first_name, self.last_name,
                                   self.auth_id, self.date_of_birth, str(self.hashed_password, 'utf-8'))
        self.dialog.hide()
        # Reloads the operators table list
        self.btn_operators_click()

    def retranslateDialogUii(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Add Operator"))
        self.label.setText(_translate("Dialog", "First name:"))
        self.label_2.setText(_translate("Dialog", "Last name:"))
        self.label_3.setText(_translate("Dialog", "Auth ID:"))
        self.label_4.setText(_translate("Dialog", "Date of brith:"))
        self.label_5.setText(_translate("Dialog", "Password:"))

    def save_operator_details(self, first_name, last_name, auth_id, date_of_birth, password):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        try:
            cursor.execute('insert into operators(first_name,last_name,auth_id,dob,password) '
                           'values(\'' + first_name + '\',\'' + last_name + '\',\'' + auth_id + '\',\'' + date_of_birth + '\''
                                                                                                                          ',\'' + password + '\')')
            conn.commit()
        except mysql.connector.Error as err:
            print("Something went wrong: {}".format(err))
            conn.rollback()

        conn.close()

    def addItemDialogUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.itemName = QtWidgets.QLineEdit(self.frame)
        self.itemName.setObjectName("itemName")
        self.horizontalLayout.addWidget(self.itemName)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.UnitOfMeasurement = QtWidgets.QLineEdit(self.frame)
        self.UnitOfMeasurement.setObjectName("UnitOfMeasurement")
        self.horizontalLayout_2.addWidget(self.UnitOfMeasurement)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.unitPrice = QtWidgets.QLineEdit(self.frame)
        self.unitPrice.setText("")
        self.unitPrice.setObjectName("unitPrice")
        self.horizontalLayout_3.addWidget(self.unitPrice)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        self.unitCost = QtWidgets.QLineEdit(self.frame)
        self.unitCost.setText("")
        self.unitCost.setObjectName("unitCost")
        self.horizontalLayout_6.addWidget(self.unitCost)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.codes = QtWidgets.QLineEdit(self.frame)
        self.codes.setText("")
        self.codes.setObjectName("codes")
        self.horizontalLayout_4.addWidget(self.codes)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.remarks = QtWidgets.QTextEdit(self.frame)
        self.remarks.setObjectName("remarks")
        self.horizontalLayout_5.addWidget(self.remarks)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addWidget(self.frame)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateAddItemUi(Dialog)
        self.buttonBox.accepted.connect(self.dialog_accepted_add_property)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateAddItemUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Add Item"))
        self.label.setText(_translate("Dialog", "Item name:"))
        self.label_2.setText(_translate("Dialog", "Units of measurement:"))
        self.label_3.setText(_translate("Dialog", "Unit price: "))
        self.label_6.setText(_translate("Dialog", "Unit cost: "))
        self.label_4.setText(_translate("Dialog", "Codes: "))
        self.label_5.setText(_translate("Dialog", "Remarks: "))

    def dialog_accepted_add_property(self):
        self.item_name = str(self.itemName.text()).strip()
        self.units_of_measurement = str(self.UnitOfMeasurement.text()).strip()
        self.unit_price = str(self.unitPrice.text()).strip()
        self.unit_cost = str(self.unitCost.text()).strip()
        self.code = str(self.codes.text()).strip()
        self.remark = str(self.remarks.toPlainText()).strip()

        self.save_item_details(self.item_name, self.units_of_measurement, self.unit_cost, self.unit_price, self.code, self.remark)
        self.dialog.hide()
        # Reloads the items table list
        self.btn_items_click()

    def save_item_details(self, item_name, units_of_measurement, unit_cost, unit_price, code, remarks):
        conn = mysql.connector.connect(user=self.dbuser, password=self.dbpassword, host=self.host, database=self.database)
        cursor = conn.cursor()
        query = 'insert into items(codes,product_name,units,unit_cost, unit_price,remarks) ' \
                'values(\'' + code + '\',\'' + item_name + '\',\'' + units_of_measurement + '\',\'' + unit_cost + '\',\'' + unit_price + '\'' \
                                                                                                                   ',\'' + remarks + '\')'
        print(query)
        try:
            cursor.execute(query)
            conn.commit()
            item_id = cursor.lastrowid

            # Inserts inventory data
            query1 = 'insert into inventory_stocks(item_id,received,sales,stocks,total_expenditure_cost,total_sales_cost) ' \
                    'values(\'' + str(item_id) + '\',\'0\',\'0\',\'0\',\'0\',\'0\')'
            print(query1)
            cursor.execute(query1)
            conn.commit()

        except mysql.connector.Error as err:
            print("Something went wrong: {}".format(err))
            conn.rollback()

        conn.close()

    def show_login_dialog(self):
        self.myapp = GUIForm()
        self.myapp.show()

    def __init__(self):
        # Dialog of modal type, not dismissible by click to other windows
        self.show_login_dialog()
        self.host = HOST
        self.dbuser = DB_USER
        self.dbpassword = DB_PASSWORD
        self.database = DB


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
