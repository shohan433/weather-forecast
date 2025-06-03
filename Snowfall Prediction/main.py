import pyttsx3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def random_data_CSV():
    np.random.seed(42)
    num_samples = 100
    temperature = np.random.randint(-10, 10, num_samples)
    humidity = np.random.randint(30, 90, num_samples)
    snowfall = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # Simulate rainy days (1) and non-rainy days (0)

    data = pd.DataFrame({'temperature': temperature, 'humidity': humidity, 'snowfall': snowfall})

    data.to_csv('snowfall_random_data.csv', index=False)


def Prediction():
    data = pd.read_csv('snowfall_random_data.csv')
    X = data[['temperature', 'humidity']]
    y = data['snowfall']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    model.predict(X_test)

    temp = int(input("Enter the value of temperature in Celsius = "))
    humid = int(input("Enter the value of humidity = "))

    new_data = pd.DataFrame({'temperature': [temp], 'humidity': [humid]})
    predicted_snowfall = model.predict(new_data)

    print(
        f'Predicted snowfall for temperature = {new_data["temperature"].iloc[0]} and '
        f'humidity = {new_data["humidity"].iloc[0]}'
        f'?'f'{" Yes" if predicted_snowfall[0] else " No"}')


def repeat():
    while True:
        print("Welcome to the Simple Snowfall Prediction Software")
        print("Developed by Mohammad BarkatUllah Chowdhury\n")
        print("** Important Note ** \nFor making this software, Mohammad BarkatUllah Chowdhury has used "
              "random generated data for temperature and humidity to train the model.")

        tk = pyttsx3.init()
        tk.say("Welcome to the Simple Snowfall Prediction Software")
        tk.say("Developed by Mohammad BarkatUllah Chowdhury")
        tk.say("Important Note ** \nFor making this software, Mohammad BarkatUllah Chowdhury has used random generated data for "
               "temperature and humidity to train the model")
        tk.runAndWait()

        Prediction()

        user = input("Do you want to use it again? Press Y/N: ").lower()
        print("\n")

        if user == 'n':
            print("Thanks for using. Have a lovely day :)")
            break

        elif user != 'y':
            print("Invalid input. Shutting down the software. Thanks for using. Have a lovely day :)")
            break


repeat()



