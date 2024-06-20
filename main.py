import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

filein = "P:\\ULSTU\\ИИС\\Datasets\\heart_2020_norm.csv"


# Метод обучения нейронной сети
def reg_neural_net():
    df = pd.read_csv(filein, sep=',')
    x, y = [df.drop("HeartDisease", axis=1).values,
            df["HeartDisease"].values]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.001, random_state=42)

    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', solver='adam', random_state=15000)
    mlp.fit(x_train, y_train)
    y_predict = mlp.predict(x_test)
    err = pred_errors(y_predict, y_test)
    make_plots(y_test, y_predict, err[0], err[1], "Нейронная сеть")


# Метод рассчёта ошибок
def pred_errors(y_predict, y_test):
    mid_square = np.round(np.sqrt(metrics.mean_squared_error(y_test, y_predict)),3)            # Рассчёт среднеквадратичной ошибки модели
    det_kp = np.round(metrics.r2_score(y_test, y_predict), 2)                                  # Рассчёт коэфициента детерминации модели
    return mid_square, det_kp


# Метод отрисовки графиков
def make_plots(y_test, y_predict, mid_sqrt, det_kp, title):
        plt.plot(y_test, c="red", label="\"y\" исходная")                                # Создание графика исходной функции
        plt.plot(y_predict, c="green", label="\"y\" предсказанная \n"
                                                  "Ср^2 = " + str(mid_sqrt) + "\n"
                                                  "Кд = " + str(det_kp))                       # Создание графика предсказанной функции
        plt.legend(loc='lower left')
        plt.title(title)
        plt.savefig('static/' + title + '.png')
        plt.close()


if __name__ == '__main__':
    reg_neural_net()
