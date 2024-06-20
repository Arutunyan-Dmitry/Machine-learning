import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import Ridge

filein = "P:\\ULSTU\\ИИС\\Datasets\\heart_2020_norm.csv"


# Метод решения задачи предсказания на всех признаках данных
def ridge_all():
    df = pd.read_csv(filein, sep=',')

    x_train = df.drop("HeartDisease", axis=1).iloc[0:round(len(df) / 100 * 99)]
    y_train = df["HeartDisease"].iloc[0:round(len(df) / 100 * 99)]
    x_test = df.drop("HeartDisease", axis=1).iloc[round(len(df) / 100 * 99):len(df)]
    y_test = df["HeartDisease"].iloc[round(len(df) / 100 * 99):len(df)]

    rid = Ridge(alpha=1.0)
    rid.fit(x_train.values, y_train.values)
    y_predict = rid.predict(x_test.values)
    err = pred_errors(y_predict, y_test.values)
    make_plots(y_test.values, y_predict, err[0], err[1], "Гребневая регрессия (все признаки)")


# Метод решения задачи предсказания на значимых признаках данных
def ridge_valuable():
    df = pd.read_csv(filein, sep=',')

    x_train = df[["BMI", "PhysicalHealth", "MentalHealth", "AgeCategory", "Race",
                  "PhysicalActivity", "GenHealth", "SleepTime", ]].iloc[0:round(len(df) / 100 * 99)]
    y_train = df["HeartDisease"].iloc[0:round(len(df) / 100 * 99)]
    x_test = df[["BMI", "PhysicalHealth", "MentalHealth", "AgeCategory", "Race",
                 "PhysicalActivity", "GenHealth", "SleepTime", ]].iloc[round(len(df) / 100 * 99):len(df)]
    y_test = df["HeartDisease"].iloc[round(len(df) / 100 * 99):len(df)]

    rid = Ridge(alpha=1.0)
    rid.fit(x_train.values, y_train.values)
    y_predict = rid.predict(x_test.values)
    err = pred_errors(y_predict, y_test.values)
    make_plots(y_test.values, y_predict, err[0], err[1], "Гребневая регрессия (значимые признаки)")


# Метод рассчёта ошибок
def pred_errors(y_predict, y_test):
    mid_square = np.round(np.sqrt(metrics.mean_squared_error(y_test, y_predict)),3)            # Рассчёт среднеквадратичной ошибки модели
    det_kp = np.round(metrics.r2_score (y_test, y_predict), 2)                                 # Рассчёт коэфициента детерминации модели
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
    ridge_all()
    ridge_valuable()
