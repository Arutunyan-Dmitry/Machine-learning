import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

filein = "P:\\ULSTU\\ИИС\\Datasets\\heart_2020_norm.csv"
fileout = "P:\\ULSTU\\ИИС\\Datasets\\heart_2020_classified.csv"


# Метод устранения шумов и кластеризации данных алгоритмом DBSCAN
def dbscan():
    df = pd.read_csv(filein, sep=',').iloc[0:10000]                  # Считывание датасета
    x = df.drop("HeartDisease", axis=1)                              # Определение кластеризуемых параметров

    eps_opt = (x.max().values.mean() + x.min().values.mean()) / 2    # Рассчёт опционального радиуса окрестности методом средней плотности

    developed_data = []                                              # Подбор значения минимального количества точек в окрестности
    for i in range(len(x)):                                          # - Начинаем с одной точки
        if i == 0:
            continue                                                 # - Увеличиваем значение кол-ва точек на 1
        dbscan = DBSCAN(eps=eps_opt, min_samples=i)                  # - Обучаем модель и получаем массив кластеров
        clusters = dbscan.fit_predict(x.values)
        if len(set(clusters)) <= 7:                                  # - Прекращаем увеличивать значение точек, если кол-во кластеров уменьшилось до требуемого
            developed_data = clusters
            break
        if list(clusters).count(-1) / len(clusters) >= 0.1:          # - Или если "шум" превышает 10% от данных
            developed_data = clusters
            break

    make_plot(x, developed_data)
    df["DBSCAN"] = developed_data
    df.to_csv(fileout, index=False)                                  # Сохраняем полученные кластеры как доп. столбец датасета


# Метод оценки эффективности кластеризации DBSCAN
def linear_reg():                                           # Создаём две выборки данных
    df = pd.read_csv(fileout, sep=',')                      # В 1й избавляемся от "шумов" и используем столбец кластеров как признак
    df_mod = df.loc[df["DBSCAN"] != -1]
    x_train_mod = df_mod.drop("HeartDisease", axis=1).iloc[0:round(len(df) / 100 * 99)]
    y_train_mod = df_mod["HeartDisease"].iloc[0:round(len(df) / 100 * 99)]
    x_test_mod = df_mod.drop("HeartDisease", axis=1).iloc[round(len(df) / 100 * 99):len(df)]
    y_test_mod = df_mod["HeartDisease"].iloc[round(len(df) / 100 * 99):len(df)]
                                                            # Во 2й оставляем обычные данные
    x_train = df.drop(["HeartDisease", "DBSCAN"], axis=1).iloc[0:round(len(df) / 100 * 99)]
    y_train = df["HeartDisease"].iloc[0:round(len(df) / 100 * 99)]
    x_test = df.drop(["HeartDisease", "DBSCAN"], axis=1).iloc[round(len(df) / 100 * 99):len(df)]
    y_test = df["HeartDisease"].iloc[round(len(df) / 100 * 99):len(df)]

    lr_mod = LinearRegression()                              # Обучаем модель без "шума" и с признаком кластеров
    lr_mod.fit(x_train_mod.values, y_train_mod.values)
    y_mod_pred = lr_mod.predict(x_test_mod.values)
    err = pred_errors(y_mod_pred, y_test_mod.values)
    make_plots(y_test_mod.values, y_mod_pred, err[0], err[1], "Регрессия с кластеризацией dbscan")

    lr = LinearRegression()                                  # Обучаем модель на исходных данных
    lr.fit(x_train.values, y_train.values)
    y_pred = lr.predict(x_test.values)
    err = pred_errors(y_pred, y_test.values)
    make_plots(y_test.values, y_pred, err[0], err[1], "Чистая линейная регрессия")


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


# Метод построения графика кластеризации
def make_plot(x, c):
    plt.scatter(x.values[:, 0], x.values[:, 13], c=c, cmap='viridis')
    plt.xlabel('BMI')
    plt.ylabel('SleepTime')
    plt.colorbar()
    plt.title('DBSCAN Clustering')
    plt.savefig('static/dbscan.png')
    plt.close()


if __name__ == '__main__':
    dbscan()
    linear_reg()
