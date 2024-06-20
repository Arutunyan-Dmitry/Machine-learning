import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import make_moons

X, y = make_moons(noise=0.3, random_state=None)                                                  # Генерация данных с пересечениями признаков
X = StandardScaler().fit_transform(X)                                                            # Стандартизация. Удаление средних, увеличение дисперсии до 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)  # Разделение данных на обучающую и тестовую выборки


# Модель линейной регрессии
def lr_prediction():
    linear = LinearRegression()         # Создание модели
    linear.fit(X_train, y_train)        # Обучение модели
    y_predict = linear.predict(X_test)  # Решение задачи предсказания

    mid_square = np.round(np.sqrt(metrics.mean_squared_error(y_test, y_predict)), 3)  # Рассчёт среднеквадратичной ошибки модели
    det_kp = np.round(metrics.r2_score(y_test, y_predict), 2)                         # Рассчёт коэфициента детерминации модели

    return "Модель линейной регрессии", y_predict, mid_square, det_kp


# Модель полиномиальной регрессии
def poly_lr_prediction():
    poly = PolynomialFeatures(degree=4, include_bias=False)   # Создание характеристик полиномиальной модели (степень - 4, обнуление свободного члена - нет)
    x_poly_train = poly.fit_transform(X_train)                # Трансформация выборки обучения (добавление недостающих аргументов многочлена 4го порядка)
    x_poly_test = poly.fit_transform(X_test)                  # Трансформация тестовой выборки (добавление недостающих аргументов многочлена 4го порядка)
    linear = LinearRegression()                               # Создание модели
    linear.fit(x_poly_train, y_train)                         # Обучение модели
    y_predict = linear.predict(x_poly_test)                   # Решение задачи предсказания

    mid_square = np.round(np.sqrt(metrics.mean_squared_error(y_test, y_predict)), 3)  # Рассчёт среднеквадратичной ошибки модели
    det_kp = np.round(metrics.r2_score(y_test, y_predict), 2)                         # Рассчёт коэфициента детерминации модели

    return "Модель полиномиальной регрессии", y_predict, mid_square, det_kp


# Модель полиномиальной гребневой регрессии
def poly_rg_prediction():
    poly = PolynomialFeatures(degree=4, include_bias=False)   # Создание характеристик полиномиальной модели (степень - 4, обнуление свободного члена - нет)
    x_poly_train = poly.fit_transform(X_train)                # Трансформация выборки обучения (добавление недостающих аргументов многочлена 4го порядка)
    x_poly_test = poly.fit_transform(X_test)                  # Трансформация тестовой выборки (добавление недостающих аргументов многочлена 4го порядка)
    ridge = Ridge(alpha=1.0)                                  # Создание гребневой модели (уср. коэф - 1.0)
    ridge.fit(x_poly_train, y_train)                          # Обучение модели
    y_predict = ridge.predict(x_poly_test)                    # Решение задачи предсказания

    mid_square = np.round(np.sqrt(metrics.mean_squared_error(y_test, y_predict)), 3)  # Рассчёт среднеквадратичной ошибки модели
    det_kp = np.round(metrics.r2_score(y_test, y_predict), 2)                         # Рассчёт коэфициента детерминации модели

    return "Модель полиномиальной регрессии", y_predict, mid_square, det_kp


# Создание графиков поотдельности (для себя)
def make_plots(models):
    i = 0
    for model in models:
        plt.plot(y_test, c="red", label="\"y\" исходная")                          # Создание графика исходной функции
        plt.plot(model[1], c="green", label="\"y\" предсказанная \n"
                                                  "Ср^2 = " + str(model[2]) + "\n"
                                                  "Кд = " + str(model[3]))               # Создание графика предсказанной функции
        plt.legend(loc='lower left')
        plt.title(model[0])
        plt.savefig('static/' + str(i + 1) + '.png')
        plt.close()
        i += 1


if __name__ == '__main__':
    models = lr_prediction(), poly_lr_prediction(), poly_rg_prediction()
    make_plots(models)

    fig, axs = plt.subplots(3, 1, layout='constrained')              # Создание общего графика для сравнения моделей
    i = 0
    for model in models:
        fig.set_figwidth(6)
        fig.set_figheight(10)
        axs[i].set_title(model[0])
        axs[i].plot(y_test, c="red", label="\"y\" исходная")
        axs[i].plot(model[1], c="green", label="\"y\" предсказанная \n"
                                               "Ср^2 = " + str(model[2]) + "\n"
                                               "Кд = " + str(model[3]))
        axs[i].legend(loc='lower left')
        i += 1
    plt.savefig('static/result.png')


