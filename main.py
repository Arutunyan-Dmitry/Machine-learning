from operator import itemgetter
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler


np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))                       # Генерируем исходные данные: 750 строк-наблюдений и 14 столбцов-признаков

Y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 +
     10 * X[:, 3] + 5 * X[:, 4] ** 5 + np.random.normal(0, 1))     # Задаем функцию-выход: регрессионную проблему Фридмана

X[:, 10:] = X[:, :4] + np.random.normal(0, .025, (size, 4))   # Добавляем зависимость признаков

ridge = Ridge(alpha=1)            # Создаём модель гребневой регрессии и обучаем её
ridge.fit(X, Y)

lr = LinearRegression()           # Создаём модель линейной регрессии и обучаем её
lr.fit(X, Y)
rfe = RFE(lr)                     # На основе линейной модели выполняем рекурсивное сокращение признаков
rfe.fit(X,Y)

rfr = RandomForestRegressor()     # Создаём и обучаем регрессор случайного леса (используется вместо устаревшего рандомизированного лассо)
rfr.fit(X, Y)


def rank_ridge_rfr_to_dict(ranks, names):                                 # Метод нормализации оценок важности для модели гребневой регрессии и регрессора случайного леса
    ranks = np.abs(ranks)
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(np.array(ranks).reshape(14, 1)).ravel()
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


def rank_rfe_to_dict(ranks, names):                                       # Метод нормализации оценок важности для модели рекурсивного сокращения признаков
    new_ranks = [float(1 / x) for x in ranks]
    new_ranks = map(lambda x: round(x, 2), new_ranks)
    return dict(zip(names, new_ranks))


if __name__ == '__main__':
    names = ["x%s" % i for i in range(1, 15)]
    ranks = dict()

    ranks["Ridge"] = rank_ridge_rfr_to_dict(ridge.coef_, names)
    ranks["Recursive Feature Elimination"] = rank_rfe_to_dict(rfe.ranking_, names)
    ranks["Random Forest Regression"] = rank_ridge_rfr_to_dict(rfr.feature_importances_, names)

    for key, value in ranks.items():                                          # Вывод нормализованных оценок важности признаков каждой модели
        ranks[key] = sorted(value.items(), key=itemgetter(1), reverse=True)
    for key, value in ranks.items():
        print(key)
        print(value)

    mean = {}                                                                 # Нахождение средних значений оценок важности по 3м моделям
    for key, value in ranks.items():
        for item in value:
            if item[0] not in mean:
                mean[item[0]] = 0
            mean[item[0]] += item[1]
    for key, value in mean.items():
        res = value / len(ranks)
        mean[key] = round(res, 2)
    mean = sorted(mean.items(), key=itemgetter(1), reverse=True)
    print("Mean")
    print(mean)
