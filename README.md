
## Лабораторная работа 2. Вариант 4.
### Задание 
Выполнить ранжирование признаков. Отобразить получившиеся значения\оценки каждого признака каждым методом\моделью и среднюю оценку. Провести анализ получившихся результатов. Какие четыре признака оказались самыми важными по среднему значению? 

Модели:

- Гребневая регрессия `Ridge`, 
- Случайное Лассо `RandomizedLasso`, 
- Рекурсивное сокращение признаков `Recursive Feature Elimination – RFE`

> **Warning**
>
> Модель "случайное лассо" `RandomizedLasso` была признана устаревшей в бибилотеке `scikit` версии 0.20. Её безболезненной заменой назван регрессор случайного леса `RandomForestRegressor`. Он будет использоваться в данной лабораторной вместо устаревшей функции.

### Как запустить
Для запуска программы необходимо с помощью командной строки в корневой директории файлов прокета прописать:
```
python main.py
```

### Используемые технологии
- Библиотека `numpy`, используемая для обработки массивов данных и вычислений
- Библиотека `sklearn` - большой набор функционала для анализа данных. Из неё были использованы инструменты:
    - `LinearRegression` - инструмент работы с моделью "Линейная регрессия"
    - `Ridge` - инструмент работы с моделью "Гребневая регрессия"
    - `RFE` - инструмент оценки важности признаков "Рекурсивное сокращение признаков"
    - `RandomForestRegressor` - инструмент работы с моделью "Регрессор случайного леса"
    - `MinMaxScaler` - инструмент масштабирования значений в заданный диапазон

### Описание работы
Программа генерирует данные для обучения моделей. Сначала генерируются признаки в количестве 14-ти штук, важность которых модели предстоит выявить.
```python
np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))
```
Затем задаётся функция зависимости выходных параметров от входных, представляющая собой регриссионную проблему Фридмана.
```python
Y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 +
     10 * X[:, 3] + 5 * X[:, 4] ** 5 + np.random.normal(0, 1))
``` 
После чего, задаются зависимости переменных `x11 x12 x13 x14` от переменных `x1 x2 x3 x4`.
```python
X[:, 10:] = X[:, :4] + np.random.normal(0, .025, (size, 4))
``` 
Первая группа переменных должна быть обозначена моделями как наименее значимая.

#### Работа с моделями
Первая модель `Ridge` - модель гребневой регрессии. 
```python
ridge = Ridge(alpha=1)
ridge.fit(X, Y)
```
Данная модель не предоставляет прямого способа оценки важности признаков, так как она использует линейную комбинацию всех признаков с коэффициентами, которые оптимизируются во время обучения модели. Можно лишь оценить относительную важность признаков на основе абсолютных значений коэффициентов, которые были найдены в процессе обучения. Получить данные коэфициенты от модели можно с помощью метода `.coef_`.

Вторая модель `RandomForestRegressor` - алгоритм ансамбля случайных деревьев решений. Он строит множество деревьев, каждое из которых обучается на случайной подвыборке данных и случайном подмножестве признаков.
```python
rfr = RandomForestRegressor()
rfr.fit(X, Y)
```
Важность признаков в Random Forest Regressor определяется на основе того, как сильно каждый признак влияет на уменьшение неопределенности в предсказаниях модели. Для получения оценок важности в данной модели используется функция `.feature_importances_`.

Третий инструмент `Recursive Feature Elimination – RFE` - алгоритм отбора признаков, который используется для оценки и ранжирования признаков по их важности.
```python
lr = LinearRegression()
lr.fit(X, Y)
rfe = RFE(lr)
rfe.fit(X,Y)
```
Оценка важности признаков в RFE происходит путем анализа, как изменяется производительность модели при удалении каждого признака. В зависимости от этого, каждый признак получает ранг. Массив рангов признаков извлекается функцией `.ranking_`

#### Нормализация оценок
Модели `Ridge` и `RandomForestRegressor` рабботают по одинаковой логике вывода значимости оценок. В данных моделях оценки значимости параметров - веса значимости, которые они представляют для модели. Очевидно, что чем выше данный показатеь, тем более значимым является признак. Для нормализации оценок необходимо взять их по модулю и привести их к диапазону от 0 до 1.
```python
ranks = np.abs(ranks)
minmax = MinMaxScaler()
ranks = minmax.fit_transform(np.array(ranks).reshape(14, 1)).ravel()
ranks = map(lambda x: round(x, 2), ranks)
```
Инструмент `Recursive Feature Elimination – RFE` работает иначе. Класс выдает не веса при коэффициентах регрессии, а именно ранг для каждого признака. Так наиболее важные признаки будут иметь ранг – "1", а менее важные признаки ранг больше "1". Коэффициенты остальных моделей тем важнее, чем больше их абсолютное значение. Для нормализации таких рангов от 0 до 1, необходимо просто взять обратное число от величины ранга признака.
```python
new_ranks = [float(1 / x) for x in ranks]
new_ranks = map(lambda x: round(x, 2), new_ranks)
```

#### Оценка работы моделей
Для оценки результатов выведем выявленные оценки значимости признаков каждой модели, а также средние оценки значимости признаков всех моделей.
```
Ridge
[('x4', 1.0), ('x1', 0.98), ('x2', 0.8), ('x14', 0.61), ('x5', 0.54), ('x12', 0.39), ('x3', 0.25), ('x13', 0.19), ('x11', 0.16), ('x6', 0.08), ('x8', 0.07), ('x7', 0.02), ('x10', 0.02), ('x9', 0.0)]
Recursive Feature Elimination
[('x1', 1.0), ('x2', 1.0), ('x3', 1.0), ('x4', 1.0), ('x5', 1.0), ('x11', 1.0), ('x13', 1.0), ('x12', 0.5), ('x14', 0.33), ('x8', 0.25), ('x6', 0.2), ('x10', 0.17), ('x7', 0.14), ('x9', 0.12)]
Random Forest Regression
[('x14', 1.0), ('x2', 0.84), ('x4', 0.77), ('x1', 0.74), ('x11', 0.36), ('x12', 0.35), ('x5', 0.28), ('x3', 0.12), ('x13', 0.12), ('x6', 0.01), ('x7', 0.01), ('x8', 0.01), ('x9', 0.01), ('x10', 0.0)]
Mean
[('x4', 0.92), ('x1', 0.91), ('x2', 0.88), ('x14', 0.65), ('x5', 0.61), ('x11', 0.51), ('x3', 0.46), ('x13', 0.44), ('x12', 0.41), ('x8', 0.11), ('x6', 0.1), ('x7', 0.06), ('x10', 0.06), ('x9', 0.04)]

```
- Модель `Ridge` верно выявила значимость признаков `x1, x2, x4, х5`, но потеряла значимый признак `x3` и ошибочно включила признак `x14` в значимые.
- Модель `RandomForestRegressor` также верно выявила значимость признаков `x1, x2, x4`, но потеряла значимые признаки `x3, х5` и ошибочно включила признак `x14` в значимые.
- Инсрумент `Recursive Feature Elimination – RFE` безошибочно выделил все значимые признаки `x1, x2, х3, x4, x5`, но ошибочно отметил признаки `x11, x13` как значимые.
- В среднем значимыми признаками были верно выявлены `x1, x2, x4, х5`, но значимый признак `x3` был потерян, а признаки `x11, х14` были признаны ошибочно значимыми.


### Вывод
Хужё всех показала себя модель `RandomForestRegressor`, потеряв два значимых признака и добавив один лишний. Модель `Ridge`и инструмент `Recursive Feature Elimination – RFE` допустили по одной ошибке, однако последний не потерял ни одного значимого признака. Значимость в среднем получилась неудовлетворительна и выдала три ошибки, как и первая модель. 

Исходя из этого, можно сделать вывод, что для ранжирования признаков лучше использовать специально созданные для этого инструменты по типу `Recursive Feature Elimination – RFE`, а не использовать коэфициенты признаков регрессионных моделей.