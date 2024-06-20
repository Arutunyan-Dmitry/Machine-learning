
## Лабораторная работа 6. Вариант 4.
### Задание 
Использовать нейронную сеть `MLPRegressor` для данных из курсовой работы. Самостоятельно сформулировав задачу. Интерпретировать результаты и оценить, насколько хорошо он подходит для решения сформулированной задачи.

### Как запустить
Для запуска программы необходимо с помощью командной строки в корневой директории файлов прокета прописать:
```
python main.py
```
После этого в папке `static` сгенерируются график, по которому оценивается результат выполнения программы.

### Используемые технологии
- Библиотека `numpy`, используемая для обработки массивов данных и вычислений
- Библиотека `pyplot`, используемая для построения графиков.
- Библиотека `pandas`, используемая для работы с данными для анализа scv формата.
- Библиотека `sklearn` - большой набор функционала для анализа данных. Из неё были использованы инструменты:
    - `train_test_split` - разделитель данных на обучающиую и тестовую выборки
    - `metrics` - набор инструменов для оценки моделей
    - `MLPRegressor` - инструмент работы с моделью "Многослойный перцептрон для задачи регрессии"

`MLPRegressor` - это тип искусственной нейронной сети, состоящей из нескольких слоев нейронов, включая входной слой, скрытые слои и выходной слой.
Этот класс позволяет создавать и обучать MLP-модель для предсказания непрерывных числовых значений.

### Описание работы
#### Описание набора данных
Набор данных - набор для определения возможности наличия ССЗ заболеваний у челоека

Названия столбцов набора данных и их описание:

 * HeartDisease - Имеет ли человек ССЗ (No / Yes),
 * BMI - Индекс массы тела человека (float),
 * Smoking - Выкурил ли человек хотя бы 5 пачек сигарет за всю жизнь (No / Yes),
 * AlcoholDrinking - Сильно ли человек употребляет алкоголь (No / Yes),
 * Stroke - Был ли у человека инсульт (No / Yes),
 * PhysicalHealth - Сколько дней за последний месяц человек чувствовал себя плохо (0-30),
 * MentalHealth - Сколько дней за последний месяц человек чувствовал себя удручённо (0-30),
 * DiffWalking - Ииспытывает ли человек трудности при ходьбе (No / Yes),
 * Sex - Пол (female, male),
 * AgeCategory - Возрастная категория (18-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80 or older),
 * Race - Национальная принадлежность человека (White, Black, Hispanic, American Indian/Alaskan Native, Asian, Other),
 * Diabetic - Был ли у человека диабет (No / Yes),
 * PhysicalActivity - Занимался ли человек спротом за последний месяц (No / Yes),
 * GenHealth - Общее самочувствие человека (Excellent, Very good, Good, Fair, Poor),
 * SleepTime - Сколько человек в среднем спит за 24 часа (0-24),
 * Asthma - Была ли у человека астма (No / Yes),
 * KidneyDisease - Было ли у человека заболевание почек (No / Yes),
 * SkinCancer - Был ли у человека рак кожи (No / Yes).

Ссылка на страницу набора на kuggle: [Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)

#### Формулировка задачи
Поскольку модель `MLPRegressor` используется для решения задачи регресси, то попробуем на ней предсказать поведение параметров при обучении на всех признаках, варьируя конфигурации модели. Сформулируем задачу:
> "Решить задачу предсказания с помощью нейронной сети, обученной на всех признаках при различных конфигурациях. Сравнить результаты работы моделей"

#### Решение задачи предсказания
Из csv файла выргузим набор данных, выделим параметр для предсказания - (столбец `HeartDisease`), и его признаки - все остальные столбцы. Разделим данные на обучающую и тестовые выборки, при условии, что 99.9% данных - для обучения, а остальные для тестов:
```python
х, y = [df.drop("HeartDisease", axis=1).values, df["HeartDisease"].values]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.001, random_state=42)
```
Создадим класс нейронной сети и определим варьируемые конфигурации. 

`hidden_layer_sizes ` - параметр, принимающий на вход количество скрытых слоёв нейронной сети и количество нейронов в каждом слое. Для определения его наилучшего значения необходимо взять минимальное количество слоёв и нейронов в слое и постепенно увеличивать его, до тех пор, пока качество модели не перестанет улучшаться или не будет достаточным.
> **Note**
>
> Экспериментально для нейронной сети `MLPRegressor` было выявленно наилучшее значение равное 100 слоям нейронной сети по 50 нейронов в каждой. Для прелоставления данных процесс оказался очень длительным, поэтому будет указан только наилучший результат.

`activation` - функция активации. В классе представлена 4мя решениями:
- `identity` - функция `f(x) = x`, абсолютно линейная идентичная функция для приведения работы нейронной сети ближе к модели линейной регрессии,
- `logistic` - логистическая сигмовидная функция вида `f(x) = 1 / (1 + exp(-x))`,
- `tanh` - гиперболическая функция тангенса `f(x) = tanh(x)`,
- `relu` - функция выпрямленной линейной единицы измерения `f(x) = max(0, x)`, проверяет больше ли х нуля (используется чаще всего).

`solver` - метод оптимизации весов. Существует в 3х вариациях:
- `Bfgs` - оптимизатор из семейства квазиньютоновских методов, 
> **Warning**
>
> Оптимизатор из семейства квазиньютоновских методов показал себя как очень жадный по времени выполнения алгоритм при этом использующий большие коэфициенты весов, что приводило к едиичным, но слишком большим погрешностям на данных. Поэтому в эксперименте варьирования он не принимал участия.

- `sgd` - метод стозастического градиентного спуска (классика),
- `adam` - оптимизированный метод стозастического градиентного спуска Кингмы, Дидерика и Джимми Барнсома.

```python
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
mlp.fit(x_train, y_train)
y_predict = mlp.predict(x_test)
err = pred_errors(y_predict, y_test)
```
Проведём эксперимент варьирования конфигураций, посчитаем ошибки предсказания и выберем наилучшую нейронную сеть.

#### Эксперимент варьирования
Рассмотрим различные функции активации.

Графики решения задачи предсказания на разных функциях активации:

![](1.png "")

Теперь для выбранной функции подберём лучший метод оптимизации весов.

Грфики решения задачи предсказания на разных методах оптимизации весов:

![](2.png "")

### Вывод
Согласно графиком, наилучшие результаты показала нейронаая сеть с функцией активации гиперболического тангенса `tanh` и методом оптимизации весов путём оптимизированного стозастического градиентного спуска Кингмы, Дидерика и Джимми Барнсома `adam`.

В целом нейронная сеть справилась неудовлетворительно с задачей предсказания, показав хоть и небольшую среднеквадратическую ошибку в 0.25, но очень низкий коэфициент детерминации в 0.23 максимально. 

Это значит, что теоретически модель может предсказать результат по признакам, однако понимания зависимостей результата от последних у неё мало. 