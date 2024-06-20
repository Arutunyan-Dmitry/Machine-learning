import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

'''
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
'''


# Метод оцифровки и нормализации данных
def normalisation(filename):
    fileout = "P:\\ULSTU\\ИИС\\Datasets\\heart_2020_norm.csv"
    df = pd.read_csv(filename, sep=',').dropna()              # Считываем данные с csv файла и удаляем строки, содержащие NaN

    for index, row in df.iterrows():
        if index % 10000 == 0:
            print("normalisation running . . .  " + str(round((index / len(df) * 100), 2)) +'%')
        if "Yes" in row["HeartDisease"]:                       # Имеет ли человек ССЗ (0 / 1)
            df.at[index, "HeartDisease"] = 1
        else:
            df.at[index, "HeartDisease"] = 0
        if "Yes" in row["Smoking"]:                            # Выкурил ли человек хотя бы 5 пачек сигарет за всю жизнь (0 / 1)
            df.at[index, "Smoking"] = 1
        else:
            df.at[index, "Smoking"] = 0
        if "Yes" in row["AlcoholDrinking"]:                    # Сильно ли человек употребляет алкоголь (0 / 1)
            df.at[index, "AlcoholDrinking"] = 1
        else:
            df.at[index, "AlcoholDrinking"] = 0
        if "Yes" in row["Stroke"]:                             # Был ли у человека инсульт (0 / 1)
            df.at[index, "Stroke"] = 1
        else:
            df.at[index, "Stroke"] = 0
        if "Yes" in row["DiffWalking"]:                        # Ииспытывает ли человек трудности при ходьбе (0 / 1)
            df.at[index, "DiffWalking"] = 1
        else:
            df.at[index, "DiffWalking"] = 0
        if "Female" in row["Sex"]:                             # Пол (Ж - 0 / М - 1)
            df.at[index, "Sex"] = 0
        else:
            df.at[index, "Sex"] = 1
        if "18-24" in row["AgeCategory"]:                      # Возрастная категория (средний возраст каждого диапазона)
            df.at[index, "AgeCategory"] = (18 + 24) / 2
        elif "25-29" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (25 + 29) / 2
        elif "30-34" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (30 + 34) / 2
        elif "35-39" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (35 + 39) / 2
        elif "40-44" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (40 + 44) / 2
        elif "45-49" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (45 + 49) / 2
        elif "50-54" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (50 + 54) / 2
        elif "55-59" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (55 + 59) / 2
        elif "60-64" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (60 + 64) / 2
        elif "65-69" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (65 + 69) / 2
        elif "70-74" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (70 + 74) / 2
        elif "75-79" in row["AgeCategory"]:
            df.at[index, "AgeCategory"] = (75 + 79) / 2
        else:
            df.at[index, "AgeCategory"] = (25 + 29) / 2
        if "White" in row["Race"]:                               # Национальная принадлежность человека
            df.at[index, "Race"] = 0                             # White - Европиойды - 0
        elif "Black" in row["Race"]:                             # Black - Негройды - 1
            df.at[index, "Race"] = 1                             # Hispanic - Испанцы - 2
        elif "Hispanic" in row["Race"]:                          # American Indian/Alaskan Native - Индусы - 3
            df.at[index, "Race"] = 2                             # Asian - Азиаты - 4
        elif "American Indian/Alaskan Native" in row["Race"]:    # Other - Другие - 5
            df.at[index, "Race"] = 3
        elif "Asian" in row["Race"]:
            df.at[index, "Race"] = 4
        else:
            df.at[index, "Race"] = 5
        if "Yes" in row["Diabetic"]:                            # Был ли у человека диабет (0 / 1)
            df.at[index, "Diabetic"] = 1
        else:
            df.at[index, "Diabetic"] = 0
        if "Yes" in row["PhysicalActivity"]:                     # Занимался ли человек спротом за последний месяц (0 / 1)
            df.at[index, "PhysicalActivity"] = 1
        else:
            df.at[index, "PhysicalActivity"] = 0
        if "Excellent" in row["GenHealth"]:                      # Общее самочувствие человека
            df.at[index, "GenHealth"] = 4                        # Excellent - Отлично - 4
        elif "Very good" in row["GenHealth"]:                    # Very good - Очень хорошо - 3
            df.at[index, "GenHealth"] = 3                        # Good - Хорошо - 2
        elif "Good" in row["GenHealth"]:                         # Fair - Нормально - 1
            df.at[index, "GenHealth"] = 2                        # "Poor" / "Other..." - Плохое или другое - 0
        elif "Fair" in row["GenHealth"]:
            df.at[index, "GenHealth"] = 1
        else:
            df.at[index, "GenHealth"] = 0
        if "Yes" in row["Asthma"]:                               # Была ли у человека астма (0 / 1)
            df.at[index, "Asthma"] = 1
        else:
            df.at[index, "Asthma"] = 0
        if "Yes" in row["KidneyDisease"]:                        # Было ли у человека заболевание почек (0 /1)
            df.at[index, "KidneyDisease"] = 1
        else:
            df.at[index, "KidneyDisease"] = 0
        if "Yes" in row["SkinCancer"]:                           # Был ли у человека рак кожи (0 / 1)
            df.at[index, "SkinCancer"] = 1
        else:
            df.at[index, "SkinCancer"] = 0

    df = df.applymap(pd.to_numeric, errors='coerce').dropna()    # Гарантированно убираем все нечисловые значения из датасета
    df.to_csv(fileout, index=False)                              # Сохраняем нормализованный датасет для дальнейшей работы
    return fileout


# Метод ранжирования параметров по степени важности
def param_range(filename, elim_kp):
    df = pd.read_csv(filename, sep=',')        # Считываем нормализованные данные и разделяем их на выборки
    x_train = df[["BMI", "Smoking", "AlcoholDrinking", "Stroke", "PhysicalHealth",
            "MentalHealth", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic",
            "PhysicalActivity", "GenHealth", "SleepTime", "Asthma", "KidneyDisease", "SkinCancer"]].iloc[
              0:round(len(df) / 100 * 99)]
    y_train = df["HeartDisease"].iloc[0:round(len(df) / 100 * 99)]
    x_test = df[["BMI", "Smoking", "AlcoholDrinking", "Stroke", "PhysicalHealth",
                  "MentalHealth", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic",
                  "PhysicalActivity", "GenHealth", "SleepTime", "Asthma", "KidneyDisease", "SkinCancer"]].iloc[
              round(len(df) / 100 * 99):len(df)]
    y_test = df["HeartDisease"].iloc[round(len(df) / 100 * 99):len(df)]

    dtc = DecisionTreeClassifier(random_state=241)                                     # Создаём модель дерева решений
    dtc.fit(x_train.values, y_train.values)                                            # Обучаем модель на данных
    y_predict = dtc.predict(x_test.values)                                             # Решаем задачу классификации на полном наборе признаков
    err = pred_errors(y_predict, y_test.values)                                        # Рассчитываем ошибки предсказания
    make_plots(y_test.values, y_predict, err[0], err[1], "Полный набор данных")    # Строим графики

    ranks = np.abs(dtc.feature_importances_)                                                 # Получаем значимость каждого признака в модели
    minmax = MinMaxScaler()                                                                  # Шкалируем и нормализуем значимость
    ranks = minmax.fit_transform(np.array(ranks).reshape(len(x_train.columns), 1)).ravel()
    ranks = map(lambda x: round(x, 2), ranks)
    ranks = dict(zip(x_train.columns, ranks))
    ranks = dict(sorted(ranks.items(), key=lambda x: x[1], reverse=True))                    # Сортируем оценки по максимуму и записываем в словарь

    print("X ranging results: \n")
    del_keys = []                                                    # Исключаем параметры, важность которых меньше elim_kp
    for key, value in ranks.items():
        if value >= elim_kp:
            print(" * " + key + ": " + str(value) + " - Approved")
        else:
            print(" * " + key + ": " + str(value) + " - Eliminated")
            del_keys.append(key)

    for key in del_keys:
        ranks.pop(key)

    return filename, ranks.keys()


# Метод решения задачи классификации, основанный только на значимых параметрах
def most_valuable_prediction(params):
    filename = params[0]
    val_p = params[1]
    df = pd.read_csv(filename, sep=',')
    x_train = df[val_p].iloc[0:round(len(df) / 100 * 99)]
    y_train = df["HeartDisease"].iloc[0:round(len(df) / 100 * 99)]
    x_test = df[val_p].iloc[round(len(df) / 100 * 99):len(df)]
    y_test = df["HeartDisease"].iloc[round(len(df) / 100 * 99):len(df)]

    dtc = DecisionTreeClassifier(random_state=241)
    dtc.fit(x_train.values, y_train.values)
    y_predict = dtc.predict(x_test.values)
    err = pred_errors(y_predict, y_test.values)
    make_plots(y_test.values, y_predict, err[0], err[1], "Только важные параметры")


# Метод рассчёта ошибок
def pred_errors(y_predict, y_test):
    mid_square = np.round(np.sqrt(metrics.mean_squared_error(y_test, y_predict)),3)            # Рассчёт среднеквадратичной ошибки модели
    det_kp = np.round(metrics.accuracy_score (y_test, y_predict), 2)                           # Рассчёт коэфициента детерминации модели
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
    # Работа системы в комплексе
    # Здесь elim_kp - значение пороговой значимости параметра (выбран эмпирически)
    most_valuable_prediction(param_range(normalisation("P:\\ULSTU\\ИИС\\Datasets\\heart_2020_cleaned.csv"), 0.05))


