import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

@njit(inline="always") # данный декоратор используется для увеличения скорости работы алгоритма
def ae_steps_counter(a: int, b: int) -> int: # функция ae_steps_counter проводит подсчёт шагов Алгоритма Евклида для двух целых чисел
    steps = 0
    while b!=0:
        a,b = b,a%b
        steps += 1
    return steps

@njit(inline="always")
def nomonte_ae_steps_counter(n: int) -> float:
    counter = 0
    total_steps = 0  # данная функция просчитывает точное среднее количество шагов алгоритма Евклида (без использования метода Монте-Карло)
    for i in range(1,n+1):
        for j in range(1,n+1):
            total_steps += ae_steps_counter(i,j)
            counter += 1
    return total_steps / counter if counter > 0 else 0

@njit(parallel=True)
def avg_ae_steps_counter(n: int, samples=5000000) -> float: # функция avg_ae_steps_counter подсчитывает среднее количество шагов Алгоритма Евклида для всех пар чисел до n
    total_steps = 0
    for _ in prange(samples):
        a = np.random.randint(1,n+1) # для ускорения работы программы введён метод Монте-Карло, который берёт случайные значения из заданного диапазона и на основе выборки подсчитывает результат
        b = np.random.randint(1, n+1)
        total_steps += ae_steps_counter(a,b)
    return total_steps / samples


def run_visualization():
    theoretical_slope = (12 * np.log(2)) / (np.pi ** 2)  # константа наклона по теории
    n_values = sorted([10 ** i for i in range(3, 11)])  # список n от 10^3 до 10^10
    average_steps = np.array([avg_ae_steps_counter(n) for n in n_values])  # расчет ср. шагов для каждого n
    ln_n = np.log(np.array(n_values))  # перевод n в натуральный логарифм для регрессии
    slope, intercept = np.polyfit(ln_n, average_steps, 1)  # расчет коэф. линейной регрессии
    theoretical_steps = theoretical_slope * ln_n + intercept  # расчет прямой по теории
    # theory_val = theoretical_slope * ln_n  # теория без подгонки intercept
    # print(f"\n{'№':<4} {'n':<12} {'T(n) эксп.':<14} {'T(n) теор.':<14} {'Абс. погр.':<14} {'Отн. погр.'}")
    # print("-" * 70)
    # for i, (n, exp, theor) in enumerate(zip(n_values, average_steps, theory_val), 1):
    #     abs_err = abs(exp - theor)
    #     rel_err = (abs_err / theor) * 100
    #     print(f"{i:<4} {n:<12} {exp:<14.4f} {theor:<14.4f} {abs_err:<14.4f} {rel_err:.4f}%")
    diff = average_steps - theoretical_steps  # вычисление разности (ошибки)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(n_values, average_steps, 'ro', label='Эксперимент', markersize=5)
    ax1.plot(n_values, theoretical_steps, color='black', alpha=0.5, label='Теория')
    ax1.set_ylabel('Среднее число шагов $T(n)$')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    ax2.vlines(n_values, 0, diff, color='blue', alpha=0.3)
    ax2.plot(n_values, diff, 'bo', markersize=4, label='Отклонение (Эксп - Теор)')
    ax2.set_ylabel(r'Разница $\Delta$')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_xlabel('Параметр $n$ (логарифмическая шкала)')
    plt.tight_layout()
    plt.show()


def main_menu(): # функция main_menu исполняет роль главного меню
    theoretical_slope = (12 * np.log(2)) / (np.pi ** 2) # переменная содержит коэффициент перед ln(n) по т. Хайльбронна-Диксона
    while True: # зацикливание дополняет главное меню, для удобства пользования
        print("\n" + "=" * 30)
        print("   АНАЛИЗ АЛГОРИТМА ЕВКЛИДА")
        print("=" * 30)
        print("1. Рассчитать среднее для конкретного n") # данные элементы являются декоративными
        print("2. Построить график (сравнение с теорией)")
        print("3. Построить график остаточного члена о(1)")
        print("4. Выход")
        choice = input("\nВыберите действие (1-4): ") # в переменной choice находится номер опции, выбранной пользователем
        if choice == '1': # в данном блоке находится подсчёт среднего количества шагов алгоритма Евклида
            try: # блок try используется для обработки ошибок
                user_n = int(input("Введите n: ")) # переменная получает на вход n от пользователя
                samples = int(input("Количество выборок (Нажмите Enter для выставления 5 млн (по умолчанию)): ") or 5000000) # переменная получает на вход количество выборок, по умолчанию значение 5 миллионов
                exp_val = avg_ae_steps_counter(user_n, samples=samples) # в данную переменную помещается экспериментальное значение количества шагов алгоритма Евклида (в среднем)
                theory_val = theoretical_slope * np.log(user_n) # в данную переменную помещается теоретическое значение количества шагов алгоритма Евклида (по т. Хайльбронна-Диксона)
                error = abs(exp_val - theory_val) # в переменную error помещается абсолютная погрешность
                rel_error = (error / theory_val) * 100 if theory_val != 0 else 0 # в переменную rel_error помещается относительная погрешность
                print(f"\n--- Результаты для n = {user_n} ---")
                print(f"Эксперимент ({samples} тестов): {exp_val:.6f}")
                print(f"Теория (асимптотика):         {theory_val:.6f}") # данный блок выводит всю полученную информацию
                print(f"Абсолютная погрешность:       {error:.6f}")
                print(f"Относительная погрешность:    {rel_error:.4f}%")
            except ValueError: # в блоке except происходит обработка ошибки и вывод соответствующего сообщения
                print("\n[Ошибка] Пожалуйста, вводите только целые числа.")
        elif choice == '2': # в данном блоке находится визуализация графика T(n) от ln(n) для экспериментального значения
            print("\nГенерация графика... Пожалуйста, подождите.")
            run_visualization()
        elif choice == "3": # в данном блоке просчитывается практическое значение C, далее просчитывается о(1) и строится график
            choice_mc_method = input("Построить график о(1) с использованием метода Монте-Карло? (ускорит работу и позволит увеличить n, однако график неточен) (y/n) ") # данная переменная хранит в себе выбор пользователя об использовании метода Монте-Карло
            pi = np.pi
            ln2 = np.log(2)                     # в данные переменные помещаются различные математическое постоянные
            gamma = 0.5772156649015328  # Постоянная Эйлера-Маскерони
            zeta_prime_2 = -0.93754825431 # значение производной дзета-функции Римана в точке 2
            c_porter = (6 * ln2 / pi ** 2) * (3 * ln2 + 4 * gamma - (24 / pi ** 2) * zeta_prime_2 - 2) - 0.5 # просчитывается значение константы Портера с помощью производной дзета-функции Римана
            theoretical_slope = (12 * ln2) / (pi ** 2) # просчитывается коэффициент при ln(n) в формуле Хайльбронна-Диксона
            c_theory = theoretical_slope*(math.log(4/3) - gamma + zeta_prime_2/(math.pi**2/6) - 0.5) + c_porter # просчитывается константа C по формуле
            error_array = [] # объявляется массив остаточных членов o(1) для метода без использования Монте-Карло
            error_array_mc = [] # обьявляется массив остаточных членов о(1) для метода с использованием Монте-Карло
            if choice_mc_method in "NnТт0": # обрабатывается отрицательный ответ на вопрос об использовании метода Монте-Карло с учётом опечаток
                n_values = [5, 10, 20, 30, 40, 50, 70, 100, 500, 1000, 5000, 10000, 20_000] # обьявляется массив, содержащий относительно небольшие значения n
                print("\nПроверка гипотезы Портера без использования метода Монте-Карло. Пожалуйста, подождите...")
                for n in n_values: # для каждого n просчитывается экспериментальное значение C и о(1)
                    avg_steps = nomonte_ae_steps_counter(n) # просчитывается среднее количество шагов полным перебором
                    c_experimental = avg_steps - theoretical_slope * np.log(n)  # C = T(n) - (12ln2/pi**2)*ln(n)
                    error_array.append(c_experimental-c_theory) # в массив error_array добавляется значение о(1)
                plt.figure(figsize=(10, 6))
                plt.plot(n_values, error_array, 'o-', label='o(1) (без метода Монте-Карло)')
                plt.axhline(y=0, color='r', linestyle='--', label='Теоретический предел (0)')
                plt.xscale('log')
                plt.xlabel('n (логарифмическая шкала)')
                plt.ylabel('Отклонение от теоретической константы')           # строится график остаточных членов о(1)
                plt.title('Проверка гипотезы Портера: сходимость остаточного члена к нулю (без метода Монте-Карло)')
                plt.grid(True, which="both", ls="-", alpha=0.5)
                plt.legend()
                plt.show()

            elif choice_mc_method in "YyНн1": # обрабатывается положительный ответ на вопрос об использовании метода Монте-Карло с учётом опечаток
                print("\nПроверка гипотезы Портера с использованием метода Монте-Карло. Пожалуйста, подождите...")
                n_values_mc = [10**i for i in range(1,14)] # бóльшие значения для метода Монте-Карло

                for n in n_values_mc:  # для каждого n просчитывается экспериментальное значение C и о(1)
                    avg_steps = avg_ae_steps_counter(n) # просчитывается среднее количество шагов с помощью метода Монте-Карло
                    c_experimental = avg_steps - theoretical_slope * np.log(n) # C = T(n) - (12ln2/pi**2)*ln(n)
                    error_array_mc.append(c_experimental - c_theory)# в массив error_array_mc добавляется значение о(1)

                plt.figure(figsize=(10, 6))
                plt.plot(n_values_mc, error_array_mc, 'o-', label='o(1) (с методом Монте-Карло)')
                plt.axhline(y=0, color='r', linestyle='--', label='Теоретический предел (0)')
                plt.xscale('log')
                plt.xlabel('n (логарифмическая шкала)')           # строится график о(1)
                plt.ylabel('Отклонение от теоретической константы')
                plt.title('Проверка гипотезы Портера: сходимость остаточного члена к нулю')
                plt.grid(True, which="both", ls="-", alpha=0.5)
                plt.legend()
                plt.show()
            else: # обрабатывается ошибка ввода с выходом в главное меню
                print("Ошибка ввода опции! Возврат к главному меню...\n")
                continue
        elif choice == '4': # в данном блоке обрабатывается конец работы программы
            print("Программа завершена.")
            break
        else:
            print("\n[Ошибка] Неверный ввод. Попробуйте еще раз.") # блок else обрабатывает неверный ввод номера операции

if __name__ == "__main__": # данный условный блок отвечает за корректный запуск программы
    main_menu()
