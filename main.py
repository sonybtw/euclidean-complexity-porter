import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit, prange
from numba.np.ufunc import parallel


@njit(inline="always") # данный декоратор используется для увеличения скорости работы алгоритма
def AE_steps_counter(a: int, b: int) -> int: # функция AE_steps_counter проводит посчёт шагов Алгоритма Евклида для двух целых чисел
    steps = 0
    while b!=0:
        a,b = b, a%b
        steps += 1
    return steps

@njit(parallel=True)
def avg_AE_steps_counter(n: int, samples=5000000) -> float: # функция avg_AE_steps_counter подсчитывает среднее количество шагов Алгоритма Евклида для всех пар чисел до n
    total_steps = 0
    for _ in prange(samples):
        a = np.random.randint(1,n+1) # для ускорения работы программы введён метод Монте-Карло, который берёт случайные значения из заданного диапазона и на основе выборки подсчитывает результат
        b = np.random.randint(1, n+1)
        total_steps += AE_steps_counter(a,b)
    return total_steps / samples



def main_menu():
    theoretical_slope = (12 * np.log(2)) / (np.pi ** 2)
    while True:
        print("\n" + "=" * 30)
        print("   АНАЛИЗ АЛГОРИТМА ЕВКЛИДА")
        print("=" * 30)
        print("1. Рассчитать среднее для конкретного N")
        print("2. Построить график (сравнение с теорией)")
        print("3. Выход")

        choice = input("\nВыберите действие (1-3): ")

        if choice == '1':
            try:
                user_n = int(input("Введите N: "))
                s = int(input("Количество выборок (Enter для 5 млн): ") or 5000000)

                exp_val = avg_AE_steps_counter(user_n, samples=s)
                theory_val = theoretical_slope * np.log(user_n)
                error = abs(exp_val - theory_val)
                rel_error = (error / theory_val) * 100 if theory_val != 0 else 0

                print(f"\n--- Результаты для N = {user_n} ---")
                print(f"Эксперимент ({s} тестов): {exp_val:.6f}")
                print(f"Теория (асимптотика):      {theory_val:.6f}")
                print(f"Абсолютная погрешность:    {error:.6f}")
                print(f"Относительная погрешность: {rel_error:.4f}%")

            except ValueError:
                print("\n[Ошибка] Пожалуйста, вводите только целые числа.")

        elif choice == '2':
            print("\nГенерация графика... Пожалуйста, подождите.")
            run_visualization()

        elif choice == '3':
            print("Программа завершена.")
            break
        else:
            print("\nНеверный ввод. Попробуйте еще раз.")



def run_visualization():
    theoretical_slope = (12 * np.log(2)) / (np.pi ** 2)
    n_values = sorted([n * 10 ** i for n in [1, 5] for i in range(3,7)])
    average_steps = [avg_AE_steps_counter(n) for n in n_values]
    ln_n = np.log(np.array(n_values))
    average_steps = np.array(average_steps)
    slope, intercept = np.polyfit(ln_n, average_steps, 1)
    plt.figure(figsize=(10, 6))
    plt.scatter(ln_n, average_steps, color='red', label='Экспериментальные данные')
    plt.plot(ln_n, slope * ln_n + intercept, 'b--', label=f'Аппроксимация (slope={slope:.4f})')
    plt.plot(ln_n, theoretical_slope * ln_n + intercept, 'g:', label=f'Теория (угл. коэффициент≈0.8428)')

    plt.title('Зависимость среднего числа шагов $T(n)$ от $\ln(n)$')
    plt.xlabel('$\ln(n)$')
    plt.ylabel('Среднее число шагов $T(n)$')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main_menu()
