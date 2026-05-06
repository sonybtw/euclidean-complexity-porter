import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit
theoretical_slope = (12 * np.log(2)) / (np.pi**2) # ≈ 0.84276

@njit # данный декоратор используется для увеличения скорости работы алгоритма
def AE_steps_counter(a: int, b: int) -> int: # функция AE_steps_counter проводит посчёт шагов Алгоритма Евклида для двух целых чисел
    steps = 0
    while b!=0:
        a,b = b, a%b
        steps += 1
    return steps

@njit
def avg_AE_steps_counter(n: int, samples=5000000) -> float: # функция avg_AE_steps_counter подсчитывает среднее количество шагов Алгоритма Евклида для всех пар чисел до n
    total_steps = 0
    for _ in range(samples):
        a = np.random.randint(1,n+1) # для ускорения работы программы введён метод Монте-Карло, который берёт случайные значения из заданного диапазона и на основе выборки подсчитывает результат
        b = np.random.randint(1, n+1)
        total_steps += AE_steps_counter(a,b)
    return total_steps / samples

n_values = sorted([n*10**i for n in [1,5] for i in range(3,7)]) # массив n_values содержит числа 10^3; 5*10^3 и т. д. Данные значения используются для корректного построения графика
average_steps = [avg_AE_steps_counter(n) for n in n_values] # соответствующие n_values значения среднего количества шагов Алгоритма Евклида

ln_n = np.log(np.array(n_values)) # просчитаны значения ln_n для всех n из n_values
average_steps = np.array(average_steps) # массив average_steps преобраззован в массив numpy для ускорения работы 

slope, intercept = np.polyfit(ln_n, average_steps, 1) # slope - экспериментальный коэффициент перед ln_n; intercept - приближение константы Портера

# с помошью pyplot был построен график

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

print(f"Экспериментальный коэффициент: {slope:.5f}")
print(f"Теоретический коэффициент: {theoretical_slope:.5f}")
print(f"Погрешность: {abs(slope - theoretical_slope)/theoretical_slope:.2%}")
