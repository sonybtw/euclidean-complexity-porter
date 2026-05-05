import numpy as np
import matplotlib.pyplot as plot
import math
from numba import njit

@njit
def AE_steps_counter(a: int, b: int) -> int:
    steps = 0
    while b!=0:
        a,b = b, a%b # ВАЖНО: если ввести сначала меньшее число, потом большее, то алгоритм сделает на 1 шаг больше!
        steps += 1 # можно фиксануть при вводе чисел
    return steps

@njit
def avg_AE_steps_counter(n: int, samples=5000000) -> float:
    total_steps = 0
    for _ in range(samples):
        a = np.random.randint(1,n+1)
        b = np.random.randint(1, n+1)
        total_steps += AE_steps_counter(a,b)
    return total_steps / samples

n_values = sorted([n*10**i for n in [1,5] for i in range(3,7)])
average_steps = [avg_AE_steps_counter(n) for n in n_values]

ln_n = [math.log(n) for n in n_values]
slope, intercept = np.polyfit(ln_n, average_steps, 1)

for i in range(len(average_steps)):
    print(f"Для n={n_values[i]} среднее значение шагов составило: {average_steps[i]}")
print(f"Угловой коэффициент: {slope}")
