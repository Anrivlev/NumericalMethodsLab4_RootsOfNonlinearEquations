import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 4 * np.sin(x / 2) + np.cos(x) * np.tanh(x) - x + 2


def xroot():
    return 4.7927776110652285104


def diff_func(x):
    return 2 * np.cos(x / 2) - np.sin(x) * np.tanh(x) + np.cos(x) / (np.cosh(x)**2) - 1


def dichotomy(a, b, eps, f, delta=2):
    # delta == 2 means bisection
    if abs(f(a)) <= eps:
        return a
    if abs(f(b)) <= eps:
        return b
    l, r = a, b
    m = l + (r - l) / delta
    while r - l > eps and abs(f(m)) > eps:
        m = l + (r - l) / delta
        if np.sign(f(l)) != np.sign(f(m)):
            r = m
        else:
            l = m
    return m


def golden_section(a, b, eps, f):
    return dichotomy(a, b, eps, f, (1 + np.sqrt(5)) / 2)


def newton(x0, eps, f, diff):
    while True:
        xnext = x0 - (f(x0) / diff(x0))
        if abs(xnext - x0) <= eps and abs(f(xnext)) <= eps:
            break
        x0 = xnext
    return xnext


def relaxation(x0, eps, f, diff, tau):
    tau = abs(tau) * np.sign(diff(x0))
    while True:
        xnext = x0 - tau * f(x0)
        if abs(xnext - x0) <= eps and abs(f(xnext)) <= eps:
            break
        x0 = xnext
    return xnext


def chord(x0, x1, eps, f):
    while True:
        x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x1 - x0) <= eps and abs(f(x1)) <= eps:
            break
    return x1


def main():
    eps = 10**(-15)
    a = 0
    b = 10

    print(dichotomy(a, b, eps, func))
    print(golden_section(a, b, eps, func))
    print(relaxation((a + b) / 2, eps, func, diff_func, 1))
    print(newton(b, eps, func, diff_func))
    print(chord((a + b) / 2, (a + b) / 2 + 2 * eps, eps, func))


main()
