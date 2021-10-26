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
    xn = list()
    while r - l > eps and abs(f(m)) > eps:
        m = l + (r - l) / delta
        xn.append(m)
        if np.sign(f(l)) != np.sign(f(m)):
            r = m
        else:
            l = m
    return m, len(xn)


def bisection(a, b, eps, f):
    return dichotomy(a, b, eps, f)


def golden_section(a, b, eps, f):
    return dichotomy(a, b, eps, f, (1 + np.sqrt(5)) / 2)


def newton(x0, eps, f, diff):
    xn = list()
    xn.append(x0)
    while True:
        xnext = x0 - (f(x0) / diff(x0))
        xn.append(xnext)
        if abs(xnext - x0) <= eps and abs(f(xnext)) <= eps:
            break
        x0 = xnext
    return xnext, (len(xn) - 1), getRateOfConvergence(xn)


def relaxation(x0, eps, f, diff, tau):
    tau = abs(tau) * np.sign(diff(x0))
    xn = list()
    xn.append(x0)
    while True:
        xnext = x0 - tau * f(x0)
        xn.append(xnext)
        if abs(xnext - x0) <= eps and abs(f(xnext)) <= eps:
            break
        x0 = xnext
    return xnext, (len(xn) - 1)


def chord(x0, x1, eps, f):
    xn = list()
    xn.append(x0)
    xn.append(x1)
    while True:
        x0, x1 = x1, x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        xn.append(x1)
        if abs(x1 - x0) <= eps and abs(f(x1)) <= eps:
            break
    return x1, (len(xn) - 2)


def getRateOfConvergence(xn):
    R = list()
    for k in range(3, len(xn)):
        R.append(np.log((xn[k] - xn[k - 1]) / (xn[k - 1] - xn[k - 2])) / np.log(
            (xn[k - 1] - xn[k - 2]) / (xn[k - 2] - xn[k - 3])))
    return R


def main1():
    eps = 10 ** (-15)
    a = 0
    b = 10

    print("bisection", bisection(a, b, eps, func))
    print("golden_section", golden_section(a, b, eps, func))
    print("relaxation", relaxation((a + b) / 2, eps, func, diff_func, 1 / 2))
    print("newton", newton((a + b) / 2, eps, func, diff_func))
    print("chord", chord((a + b) / 2, (a + b) / 2 + 2 * eps, eps, func))


def main2():
    a = 0
    b = 10
    iterations_count = dict()
    eps_min = 10**(-7)
    eps_max = 10**(-3)
    eps_step = 10**(-7)
    eps_range = np.arange(eps_min, eps_max, eps_step)
    iterations_count[bisection] = list()
    iterations_count[golden_section] = list()
    iterations_count[relaxation] = list()
    iterations_count[newton] = list()
    iterations_count[chord] = list()
    for eps in eps_range:
        iterations_count[bisection].append(bisection(a, b, eps, func)[1])
        iterations_count[golden_section].append(golden_section(a, b, eps, func)[1])
        iterations_count[relaxation].append(relaxation((a + b) / 2, eps, func, diff_func, 1 / 2)[1])
        iterations_count[newton].append(newton((a + b) / 2, eps, func, diff_func)[1])
        iterations_count[chord].append(chord((a + b) / 2, (a + b) / 2 + 2 * eps, eps, func)[1])
    i = 0
    N_bisection = list()
    for eps in eps_range:
        N_bisection.append(int(np.log2(abs((b-a) / eps))) + 1)
    for key in iterations_count:
        i += 1
        plt.subplot(1, len(iterations_count), i)
        plt.title(key.__name__)
        plt.xlabel("epsilon")
        plt.ylabel("Iterations")
        plt.grid()
        plt.plot(eps_range, iterations_count[key], color='k', label='Число итераций')
        plt.legend()

    plt.subplot(1, len(iterations_count), 1)
    plt.plot(eps_range, N_bisection, color='k', ls='--')
    plt.show()


main1()
main2()
