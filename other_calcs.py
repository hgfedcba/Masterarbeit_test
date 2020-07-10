# Here I put other calculations to test my NN
import numpy as np
import Config


def finite_differences(r, sigma, S0, K, T):
    fancy_v = 0.5
    omega = 1
    a = -5
    b = 5
    m = 100
    nu_max = 100
    epsilon = 0.00001

    delta_x_tilda = (b - a) / m
    delta_t_tilda = sigma ** 2 * T / (2 * nu_max)

    x_tilda = np.zeros(m + 1)
    for i in range(m + 1):
        x_tilda[i] = a + i * delta_x_tilda
    lambda_local = delta_t_tilda / delta_x_tilda ** 2

    w = np.zeros(m)
    for i in range(m):
        w[i] = Config.g(0, x_tilda[i])

    # t_tilda loop
    t_tilda = np.zeros(nu_max)
    for nu in range(nu_max):
        t_tilda[nu] = nu * delta_t_tilda
    for nu in range(nu_max):
        g_indexed = np.zeros(nu_max, m)
        for i in range(m + 1):
            g_indexed[nu][i] = Config.g(t_tilda[nu], x_tilda[i])
            g_indexed[nu + 1][i] = Config.g(t_tilda[nu + 1], x_tilda[i])
        b_indexed = np.zeros(m)
        for i in range(2, m - 1):
            b[i] = w[i] + lambda_local * (1 - fancy_v) * (w[i + 1] - 2 * w[i] + w[i - 1])
        b[1] = w[1] + lambda_local * (1 - fancy_v) * (w[2] - 2 * w[1] + g_indexed[nu][0] + lambda_local * fancy_v * g_indexed[nu + 1][0])
        b[m - 1] = w[m - 1] + lambda_local * (1 - fancy_v) * (g[nu][m] - 2 * w[m - 1] + w[m - 2]) + lambda_local * fancy_v * Config.g[nu + 1][m]

        v_indexed = np.zeros(m)
        for i in range(m):
            v_indexed[i] = max(w[i], Config.g[nu + 1][i])

        v_new = np.zeros(m)
        while np.linalg.norm(v_new - v_indexed) > epsilon:
            assert True
