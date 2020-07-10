import math
import numpy as np
import torch
import pytest
import time


class Config:
    def __init__(self, string):
        self.other_computation_exists = False

        if string == "1":
            # 1 american puts
            self.T = 10
            self.N = 5
            self.xi = 30
            self.gamma = 0.001
            self.random_seed = 23343

            self.d = 1  # dimension
            r = 0.1  # interest rate
            beta = 0.2  # volatility
            K = 40  # strike price
            delta = 0  # dividend rate

            def sigma(x):
                out = beta * x
                return out

            self.sigma = sigma

            """
            def mu(x):
                out = (r - delta) * x
                return out
            """

            def mu(x):
                return r - sigma(x) ** 2 / 2

            self.mu = mu

            def g(t_in, x):
                t = torch.ones(1) * t_in
                sum = torch.zeros(1)
                c = torch.ones(1) * K
                for j in range(self.d):
                    sum += torch.max(c - x[j], torch.zeros(1))
                return torch.exp(-r * t) * sum

            self.g = g

            self.other_computation_exists = True

            def binomial_trees(S0, r, sigma, T, N, K):
                delta_T = T / N

                alpha = math.exp(r * delta_T)
                beta_local = ((alpha ** -1) + alpha * math.exp(sigma ** 2 * delta_T)) / 2
                u = beta_local + (beta_local ** 2 - 1) ** 0.5
                d = u ** -1
                q = (alpha - d) / (u - d)

                S = np.ones((N + 1, N + 1)) * S0

                for i in range(1, N + 1):
                    for j in range(i + 1):
                        S[j][i] = S[0][0] * (u ** j) * (d ** (i - j))
                        assert True

                V = np.ones((N + 1, N + 1))

                for i in range(N, -1, -1):
                    for j in range(i, -1, -1):
                        if i == N:
                            V[j][i] = max(K - S[j][i], 0)
                        else:
                            h1 = max(K - S[j][i], 0)
                            h2 = alpha ** -1 * (q * V[j + 1][i + 1] + (1 - q) * V[j][i + 1])
                            V[j][i] = max(h1, h2)
                return V[0][0]  # tested

            def binomial_trees_BS(S0, r_const, mu, sigma, T, k, K):
                r = math.exp(r_const / k) - 1
                beta = 0.5 * (1 + r) ** -1 + 0.5 * (1 + r) * math.exp(sigma ** 2 / k)
                u = beta + (beta ** 2 - 1) ** 0.5
                d = 1 / u
                p = (math.exp((mu + sigma ** 2 / 2) / k) - d) / (u - d)

                N = k * T

                alpha = math.exp(r_const / k)

                S = np.ones((N + 1, N + 1)) * S0

                for i in range(1, N + 1):
                    for j in range(i + 1):
                        S[j][i] = S[0][0] * (u ** j) * (d ** (i - j))
                        assert True

                V = np.ones((N + 1, N + 1))

                for i in range(N, -1, -1):
                    for j in range(i, -1, -1):
                        if i == N:
                            V[j][i] = max(K - S[j][i], 0)
                        else:
                            h1 = max(K - S[j][i], 0)
                            h2 = alpha ** -1 * (p * V[j + 1][i + 1] + (1 - p) * V[j][i + 1])
                            V[j][i] = max(h1, h2)
                return V[0][0]

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
                    w[i] = g(0, x_tilda[i])

                # t_tilda loop
                t_tilda = np.zeros(nu_max)
                for nu in range(nu_max):
                    t_tilda[nu] = nu * delta_t_tilda
                for nu in range(nu_max):
                    g_indexed = np.zeros(nu_max, m)
                    for i in range(m + 1):
                        g_indexed[nu][i] = g(t_tilda[nu], x_tilda[i])
                        g_indexed[nu + 1][i] = g(t_tilda[nu + 1], x_tilda[i])
                    b_indexed = np.zeros(m)
                    for i in range(2, m - 1):
                        b[i] = w[i] + lambda_local * (1 - fancy_v) * (w[i + 1] - 2 * w[i] + w[i - 1])
                    b[1] = w[1] + lambda_local * (1 - fancy_v) * (w[2] - 2 * w[1] + g_indexed[nu][0] + lambda_local * fancy_v * g_indexed[nu + 1][0])
                    b[m - 1] = w[m - 1] + lambda_local * (1 - fancy_v) * (g[nu][m] - 2 * w[m - 1] + w[m - 2]) + lambda_local * fancy_v * g[nu + 1][m]

                    v_indexed = np.zeros(m)
                    for i in range(m):
                        v_indexed[i] = max(w[i], g[nu + 1][i])

                    v_new = np.zeros(m)
                    while np.linalg.norm(v_new - v_indexed) > epsilon:
                        assert True

            # finite_differences(r, sigma(1), self.xi, K, self.T)

            # start_time = time.time()
            assert binomial_trees(100, 0.03, 0.24, 0.75, 500, 95) == pytest.approx(5.047, 0.001)
            # print("--- %s seconds ---" % (time.time() - start_time))

            # start_time = time.time()
            h1 = binomial_trees_BS(self.xi, r, mu(1), sigma(1), self.T, self.N * 10, K)
            self.other_computation = binomial_trees(self.xi, r, sigma(1), self.T, self.N * 10, K)
            h = self.other_computation
            h1 = mu(1)

            # print("--- %s seconds ---" % (time.time() - start_time))
            assert mu(1) > 0
            assert sigma(1) > 0
