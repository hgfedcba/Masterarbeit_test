import math
import numpy as np
import torch
import pytest
import time
import other_calcs


class Config:
    def __init__(self, string):
        self.other_computation_exists = False

        if string == "1":
            # 1 american puts
            self.T = 10
            self.N = 5
            self.xi = 30
            self.gamma = 0.01
            self.random_seed = 23343

            self.d = 1  # dimension
            r = 0.1  # interest rate
            beta = 0.2  # volatility
            K = 40  # strike price
            delta = 0  # dividend rate

            def sigma(x):
                if isinstance(x, int) or isinstance(x, float):
                    out = beta
                else:
                    out = beta * np.ones(x.shape[0])
                return out

            self.sigma = sigma

            """
            def mu(x):
                out = (r - delta) * x
                return out
            """

            def mu(x):
                if isinstance(x, int) or isinstance(x, float):
                    out = (r - sigma(x) ** 2 / 2)
                else:
                    out = (r - sigma(x) ** 2 / 2) * np.ones(x.shape[0])
                return out

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


            # other_calcs.finite_differences(r, sigma(1), self.xi, K, self.T)

            # start_time = time.time()
            assert binomial_trees(100, 0.03, 0.24, 0.75, 500, 95) == pytest.approx(5.047, 0.001)
            # print("--- %s seconds ---" % (time.time() - start_time))

            # start_time = time.time()
            h1 = binomial_trees_BS(self.xi, r, mu(1), sigma(1), self.T, self.N * 10, K)
            assert mu(1) == r - sigma(1) ** 2 / 2
            self.other_computation = binomial_trees(self.xi, r, sigma(1), self.T, self.N * 10, K)
            h = self.other_computation
            h1 = mu(1)

            # print("--- %s seconds ---" % (time.time() - start_time))
            assert mu(1) > 0
            assert sigma(1) > 0
