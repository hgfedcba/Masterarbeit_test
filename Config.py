import math
import numpy as np
import torch
import pytest
import time
import other_calcs
import torch.optim as optim


class Config:
    def __init__(self, string, log):
        self.other_computation_exists = False

        self.parameter_string = ""

        if string == "1":
            # 1 american puts
            self.T = 10
            self.N = 5
            self.xi = 30
            self.lr = 0.01
            self.random_seed = 23343

            self.d = 1  # dimension
            self.r = 0.1  # interest rate
            self.beta = 0.2  # volatility
            self.K = 40  # strike price
            self.delta = 0  # dividend rate

            def sigma(x):
                if isinstance(x, int) or isinstance(x, float):
                    out = self.beta
                else:
                    out = self.beta * np.identity(x.shape[0])
                return out

            self.sigma = sigma

            """
            def mu(x):
                out = (r - delta) * x
                return out
            """

            def mu(x):
                if isinstance(x, int) or isinstance(x, float):
                    out = (self.r - sigma(x) ** 2 / 2)
                else:
                    # TODO: Higher dimension is NOT supported here
                    out = (r - delta) * np.ones(x.shape[0])
                return out

            self.mu = mu

            def g(t_in, x):
                t = torch.ones(1) * t_in
                sum = torch.zeros(1)
                c = torch.ones(1) * self.K
                for j in range(self.d):
                    sum += torch.max(c - x[j], torch.zeros(1))
                return torch.exp(-self.r * t) * sum

            self.g = g

            self.other_computation_exists = True

            a1 = self.binomial_trees(self.xi, r, sigma(1), self.T, self.N * 10, self.K)
            a2 = self.binomial_trees_BS(self.xi, r, mu(1), sigma(1), self.T, self.N * 10 / self.T, K)
            assert a1 == a2 or mu(1) != r - sigma(1) ** 2 / 2
            # other_calcs.finite_differences(r, sigma(1), self.xi, K, self.T)

            # start_time = time.time()
            # assert binomial_trees(100, 0.03, 0.24, 0.75, 500, 95) == pytest.approx(5.047, 0.001)
            assert self.binomial_trees_BS(100, 0.03, 3.0 / 2500, 0.24, 0.75, 500 / 0.75, 95) == pytest.approx(5.047, 0.001)
            # print("--- %s seconds ---" % (time.time() - start_time))

            # start_time = time.time()
            # h1 = binomial_trees(self.xi, r, sigma(1), self.T, self.N * 10, K)
            # assert mu(1) == r - sigma(1) ** 2 / 2
            self.other_computation = a2
            # h = self.other_computation
            # h1 = mu(1)

            # print("--- %s seconds ---" % (time.time() - start_time))
            assert mu(1) > 0
            assert sigma(1) > 0

        elif string == "2":
            # 2 american puts
            self.internal_neurons = 50
            self.activation1 = torch.tanh
            self.activation2 = torch.sigmoid
            self.optimizer = optim.Adam

            self.max_number_iterations = 50
            self.batch_size = 16
            self.val_size = 16
            self.T = 10
            self.N = 10
            self.xi = 40  # der unterschied zwischen 38 und 40 sind 15.9-15 beim baum. Das finde ich falsch.
            self.lr = 0.0001  # lernrate
            self.lr_sheduler_breakpoints = [50, 100, 1000]
            self.random_seed = 23343

            self.d = 1  # dimension
            # TODO: diskontieren?!
            self.r = 0.05  # interest rate
            self.K = 40  # strike price
            self.delta = 0  # dividend rate

            constant_sigma = 0.25

            # TODO: richtig?
            constant_sigma = constant_sigma ** 0.5

            def sigma(x):
                if isinstance(x, int) or isinstance(x, float):
                    out = constant_sigma
                else:
                    out = constant_sigma * np.identity(x.shape[0])
                return out

            self.sigma = sigma

            """
            def mu(x):
                out = (r - delta) * x
                return out
            """

            # constant_mu = 0.2
            constant_mu = self.r - constant_sigma ** 2 / 2

            def mu(x):
                if isinstance(x, int) or isinstance(x, float):
                    out = constant_mu
                else:
                    # TODO: Higher dimension is NOT supported here
                    out = constant_mu * np.ones(x.shape[0])
                return out

            self.mu = mu
            """
            def g(t_in, x):
                t = torch.ones(1) * t_in
                sum = torch.zeros(1)
                c = torch.ones(1) * K
                for j in range(self.d):
                    sum += torch.max(c - x[j], torch.zeros(1))
                    # sum += c - x[j]
                # return torch.exp(-r * t) * sum
                return sum
            """

            def g(t_in, x):
                sum = 0
                for j in range(self.d):
                    sum += max(self.K - x[j].item(), 0)
                    # sum += c - x[j]
                # return torch.exp(-r * t) * sum
                return sum

            self.g = g

            self.compute_other_value()

            assert sigma(1) > 0
            assert self.r >= 0

    def binomial_trees(self, S0, r, sigma, T, N, K):
        delta_T = T / N

        alpha = math.exp(r * delta_T)
        beta_local = ((alpha ** -1) + alpha * math.exp(sigma ** 2 * delta_T)) / 2
        u = beta_local + (beta_local ** 2 - 1) ** 0.5
        d = u ** -1
        q = (alpha - d) / (u - d)
        assert 1 > q > 0

        S = np.ones((N + 1, N + 1)) * S0

        for i in range(1, N + 1):
            for j in range(i + 1):
                S[j][i] = S[0][0] * (u ** j) * (d ** (i - j))
                assert True

        V = np.ones((N + 1, N + 1))
        V_map = np.ones((N + 1, N + 1)) * -2
        for i in range(N, -1, -1):
            for j in range(i, -1, -1):
                if i == N:
                    V[j][i] = max(K - S[j][i], 0)
                else:
                    h1 = max(K - S[j][i], 0)
                    h2 = alpha ** -1 * (q * V[j + 1][i + 1] + (1 - q) * V[j][i + 1])
                    V[j][i] = max(h1, h2)
                    V_map[j][i] = h1 > h2  # a 1 indicates exercising is good
        return V[0][0]  # tested

    def binomial_trees_BS(self, S0, r_const, mu, sigma, T, k, K):
        r = math.exp(r_const / k) - 1
        beta = 0.5 * (1 + r) ** -1 + 0.5 * (1 + r) * math.exp(sigma ** 2 / k)
        u = beta + (beta ** 2 - 1) ** 0.5
        d = 1 / u
        p = (math.exp((mu + sigma ** 2 / 2) / k) - d) / (u - d)
        assert 0 < p < 1, "Increase q to resolve this issue"

        N = int(k * T)

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
        return V[0][0]  # identical output to above when mu = r-sigma**2 and k = N/T

    def compute_other_value(self):
        self.other_computation_exists = True
        if self.mu(1) == self.r - self.sigma(1) ** 2 / 2:
            self.other_computation = self.binomial_trees(self.xi, self.r, self.sigma(1), self.T, self.N * 10, self.K)
        else:
            self.other_computation = self.binomial_trees_BS(self.xi, self.r, self.mu(1), self.sigma(1), self.T, self.N / self.T * 10, self.K)
        print("other computation yield: " + str(self.other_computation))
