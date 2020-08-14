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
        """
        if string == "1":
            # 1 american puts
            self.T = 10
            self.N = 5
            self.xi = 30
            self.initial_lr = 0.01
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

            def mu(x):
                if isinstance(x, int) or isinstance(x, float):
                    out = (self.r - sigma_c_x(x) ** 2 / 2)
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
        """

        # TODO: stoppfunktion lernen, nicht u,U
        if string == "am_put1":
            # 2 american puts
            self.algorithm = 0  # 0 is source, 1 is mine

            self.internal_neurons = 50
            self.activation1 = torch.tanh
            # self.activation1 = torch.nn.functional.selu
            # self.activation1 = torch.nn.SELU()
            self.activation2 = torch.sigmoid
            self.optimizer = optim.Adam

            self.validation_frequency = 2
            self.antithetic_variables = True  # only in validation!

            self.pretrain = True
            self.pretrain_func = self.am_put_default_pretrain
            self.pretrain_iterations = 800

            self.stop_paths_in_plot = True  # TODO:use

            self.max_number_iterations = 51
            self.max_minutes_for_iteration = 5
            self.batch_size = 32
            self.val_size = 64
            self.final_val_size = 512

            self.T = 10
            self.N = 10
            self.xi = 40
            self.initial_lr = 0.0001  # lernrate
            self.do_lr_decay = True
            self.lr_multiplicative_factor = lambda epoch: 0.98
            self.random_seed = 23343

            self.d = 2  # dimension
            # TODO:different
            if self.d > 1:
                self.pretrain = False
            self.r = 0.05  # interest rate
            self.K = 40  # strike price
            self.delta = 0  # dividend rate

            self.sigma_constant = 0.25
            self.mu_constant = self.r

            # TODO: better in code
            # self.preset_config_4312()
            self.preset_config_4411()

            self.sigma = self.sigma_c_x
            self.mu = self.mu_c_x

            self.g = self.american_put

            self.compute_other_value()

            assert self.sigma_c_x(1) > 0
            assert self.r >= 0

    def preset_config_4312(self):
        # only works in 1d
        self.r = 0.06
        self.sigma_constant = 0.4  # beta
        self.mu_constant = self.r
        self.K = 40
        self.xi = 40
        self.T = 1
        self.N = 50
        self.max_number_iterations = 3000
        self.final_val_size = 4096

    def preset_config_4411(self):
        self.r = 0.05
        self.delta = 0.1
        self.sigma_constant = 0.2
        self.K = 100
        self.xi = 100
        self.T = 3
        self.N = 9
        self.d = 2
        # self.max_number_iterations = 3000 + self.d
        self.pretrain_func = self.am_call_default_pretrain
        self.g = self.bermudan_max_call
        # self.val_size = 256

    # TODO:better
    def sigma_c_x(self, x):
        if type(x).__module__ != np.__name__:
            out = self.sigma_constant * x
        else:
            out = self.sigma_constant * (np.identity(x.shape[0]) * x)
        return out

    def mu_c_x(self, x):
        out = (self.mu_constant - self.delta) * x
        return out

    def american_put(self, t_in, x):
        summands = []
        for j in range(self.d):
            summands.append(max(self.K - x[j].item(), 0))
            # sum += c - x[j]
        # return torch.exp(-r * t) * sum
        return math.exp(-self.r * t_in) * sum(summands)

    def bermudan_max_call(self, t_in, x):
        return math.exp(-self.r * t_in) * max(max(x) - self.K, 0)

    def american_put_from_bm(self, t_in, x):
        summands = []
        for j in range(self.d):
            summands.append(x[j].item())
        return math.exp(-self.r * t_in) * max(self.K - self.xi * math.exp((self.r - 0.5 * self.sigma_constant ** 2) * t_in + self.sigma_constant / self.d ** 0.5 + sum(summands)), 0)

    def am_put_default_pretrain(self, x):
        # reasonable
        # return (torch.relu(38 - x) / 30)
        """
        # overly accurate
        h1 = -torch.relu(28 - x) / 10
        h2 = torch.relu(x - 38) / 10
        h3 = (38 - x) / 10
        h4 = h1 + h2 + h3 + 3.8
        return h4
        """
        # TODO:Stupid
        b = (self.xi - 4)
        a = b - 15
        out = (b - x) / 15 + torch.relu(x - b) / 15 - torch.relu(a - x) / 15
        return out

    def am_call_default_pretrain(self, x):
        a = (self.xi + 4)
        b = a + 15
        return (b - x) / 15 + torch.relu(x - b) / 15 - torch.relu(a - x) / 15

    # Model, Net, Meta
    # TODO: 3 obere sachen einzeln. dadurch kann ich auch sehr viel einfacher bestimmte abschnitte aus dem parameterstring entfernen
    def get_parameter_string(self):
        sigma_dict = {
            self.sigma_c_x: str(self.sigma_constant) + " * x"
        }

        mu_dict = {
            self.mu_c_x: str(self.mu_constant) + " * x"
        }

        g_dict = {
            self.american_put: "american put"
        }
        """
        activation_func_dict = {
            torch.tanh            : "torch.tanh",
            torch.sigmoid         : "torch.sigmoid",
            torch.nn.ELU()        : "torch.nn.ELU",
            torch.nn.Hardshrink() : "torch.nn.Hardshrink",
            torch.nn.Hardsigmoid(): "torch.nn.Hardsigmoid",
            torch.nn.Hardtanh()   : "torch.nn.Hardtanh",
            # torch.nn.Hardswish()         : "torch.nn.Hardswish",
            torch.nn.LeakyReLU()  : "torch.nn.LeakyReLU",
            torch.nn.LogSigmoid() : "torch.nn.LogSigmoid",
            # torch.nn.MultiheadAttention(): "torch.nn.MultiheadAttention",
            torch.nn.PReLU()      : "torch.nn.PReLU",
            torch.nn.ReLU()       : "torch.nn.ReLU",
            torch.nn.ReLU6()      : "torch.nn.ReLU6",
            torch.nn.SELU()       : "torch.nn.SELU",
            torch.nn.CELU()       : "torch.nn.CELU",
            torch.nn.GELU()       : "torch.nn.GELU",
            torch.nn.Sigmoid()    : "torch.nn.Sigmoid",
            torch.nn.Softplus()   : "torch.nn.Softplus",
            torch.nn.Softshrink() : "torch.nn.Softshrink",
            torch.nn.Softsign()   : "torch.nn.Softsign",
            torch.nn.Tanh()       : "torch.nn.Tanh",
            torch.nn.Tanhshrink() : "torch.nn.Tanhshrink",
            # torch.nn.Threshold()  : "torch.nn.Threshold"
        }
        """

        """
        'internal_activation_func': [torch.nn.ELU(), torch.nn.Hardshrink(), torch.nn.Hardsigmoid(), torch.nn.Hardtanh(), torch.nn.LeakyReLU(), torch.nn.LogSigmoid(), torch.nn.PReLU(), torch.nn.ReLU(), torch.nn.ReLU6(), torch.nn.SELU(), torch.nn.CELU(), torch.nn.GELU(), torch.nn.Sigmoid(),
                                     torch.nn.Softplus(), torch.nn.Softshrink(), torch.nn.Softsign(), torch.nn.Tanh(), torch.nn.Tanhshrink()],
        """
        activation_func_dict = {
            # TODO: commented functions might work in theory
            torch.tanh                        : "torch.tanh",
            torch.sigmoid                     : "torch.sigmoid",
            # torch.nn.functional.threshold     : "torch.nn.functional.threshold",
            torch.nn.functional.relu          : "torch.nn.functional.relu",
            torch.nn.functional.hardtanh      : "torch.nn.functional.hardtanh",
            # torch.nn.functional.hardswish     : "torch.nn.functional.hardswish",
            torch.nn.functional.relu6         : "torch.nn.functional.relu6",
            torch.nn.functional.elu           : "torch.nn.functional.elu",
            torch.nn.functional.selu          : "torch.nn.functional.selu",
            torch.nn.functional.celu          : "torch.nn.functional.celu",
            torch.nn.functional.leaky_relu    : "torch.nn.functional.leaky_relu",
            # torch.nn.functional.prelu         : "torch.nn.functional.prelu",
            torch.nn.functional.rrelu         : "torch.nn.functional.rrelu",
            # torch.nn.functional.glu           : "torch.nn.functional.glu",
            torch.nn.functional.gelu          : "torch.nn.functional.gelu",
            torch.nn.functional.logsigmoid    : "torch.nn.functional.logsigmoid",
            torch.nn.functional.hardshrink    : "torch.nn.functional.hardshrink",
            torch.nn.functional.tanhshrink    : "torch.nn.functional.tanhshrink",
            torch.nn.functional.softsign      : "torch.nn.functional.softsign",
            torch.nn.functional.softplus      : "torch.nn.functional.softplus",
            torch.nn.functional.softmin       : "torch.nn.functional.softmin",
            torch.nn.functional.softmax       : "torch.nn.functional.softmax",
            torch.nn.functional.softshrink    : "torch.nn.functional.softshrink",
            torch.nn.functional.gumbel_softmax: "torch.nn.functional.gumbel_softmax",
            torch.nn.functional.log_softmax   : "torch.nn.functional.log_softmax",
            torch.nn.functional.hardsigmoid   : "torch.nn.functional.hardsigmoid",
        }
        optimizer_dict = {
            optim.Adam: "Adam"
        }

        pretrain_func_dict = {
            self.am_put_default_pretrain: "am_put_default_pretrain"
        }

        lr_decay_dict = {
            "hi": "hi"  # TODO
        }
        parameter_string = "d: ", self.d, " T: ", self.T, " N: ", self.N, " xi: ", self.xi, " r: ", self.r, " K: ", self.K, " delta: ", self.delta, " sigma(x): ", sigma_dict.get(
            self.sigma), " mu(x): ", mu_dict.get(self.mu), " g: ", g_dict.get(self.g), " internal_neurons: ", self.internal_neurons, " internal_activation_func: ", activation_func_dict.get(
            self.activation1), " final activation func: ", activation_func_dict.get(self.activation2), " optimizer: ", optimizer_dict.get(
            self.optimizer), " pretrain?: ", self.pretrain, " pretrain_func: ", pretrain_func_dict.get(
            self.pretrain_func), " pretrain_iterations: ", self.pretrain_iterations, " max_iterations: ", self.max_number_iterations, " max_duration(min): ", self.max_minutes_for_iteration, \
                           " batch_size: ", self.batch_size, " val_size: ", self.val_size, " final_val_size: ", self.final_val_size, " initial_lernrate: ", self.initial_lr, " lr_factor", \
                           self.lr_multiplicative_factor(1), " validation_frequency: ", self.validation_frequency, " antithetic_variables: ", self.antithetic_variables, " stop_paths_in_plot: ", \
                           self.stop_paths_in_plot

        parameter_string = ''.join(str(s) + "\t" for s in parameter_string)
        parameter_string += "\n"
        return parameter_string

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
                    V_map[j][i] = h1 > h2  # a one indicates exercising is good
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
        if self.mu(1) == self.r:
            self.other_computation = self.binomial_trees(self.xi, self.r, self.sigma(1), self.T, self.N, self.K)
        else:
            self.other_computation = self.binomial_trees_BS(self.xi, self.r, self.mu(1), self.sigma(1), self.T, self.N / self.T * 10, self.K)
        # self.other_computation *= math.exp(self.r * self.T)
        print("other computation yield: " + str(self.other_computation))
