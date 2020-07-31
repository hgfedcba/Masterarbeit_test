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

        elif string == "am_put1":
            # 2 american puts
            self.internal_neurons = 50
            self.activation1 = torch.tanh
            self.activation2 = torch.sigmoid
            self.optimizer = optim.Adam

            self.validation_frequency = 2
            self.antithetic_variables = True  # only in validation!
            self.pretrain = True  # TODO: USE
            self.pretrain_func = -1  # TODO:USE

            self.stop_paths_in_plot = True  # TODO:use

            self.max_number_iterations = 51
            self.max_minutes_for_iteration = 5
            self.batch_size = 16
            self.val_size = 32
            self.final_val_size = 128
            self.T = 10
            self.N = 10
            self.xi = 40
            self.lr = 0.0001  # lernrate
            self.lr_sheduler_breakpoints = [100, 1000, 10000, 100000]
            self.random_seed = 23343

            def generate_partition(T):
                out = np.zeros(self.N + 1)

                for n in range(self.N):
                    out[n + 1] = (n + 1) * T / self.N
                    assert out[n] != out[n + 1]

                return out

            self.time_partition = generate_partition(self.T)  # 0=t_0<...<t_N=T

            self.d = 1  # dimension
            # TODO: diskontieren?!
            self.r = 0.05  # interest rate
            self.K = 40  # strike price
            self.delta = 0  # dividend rate

            self.sigma_constant = 0.15

            self.sigma = self.sigma_c_x

            """
            def mu(x):
                out = (r - delta) * x
                return out
            """

            self.mu_constant = (self.r - self.sigma_constant ** 2 / 2)

            self.mu = self.mu_c_x
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

            self.g = self.american_put

            self.compute_other_value()

            assert self.sigma_c_x(1) > 0
            assert self.r >= 0

    # TODO:better
    def sigma_c_x(self, x):
        if type(x).__module__ != np.__name__:
            out = self.sigma_constant * x
        else:
            out = self.sigma_constant * np.identity(x.shape[0]) @ x
        return out

    def mu_c_x(self, x):
        out = self.mu_constant * x
        return out

    def american_put(self, t_in, x):
        # TODO: in-place operation?
        sum = 0
        for j in range(self.d):
            sum += max(self.K - x[j].item(), 0)
            # sum += c - x[j]
        # return torch.exp(-r * t) * sum
        return math.exp(-self.r * t_in) * sum

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

        activation_func_dict = {
            torch.tanh                 : "torch.tanh",
            torch.sigmoid              : "torch.sigmoid",
            torch.nn.ELU               : "torch.nn.ELU",
            torch.nn.Hardshrink        : "torch.nn.Hardshrink",
            torch.nn.Hardsigmoid       : "torch.nn.Hardsigmoid",
            torch.nn.Hardtanh          : "torch.nn.Hardtanh",
            # torch.nn.Hardswish         : "torch.nn.Hardswish",
            # torch.nn.functional.hardswish: "torch.nn.functional.hardswish",
            torch.nn.LeakyReLU         : "torch.nn.LeakyReLU",
            torch.nn.LogSigmoid        : "torch.nn.LogSigmoid",
            torch.nn.MultiheadAttention: "torch.nn.MultiheadAttention",
            torch.nn.PReLU             : "torch.nn.PReLU",
            torch.nn.ReLU              : "torch.nn.ReLU",
            torch.nn.ReLU6             : "torch.nn.ReLU6",
            torch.nn.SELU              : "torch.nn.SELU",
            torch.nn.CELU              : "torch.nn.CELU",
            torch.nn.GELU              : "torch.nn.GELU",
            torch.nn.Sigmoid           : "torch.nn.Sigmoid",
            torch.nn.Softplus          : "torch.nn.Softplus",
            torch.nn.Softshrink        : "torch.nn.Softshrink",
            torch.nn.Softsign          : "torch.nn.Softsign",
            torch.nn.Tanh              : "torch.nn.Tanh",
            torch.nn.Tanhshrink        : "torch.nn.Tanhshrink",
            torch.nn.Threshold         : "torch.nn.Threshold"
        }

        optimizer_dict = {
            optim.Adam: "Adam"
        }

        pretrain_func_dict = {
            self.pretrain_func: "TODO"  # TODO
        }

        lr_decay_dict = {
            "hi": "hi"  # TODO
        }
        parameter_string = "d: ", self.d, " T: ", self.T, " N: ", self.N, " xi: ", self.xi, " r: ", self.r, " K: ", self.K, " delta: ", self.delta, " sigma(x): ", sigma_dict.get(
            self.sigma), " mu(x): ", mu_dict.get(self.mu), " g: ", g_dict.get(self.g), " internal_neurons: ", self.internal_neurons, " internal_activation_func: ", activation_func_dict.get(
            self.activation1), " final activation func: ", activation_func_dict.get(self.activation2), " optimizer: ", optimizer_dict.get(
            self.optimizer), " pretrain?: ", self.pretrain, " pretrain_func: ", pretrain_func_dict.get(
            self.pretrain_func), " max_iterations: ", self.max_number_iterations, " max_duration(min): ", self.max_minutes_for_iteration, " batch_size: ", self.batch_size, " val_size: ", self.val_size, " final_val_size: ", self.final_val_size, " initial_lernrate: ", self.lr, " type_of_lr_decay", lr_decay_dict.get(
            "hi"), " validation_frequency: ", self.validation_frequency, " antithetic_variables: ", self.antithetic_variables, " stop_paths_in_plot: ", self.stop_paths_in_plot

        parameter_string = ''.join(str(s) + "\t" for s in parameter_string)
        parameter_string += "\n"
        return parameter_string

    """
    import math
    import numpy as np
    import scipy.stats
    import matplotlib.pyplot as plt


    # computes the price of american and european puts in the CRR model
    def CRR_AmEuPut(S_0, r, sigma, T, M, EU, K):
        # compute values of u, d and q
        delta_t = T / M
        alpha = math.exp(r * delta_t)
        beta = 1 / 2 * (1 / alpha + alpha * math.exp(math.pow(sigma, 2) * delta_t))
        u = beta + math.sqrt(math.pow(beta, 2) - 1)
        d = 1 / u
        q = (math.exp(r * delta_t) - d) / (u - d)

        # allocate matrix S
        S = np.zeros((M + 1, M + 1))

        # fill matrix S with stock prices
        # going down in the matrix means that the stock price goes up
        for i in range(1, M + 2, 1):
            for j in range(1, i + 1, 1):
                S[j - 1, i - 1] = S_0 * math.pow(u, j - 1) * math.pow(d, i - j)

        # payoff for put option with strike K
        def g(S_vector):
            return np.maximum(K - S_vector, 0)

        # allocate memory for option prices
        V = np.zeros((M + 1, M + 1))

        # option price at maturity, formula (1.16) in lecture notes
        V[:, M] = g(S[:, M])

        # Loop goes backwards through the columns
        for i in range(M - 1, -1, -1):
            # backwards recursion for European option, formula (1.14)
            V[0:(i + 1), i] = np.exp(-r * delta_t) * (q * V[1:(i + 2), i + 1] + (1 - q) * V[0:(i + 1), i + 1])

            # only for american options we compare 'exercising the option', i.e. immediate payoff,
            # with 'waiting', i.e. the expected (discounted) value of the next timestep which corresponds to the price
            # of a european option, formula (1.15)
            if EU == 0:
                V[0:(i + 1), i] = np.maximum(g(S[0:(i + 1), i]), V[0:(i + 1), i])

        # first entry of the matrix is the initial option price
        return V[0, 0]

    # test parameters
    S_0 = self.xi
    r = self.r
    sigma = self.sigma(1)
    T = self.T
    M = np.ones(1) * self.N * 10
    EU = 0
    K = self.K
    t = 0


    # BS-formula
    def EuPut_BlackScholes(t, S_t, r, sigma, T, K):
        d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
        d_2 = d_1 - sigma * math.sqrt(T - t)
        Put = K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(-d_2) - S_t * scipy.stats.norm.cdf(-d_1)
        return Put

    uoesb = CRR_AmEuPut(S_0, r, sigma, T, 100, EU, K)
    """
    """
    # calculate prices in CRR-model
    V0 = np.zeros(len(M))
    for i in range(0, len(M)):
        V0[i] = CRR_AmEuPut(S_0, r, sigma, T, M[i], EU, K)

    # calculate price with BS-formula
    BS_price = EuPut_BlackScholes(t, S_0, r, sigma, T, K) * np.ones(len(M))

    # plot comparison and do not forget to label everything
    plt.plot(M, V0, 'b-', label='Price in the binomial model')
    plt.plot(M, BS_price, 'red', label='Price with BS-Formula')
    plt.title('European Put option prices in the binomial model')
    plt.xlabel('number of steps')
    plt.ylabel('Option price')
    plt.legend()

    plt.show()
    """
    # C-Exercise 23, SS 2020
    """
    import numpy as np
    import math
    import matplotlib.pyplot as plt


    def SimPaths_Ito_Euler(X0, a, b, T, m, N):
        Delta_t = T / m
        Delta_W = np.random.normal(0, math.sqrt(Delta_t), (N, m))

        # Initialize matrix which contains the process values
        X = np.zeros((N, m + 1))

        # Assign first column starting values
        X[:, 0] = X0 * np.ones(N)

        # Recursive column-wise simulation according to the algorithms in Section ?.? using one vector of Brownian motion increments
        for i in range(0, m):
            X[:, i + 1] = X[:, i] + a(X[:, i], i * Delta_t) * Delta_t + b(X[:, i], i * Delta_t) * Delta_W[:, i]

        return X

    # Testing Parameters
    N = 10
    m = 10

    nu0 = math.pow(0.3, 2)
    kappa = math.pow(0.3, 2)
    lmbda = 2.5
    sigma_tilde = 0.2
    T = 10

    # Function for the drift of the variance process in the heston model (no time dependence)
    def a(x, t):
        return kappa - lmbda * x

    def a(x, t):
        return self.mu(1)

    # Function for the standard deviation of the variance process in the heston model (no time dependence)
    def b(x, t):
        return np.sqrt(x) * sigma_tilde

    def b(x, t):
        return self.sigma(1)

    X = SimPaths_Ito_Euler(self.xi, a, b, T, m, N)

    plt.clf()
    t = np.linspace(0, T, m + 1)
    # plot the first paths in the matrix
    for i in range(0, 10):
        plt.plot(t, X[i, :])
    plt.xlabel('Time')
    plt.ylabel('Process Value')
    plt.show()
    """
    # C-Exercise 31
    """
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    # from C_Exercise_07 import CRR_AmEuPut


    def BS_AmPut_FiDi_Explicit(r, sigma, a, b, m, nu_max, T, K):
        return BS_AmPut_FiDi_General(r, sigma, a, b, m, nu_max, T, K, 0)


    # This is the Code from p.82-83
    def BS_AmPut_FiDi_General(r, sigma, a, b, m, nu_max, T, K, theta):
        # Compute delta_x, delta_t, x, lambda and set starting w
        q = (2 * r) / (sigma * sigma)
        delta_x = (b - a) / m
        delta_t = (sigma * sigma * T) / (2 * nu_max)
        lmbda = delta_t / (delta_x * delta_x)
        x = np.ones(m + 1) * a + np.arange(0, m + 1) * delta_x
        t = delta_t * np.arange(0, nu_max + 1)
        g_nu = np.maximum(np.exp(x * 0.5 * (q - 1)) - np.exp(x * 0.5 * (q + 1)), np.zeros(m + 1))
        w = g_nu[1:m]

        # Building matrix for t-loop
        lambda_theta = lmbda * theta
        diagonal = np.ones(m - 1) * (1 + 2 * lambda_theta)
        secondary_diagonal = np.ones(m - 2) * (- lambda_theta)
        b = np.zeros(m - 1)

        # t-loop as on p.83.
        for nu in range(0, nu_max):
            g_nuPlusOne = math.exp((q + 1) * (q + 1) * t[nu + 1] / 4.0) * np.maximum(np.exp(x * 0.5 * (q - 1))
                                                                                     - np.exp(x * 0.5 * (q + 1)),
                                                                                     np.zeros(m + 1))
            b[0] = w[0] + lmbda * (1 - theta) * (w[1] - 2 * w[0] + g_nu[0]) + lambda_theta * g_nuPlusOne[0]
            b[1:m - 2] = w[1:m - 2] + lmbda * (1 - theta) * (w[2:m - 1] - 2 * w[1:m - 2] + w[0:m - 3])
            b[m - 2] = w[m - 2] + lmbda * (1 - theta) * (g_nu[m] - 2 * w[m - 2] + w[m - 3]) + lambda_theta * g_nuPlusOne[m]

            # Use Brennan-Schwartz algorithm to solve the linear equation system
            w = solve_system_put(diagonal, secondary_diagonal, secondary_diagonal, b, g_nuPlusOne[1:m])

            g_nu = g_nuPlusOne

        S = K * np.exp(x[1:m])
        v = K * w * np.exp(- 0.5 * x[1:m] * (q - 1) - 0.5 * sigma * sigma * T * ((q - 1) * (q - 1) / 4 + q))

        return S, v


    # This is the code from Lemma 5.3
    def solve_system_put(alpha, beta, gamma, b, g):
        n = len(alpha)
        alpha_hat = np.zeros(n)
        b_hat = np.zeros(n)
        x = np.zeros(n)

        alpha_hat[n - 1] = alpha[n - 1]
        b_hat[n - 1] = b[n - 1]
        for i in range(n - 2, -1, -1):
            alpha_hat[i] = alpha[i] - beta[i] / alpha_hat[i + 1] * gamma[i]
            b_hat[i] = b[i] - beta[i] / alpha_hat[i + 1] * b_hat[i + 1]
        x[0] = np.maximum(b_hat[0] / alpha_hat[0], g[0])
        for i in range(1, n):
            x[i] = np.maximum((b_hat[i] - gamma[i - 1] * x[i - 1]) / alpha_hat[i], g[i])
        return x


    # Initialize the test parameters given in the exercise.
    r = self.r
    sigma = self.sigma(1)
    # a = - 0.7
    # b = 0.4
    # nu_max = 2000
    a = -0.5
    b = 0.5
    nu_max = 100
    m = self.N
    T = self.T
    K = self.K

    initial_stock, option_prices = BS_AmPut_FiDi_Explicit(r, sigma, a, b, m, nu_max,
                                                          T, K)
    exercise7 = np.zeros(len(initial_stock))
    for j in range(0, len(initial_stock)):
        exercise7[j] = CRR_AmEuPut(initial_stock[j], r, sigma, T, 500, 0, K)

    # Compute the absolute difference between the approximation and the option prices from exercise 7
    absolute_errors = np.abs(option_prices - exercise7)

    print(option_prices)

    # Compare the results by plotting the absolute error.
    plt.plot(initial_stock, absolute_errors)
    plt.plot(initial_stock, option_prices)
    plt.plot(initial_stock, exercise7)
    plt.xlabel('initial stock price')
    plt.ylabel('absolute difference')
    plt.title('The absolute difference between the finite difference approximation with the explicit scheme and the'
              ' option prices from exercise 7')
    plt.show()
    """
    """
    # C-Exercise 24, SS 2020
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    def Sim_Paths_GeoBM(X0, mu, sigma, T, N):
        Delta_t = T / N
        Delta_W = np.random.normal(0, math.sqrt(Delta_t), (N, 1))

        # Initialize vectors with starting value
        X_exact = X0 * np.ones(N + 1)
        X_Euler = X0 * np.ones(N + 1)
        X_Milshtein = X0 * np.ones(N + 1)

        # Recursive simulation according to the algorithms in Section ?.? using identical Delta_W
        for i in range(0, N):
            X_exact[i + 1] = X_exact[i] * np.exp((mu - math.pow(sigma, 2) / 2) * Delta_t + sigma * Delta_W[i])
            X_Euler[i + 1] = X_Euler[i] * (1 + mu * Delta_t + sigma * Delta_W[i])
            X_Milshtein[i + 1] = X_Milshtein[i] * (1 + mu * Delta_t + sigma * Delta_W[i] + math.pow(sigma, 2) / 2 * (math.pow((Delta_W[i]), 2) - Delta_t))

        return X_exact, X_Euler, X_Milshtein

    # test parameters
    X0 = 40
    sigma = 0.25
    mu = 0.05 - math.pow(sigma, 2) / 2
    T = 10
    N = np.array([10, 100, 1000, 10000])

    # plot
    plt.clf()
    for i in range(0, 4):
        X_exact, X_Euler, X_Milshtein = Sim_Paths_GeoBM(X0, mu, sigma, T, N[i])
        plt.subplot(2, 2, i + 1)
        plt.plot(np.arange(0, N[i] + 1) * T / N[i], X_exact, label='Exact Simulation')
        plt.plot(np.arange(0, N[i] + 1) * T / N[i], X_Euler, 'red', label='Euler approximation')
        plt.plot(np.arange(0, N[i] + 1) * T / N[i], X_Milshtein, 'green', label='Milshtein approximation')
        plt.xlabel('t')
        plt.ylabel('X(t)')
        plt.title('N=' + str(N[i]))
        plt.legend()

    plt.show()
    """


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
        if self.mu(1) == self.r - self.sigma(1) ** 2 / 2:
            self.other_computation = self.binomial_trees(self.xi, self.r, self.sigma(1), self.T, self.N * 10, self.K)
        else:
            self.other_computation = self.binomial_trees_BS(self.xi, self.r, self.mu(1), self.sigma(1), self.T, self.N / self.T * 10, self.K)
        print("other computation yield: " + str(self.other_computation))
