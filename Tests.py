import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import scipy
import math
from scipy import stats
import time


class Tests:
    def __init__(self, out, Model):
        self.out = out
        self.Model = Model

    def test_good(self):
        import torch
        from torch import nn
        from torch.autograd import Variable
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import torch.optim as optim


        def f_x(x):
            return (torch.relu(40 - x) / 8) ** 2
            # return x * x / 64 - 5 * x / 4 + 25


        x_values = np.ones((21, 1))
        for i in range(0, 21):
            x_values[i] = i + 30  # True


        class Net(nn.Module):
            def __init__(self, d=1):
                super(Net, self).__init__()
                # an affine operation: y = Wx + b
                internal_neurons = 50
                self.fc1 = nn.Linear(d, internal_neurons)
                self.fc2 = nn.Linear(internal_neurons, internal_neurons)
                self.fc3 = nn.Linear(internal_neurons, 1)

            def forward(self, y):
                y = torch.tanh(self.fc1(y))
                y = torch.tanh(self.fc2(y))
                y = torch.sigmoid(self.fc3(y))
                # y = self.fc3(y)
                return y


        # net = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 1))
        net = Net()

        optimizer = optim.Adam(net.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        epochs = 800


        def out(k):
            a = 30
            b = 50

            import matplotlib.backends.backend_pdf as pdfp
            from pylab import plot, show, grid, xlabel, ylabel
            import matplotlib.pyplot as plt
            # pdf = pdfp.PdfPages("graph" + str(k) + ".pdf")

            t = np.linspace(a, b, 20)
            x = np.zeros(t.shape[0])
            c_fig = plt.figure()

            for j in range(len(t)):
                h = torch.tensor(np.ones(1) * t[j], dtype=torch.float32)
                x[j] = net(h)
            plt.ylim([0, 1])
            plot(t, x, linewidth=4)
            xlabel('x', fontsize=16)
            ylabel('net(x)', fontsize=16)
            grid(True)
            show()
            # pdf.savefig(c_fig)

            # pdf.close()
            plt.close(c_fig)


        def train():
            net.train()
            losses = []
            for epoch in range(1, epochs):
                x_train = Variable(torch.from_numpy(x_values)).float()
                y_train = f_x(x_train)
                y_pred = net(x_train)
                loss = ((y_pred - y_train) ** 2).sum()
                print("epoch #", epoch)
                print(loss.item())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return losses


        print("training start....")
        losses = train()
        plt.plot(range(1, epochs), losses)
        plt.xlabel("epoch")
        plt.ylabel("loss train")
        plt.ylim([0, 100])
        plt.show()

        out(epochs)

    def test(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # an affine operation: y = Wx + b
                self.fc1 = nn.Linear(100, 100)  # 6*6 from image dimension
                self.fc2 = nn.Linear(100, 100)
                self.fc3 = nn.Linear(100, 50)
                self.fc4 = nn.Linear(50, 1)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = self.fc4(x)
                return x

        net = Net()
        """
        params = list(net.parameters())
        print(len(params))
        for k in range(8):
            print(params[k].size())
        """
        """
        # create your optimizer
        optimizer = optim.SGD(net.parameters(), lr=0.01)
    
        optimizer.zero_grad()  # zero the gradient buffers
    
        for k in range(5):
            input = torch.randn(100)
    
            # in your training loop:
            # optimizer.zero_grad()
            output = net(input)
            loss = torch.norm(output - torch.ones(100))
            loss.backward()
            optimizer.step()
    
            print(net.fc1.bias.grad)
        """

        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

        inputs = torch.randn(2, 100) * 30 + 10
        min_loss = 10000000

        for epoch in range(100):  # loop over the dataset multiple times

            running_loss = 0.0
            for i in range(1):
                # get the inputs; data is a list of [inputs, labels]
                input = inputs

                # zero the parameter gradients
                optimizer.zero_grad()

                def g(x, t_in=1):
                    t = torch.ones(1) * t_in
                    sum = torch.zeros(1)
                    c = torch.ones(1) * 150
                    for k in range(2):
                        sum += torch.max((x[k] + k + 1) ** 2, torch.zeros(1))
                    return torch.exp(-1.05 * t) * sum

                # forward + backward + optimize
                outputs = net(input)
                # loss = torch.norm(outputs - torch.ones(100))
                loss = self.rectified_minimum(x=g(outputs), const=torch.ones(1), factor=1 / (epoch + 1))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                """
                if i % 1 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss))
                    running_loss = 0.0
                """
                print(outputs)
                print(running_loss)
                print("\n")
                min_loss = min(min_loss, running_loss)
                running_loss = 0.0

        print('Finished Training')
        print(min_loss)

    def test4(self):
        import torch
        from torch import nn
        from torch.autograd import Variable
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import torch.optim as optim


        def f_x(x):
            return x * x / 64 - 5 * x / 4 + 25
            return 2 * x * x + 3 * x  # random function to learn


        # Building dataset
        def build_dataset():
            # Given f(x), is_f_x defines whether the function is satisfied
            data = []
            for i in range(1, 100):
                data.append((i, f_x(i), 1))  # True
            for j in range(100, 201):
                data.append((j, f_x(j) + j * j, 0))  # Not true
            column_names = ["x", "f_x", "is_f_x"]
            df = pd.DataFrame(data, columns=column_names)
            return df


        df = build_dataset()
        print("Dataset is built!")

        labels = df.is_f_x.values
        features = df.drop(columns=['is_f_x']).values

        print("shape of features:", features.shape)
        print("shape of labels: ", labels.shape)

        # Building nn
        net = nn.Sequential(nn.Linear(features.shape[1], 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 2))

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, shuffle=True, random_state=34)

        # parameters
        optimizer = optim.Adam(net.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss()
        epochs = 300

        def out(k):
            folder_name = "Testrun1"
            working_directory = pathlib.Path().absolute()
            output_location = working_directory / f'{folder_name}'

            a = min(40, 40 + self.Model.getT() * self.Model.getmu(1))
            b = max(40, 40 + self.Model.getT() * self.Model.getmu(1))
            a = a - 1.5 * self.Model.getT() * self.Model.getsigma(1)
            b = b + 1.5 * self.Model.getT() * self.Model.getsigma(1)

            # TODO: copy graph so i only use a copy when it was still open

            import matplotlib.backends.backend_pdf as pdfp
            from pylab import plot, show, grid, xlabel, ylabel
            import matplotlib.pyplot as plt
            pdf = pdfp.PdfPages("graph" + str(k) + ".pdf")

            t = np.linspace(a, b, 20)
            x = np.zeros(t.shape[0])
            c_fig = plt.figure()

            for j in range(len(t)):
                h = torch.tensor(np.ones(1) * t[j], dtype=torch.float32)
                help = net(h)
                x[j] = net(h)
            help = x
            plt.ylim([0, 1])
            plot(t, x, linewidth=4)
            xlabel('x', fontsize=16)
            ylabel('net(x)', fontsize=16)
            grid(True)
            pdf.savefig(c_fig)

            pdf.close()
            plt.close(c_fig)


        def train():
            net.train()
            losses = []
            for epoch in range(1, 200):
                x_train = Variable(torch.from_numpy(features_train)).float()
                y_train = Variable(torch.from_numpy(labels_train)).long()
                hx = x_train[11]
                hy = y_train[11]
                y_pred = net(x_train)
                hypred = net(hx)
                if epoch == 190:
                    assert True
                loss = criterion(y_pred, y_train)
                print("epoch #", epoch)
                print(loss.item())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return losses


        print("training start....")
        losses = train()
        plt.plot(range(1, 200), losses)
        plt.xlabel("epoch")
        plt.ylabel("loss train")
        plt.ylim([0, 100])
        plt.show()

        print("testing start ... ")
        x_test = Variable(torch.from_numpy(features_test)).float()
        x_train = Variable(torch.from_numpy(features_train)).float()


        def test():
            pred = net(x_test)
            pred = torch.max(pred, 1)[1]
            print("Accuracy on test set: ", accuracy_score(labels_test, pred.data.numpy()))

            p_train = net(x_train)
            p_train = torch.max(p_train, 1)[1]
            print("Accuracy on train set: ", accuracy_score(labels_train, p_train.data.numpy()))


        test()

        # out(epochs)


    def test5(self):
        import torch
        from torch import nn
        from torch.autograd import Variable
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import torch.optim as optim


        def f_x(x):
            return x * x / 64 - 5 * x / 4 + 25


        # Building dataset
        def build_dataset():
            # Given f(x), is_f_x defines whether the function is satisfied
            x_values = np.ones((21, 1))
            for i in range(0, 21):
                x_values[i] = i + 30  # True
            return x_values

        x_values = build_dataset()

        # Building nn
        net = nn.Sequential(nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1))

        # parameters
        optimizer = optim.Adam(net.parameters(), lr=0.00001)
        epochs = 200

        def out(k):
            # folder_name = "Testrun1"
            # working_directory = pathlib.Path().absolute()
            # output_location = working_directory / f'{folder_name}'

            a = min(40, 40 + self.Model.getT() * self.Model.getmu(1))
            b = max(40, 40 + self.Model.getT() * self.Model.getmu(1))
            a = a - 1.5 * self.Model.getT() * self.Model.getsigma(1)
            b = b + 1.5 * self.Model.getT() * self.Model.getsigma(1)

            # TODO: copy graph so i only use a copy when it was still open

            import matplotlib.backends.backend_pdf as pdfp
            from pylab import plot, show, grid, xlabel, ylabel
            import matplotlib.pyplot as plt
            # pdf = pdfp.PdfPages("graph" + str(k) + ".pdf")

            t = np.linspace(a, b, 20)
            x = np.zeros(t.shape[0])
            c_fig = plt.figure()

            for j in range(len(t)):
                h = torch.tensor(np.ones(1) * t[j], dtype=torch.float32)
                x[j] = net(h)
            plt.ylim([0, 1])
            plot(t, x, linewidth=4)
            xlabel('x', fontsize=16)
            ylabel('net(x)', fontsize=16)
            grid(True)
            show()
            # pdf.savefig(c_fig)

            # pdf.close()
            plt.close(c_fig)


        def train():
            net.train()
            losses = []
            for epoch in range(1, epochs):
                x_train = Variable(torch.from_numpy(x_values)).float()
                y_train = f_x(x_train)
                y_pred = net(x_train)
                loss = torch.sum(torch.abs(y_pred - y_train))
                print("epoch #", epoch)
                print(loss.item())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return losses


        print("training start....")
        losses = train()
        plt.plot(range(1, epochs), losses)
        plt.xlabel("epoch")
        plt.ylabel("loss train")
        plt.ylim([0, 100])
        plt.show()

        out(epochs)

    def test6(self):
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

    def test7(self):
        # C-Exercise 23, SS 2020

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

    def test8(self):
        # C-Exercise 31

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

    def test9(self):
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

    def test10(self):
        def Sim_Paths_GeoBM(self, X0, mu, sigma, T, N):
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
                # X_Euler[i + 1] = X_Euler[i] + mu * Delta_t + sigma * Delta_W[i]
                X_Milshtein[i + 1] = X_Milshtein[i] * (1 + mu * Delta_t + sigma * Delta_W[i] + math.pow(sigma, 2) / 2 * (math.pow((Delta_W[i]), 2) - Delta_t))

            X_Euler = np.reshape(X_Euler, (1, 11))

            return X_Euler
            return X_exact, X_Euler, X_Milshtein



class Tutorial:
    def __init__(self):

        x = torch.rand(5, 3)

        # let us run this cell only if CUDA is available
        # We will use ``torch.device`` objects to move tensors in and out of GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")  # a CUDA device object
            y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
            x = x.to(device)  # or just use strings ``.to("cuda")``
            z = x + y
            print(z)
            print(z.to("cpu", torch.double))


        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                # 1 input image channel, 6 output channels, 3x3 square convolution
                # kernel
                self.conv1 = nn.Conv2d(1, 6, 3)
                self.conv2 = nn.Conv2d(6, 16, 3)
                # an affine operation: y = Wx + b
                self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                # Max pooling over a (2, 2) window
                x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
                # If the size is a square you can only specify a single number
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = x.view(-1, self.num_flat_features(x))
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

            def num_flat_features(self, x):
                size = x.size()[1:]  # all dimensions except the batch dimension
                num_features = 1
                for s in size:
                    num_features *= s
                return num_features


        net = Net()
        print(net)

        params = list(net.parameters())
        print(len(params))
        print(params[0].size())  # conv1's .weight

        input = torch.randn(1, 1, 32, 32)
        out = net(input)
        print(out)

        input = torch.randn(1, 1, 32, 32)
        out = net(input)
        print(out)
