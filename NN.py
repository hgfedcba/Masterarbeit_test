import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import scipy
import math
from scipy import stats
import time
import pytest
import Out


class Net(nn.Module):
    def __init__(self, d, internal_neurons, activation1, activation2):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(d, internal_neurons)
        self.fc2 = nn.Linear(internal_neurons, internal_neurons)
        self.fc3 = nn.Linear(internal_neurons, 1)

        self.activation1 = activation1
        self.activation2 = activation2

    def forward(self, y):
        """
        y = torch.tanh(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = torch.sigmoid(self.fc3(y))
        """
        y = self.activation1(self.fc1(y))
        y = self.activation1(self.fc2(y))
        y = self.activation2(self.fc3(y))
        # y = self.fc3(y)
        return y


class NN:
    def __init__(self, config, Model, log, out):

        self.log = log
        self.lr = config.lr  # Lernrate
        self.lr_sheduler_breakpoints = config.lr_sheduler_breakpoints
        self.nu = config.N * (2 * config.d + 1) * (config.d + 1)
        self.N = config.N
        self.d = config.d
        self.u = []
        self.Model = Model
        self.t = self.generate_partition(self.Model.getT())  # 0=t_0<...<t_N=T
        self.net_net_duration = []

        self.internal_neurons = config.internal_neurons
        self.activation1 = config.activation1
        self.activation2 = config.activation2
        self.optimizer = config.optimizer

        self.validation_frequency = config.validation_frequency
        self.antithetic_variables = config.antithetic_variables

        self.out = out

        # TODO: use this
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    def define_nets(self):
        self.u = []
        for n in range(self.N):
            # activation_function = nn.SELU()
            # net = nn.Sequential(nn.Linear(1, 300), activation_function, nn.Linear(300, 300), activation_function, nn.Linear(300, 300), activation_function, nn.Linear(300, 1), torch.sigmoid())
            net = Net(self.d, self.internal_neurons, self.activation1, self.activation2)
            self.u.append(net)

    def optimization(self, M, J, L):
        log = self.log
        np.random.seed(1337)
        torch.manual_seed(1337)

        self.define_nets()

        train_individual_payoffs = []
        train_average_payoff = []

        params = []
        for k in range(len(self.u)):
            params += list(self.u[k].parameters())
        optimizer = self.optimizer(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sheduler_breakpoints, gamma=self.lr)

        val_bm_list = []
        val_path_list = []
        val_continuous_value_list = []
        val_discrete_value_list = []
        val_individual_payoffs = []
        val_duration = []
        actual_stopping_time = []

        for l in range(L):
            if not self.antithetic_variables or l < L / 2:
                val_bm_list.append(self.generate_bm())
            elif l == L / 2:
                val_bm_list.extend([-item for item in val_bm_list])
            val_path_list.append(self.generate_path(val_bm_list[l]))

        pretrain_start = time.time()
        self.pretrain()
        log.info("pretrain took \t%s" % (time.time() - pretrain_start))

        train_duration = []

        for m in range(M):
            self.net_net_duration.append(0)
            m_th_iteration_time = time.time()

            train_duration.append(time.time())
            self.train(optimizer, train_individual_payoffs, train_average_payoff, J, m)
            train_duration[m] = time.time() - train_duration[m]

            # validation
            if m % self.validation_frequency == 0:
                h = self.validate(L, val_individual_payoffs, val_duration, val_path_list, val_continuous_value_list, val_discrete_value_list)
                actual_stopping_time.append(h)
                log.info(
                    "After \t%s iterations the continuous value is\t %s and the discrete value is \t%s" % (m, round(val_continuous_value_list[-1].item(), 3), round(val_discrete_value_list[-1], 3)))

            scheduler.step()  # TODO:verify

            if m == 25:
                assert True

        return train_individual_payoffs, train_average_payoff, val_continuous_value_list, val_discrete_value_list, val_path_list, actual_stopping_time, train_duration, val_duration, self.net_net_duration

    def pretrain(self):
        from torch.autograd import Variable
        import matplotlib.pyplot as plt
        for m in range(len(self.u)):
            def f_x(x):
                return (torch.relu(39 - x) / 8) ** 2

            x_values = np.ones((21, 1))
            for i in range(0, 21):
                x_values[i] = i + 30  # True

            net = self.u[m]

            optimizer = optim.Adam(net.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
            epochs = 800

            def out(k):
                a = 30
                b = 50

                import matplotlib.backends.backend_pdf as pdfp
                from pylab import plot, show, grid, xlabel, ylabel
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
                    # print("epoch #", epoch)
                    # print(loss.item())
                    losses.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if losses[-1] < 1:
                        epoch = epochs
                return losses


            # print("training start....")
            losses = train()
            """
            plt.plot(range(1, epochs), losses)
            plt.xlabel("epoch")
            plt.ylabel("loss train")
            plt.ylim([0, 100])
            plt.show()
            plt.close()

            self.out.draw_function(self.u[m])
            """

    def train(self, optimizer, individual_payoffs, average_payoff, J, m):
        bm_list = []
        training_path_list = []
        individual_payoffs.append([])
        U = torch.empty(J, self.N + 1)

        for j in range(J):
            bm_list.append(self.generate_bm())
            training_path_list.append(self.generate_path(bm_list[j]))

            # U[j, :] = self.generate_stopping_time_factors_from_path(training_path_list[j])
            h = self.generate_stopping_time_factors_from_path(training_path_list[j])
            U[j, :] = h[:, 0]
            individual_payoffs[m].append(self.calculate_payoffs(U[j, :], training_path_list[j], self.Model.getg, self.t))
        mean = torch.sum(torch.stack(individual_payoffs[m])) / len(individual_payoffs[m])
        average_payoff.append(mean)

        loss = -average_payoff[m]
        # loss = average_payoff[m] # wrong!!!
        optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        # TODO: deactivate requires grad
        # loss = torch.norm(U)
        # loss.backward()
        t = time.time()
        optimizer.step()
        self.net_net_duration[-1] += time.time() - t

    def validate(self, L, val_individual_payoffs, val_duration, val_path_list, val_continuos_value_list, val_discrete_value_list):
        val_individual_payoffs.append([])
        U = torch.empty(L, self.N + 1)
        val_duration.append(time.time())

        local_list = []
        actual_stopping_times = []
        tau_list = []
        for l in range(L):
            h = self.generate_stopping_time_factors_from_path(val_path_list[l])
            U[l, :] = h[:, 0]
            val_individual_payoffs[-1].append(self.calculate_payoffs(U[l, :], val_path_list[l], self.Model.getg, self.t))

            for_debugging1 = val_path_list[l]
            for_debugging2 = h[:, 0]
            for_debugging3 = val_individual_payoffs[-1][l]

            # part 2: discrete
            tau_set = np.zeros(self.N + 1)
            for n in range(tau_set.size):
                h1 = torch.sum(U[l, 0:n + 1]).item()
                h2 = 1 - U[l, n].item()
                h3 = sum(U[l, 0:n + 1]) >= 1 - U[l, n]
                tau_set[n] = torch.sum(U[l, 0:n + 1]).item() >= 1 - U[l, n].item()
            tau_list.append(np.argmax(tau_set))  # argmax returns the first "True" entry
            for_debugging4 = tau_list[l]

            actual_stopping_time = np.zeros(self.N + 1)
            actual_stopping_time[tau_list[l]] = 1
            local_list.append(self.calculate_payoffs(actual_stopping_time, val_path_list[l], self.Model.getg, self.t).item())
            actual_stopping_times.append(actual_stopping_time)

        val_discrete_value_list.append(sum(local_list) / L)
        val_continuos_value_list.append(torch.sum(torch.stack(val_individual_payoffs[-1])) / len(val_individual_payoffs[-1]))
        val_duration[-1] = time.time() - val_duration[-1]

        return actual_stopping_times

    def generate_partition(self, T):
        out = np.zeros(self.N + 1)

        for n in range(self.N):
            out[n + 1] = (n + 1) * T / self.N
            assert out[n] != out[n + 1]

        return out

    def generate_bm(self):
        # Ein Rückgabewert ist ein np.array der entsprechenden Länge, in dem die Werte über den gesamten sample path eingetragen sind
        out = np.zeros((self.Model.getd(), self.N + 1))
        for m in range(self.Model.getd()):
            for n in range(self.N):
                # TODO:verify square root
                out[m, n + 1] = scipy.stats.norm.rvs(loc=out[m, n], scale=(self.t[n + 1] - self.t[n]) ** 0.5)

        return out

    def generate_path(self, bm):
        out = np.zeros((self.Model.getd(), self.N + 1))
        out[:, 0] = self.Model.getxi()
        for n in range(self.N):
            h = out[:, n]
            part2 = self.Model.getmu(out[:, n]) * (self.t[n + 1] - self.t[n])
            part3 = self.Model.getsigma(out[:, n]) @ (bm[:, n + 1] - bm[:, n])
            out[:, n + 1] = out[:, n] + part2 + part3
            # out[:, n + 1] = out[:, n] * (1 + part2 + part3)  # TODO: hä

        return out

    def generate_stopping_time_factors_from_path(self, x_input):
        local_N = x_input.shape[1]
        U = []
        sum = []
        x = []
        # x = torch.from_numpy(x_input) doesn't work for some reason

        h = []

        for n in range(local_N):
            if n > 0:
                sum.append(sum[n - 1] + U[n - 1])  # 0...n-1
            else:
                sum.append(0)
                # drop requires_grad?
            x.append(torch.tensor(x_input[:, n], dtype=torch.float32, requires_grad=True))
            if n < self.N:
                t = time.time()
                h.append(self.u[n](x[n]))
                self.net_net_duration[-1] += time.time() - t
            else:
                h.append(1)
            # max = torch.max(torch.tensor([h1, h2]))
            # U[n] = max * (torch.ones(1) - sum)
            U.append(h[n] * (torch.ones(1) - sum[n]))

        z = torch.stack(U)
        if torch.sum(z).item() != pytest.approx(1, 0.00001):
            whatever = torch.sum(z).item()
            assert True
        assert torch.sum(z).item() == pytest.approx(1, 0.00001), "Value: " + str(torch.sum(z).item())  # TODO: solve this better

        # w = torch.unsqueeze(z, 0)

        return z

    def calculate_payoffs(self, U, x, g, t):
        sum = torch.zeros(1)
        for n in range(self.N + 1):
            sum += U[n] * g(t[n], x[:, n])
        return sum

    def rectified_minimum(self, x, const, factor):
        return torch.min(x, const + x * (factor + 100))

    def rectified_max(self, x, const, factor):
        return torch.max(x, const + x * (factor + 100))
