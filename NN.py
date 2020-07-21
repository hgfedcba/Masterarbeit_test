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


# torch.nn.LeakyReLU
# LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)

# torch.nn.PReLU(num_parameters=1, init=0.25)
# PReLU(x)=max(0,x)+a∗min(0,x)


class Net(nn.Module):
    def __init__(self, d):
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


class NN:
    def __init__(self, config, Model, log):

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


    def define_nets(self):
        self.u = []
        for n in range(self.N):
            # activation_function = nn.SELU()
            # net = nn.Sequential(nn.Linear(1, 300), activation_function, nn.Linear(300, 300), activation_function, nn.Linear(300, 300), activation_function, nn.Linear(300, 1), torch.sigmoid())
            net = Net(self.d)
            self.u.append(net)

    def optimization(self, M, J, L):
        log = self.log
        np.random.seed(1337)
        torch.manual_seed(1337)

        self.define_nets()

        individual_payoffs = []
        average_payoff = []

        params = []
        for k in range(len(self.u)):
            params += list(self.u[k].parameters())
        optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sheduler_breakpoints, gamma=self.lr)

        val_bm_list = []
        val_path_list = []
        val_value_list = []
        val_individual_payoffs = []
        val_duration = []

        for l in range(L):
            val_bm_list.append(self.generate_bm())
            val_path_list.append(self.generate_path(val_bm_list[l]))

        train_duration = []

        for m in range(M):
            self.net_net_duration.append(0)
            m_th_iteration_time = time.time()
            bm_list = []
            training_path_list = []
            individual_payoffs.append([])
            U = torch.empty(J, self.N + 1)

            # zero the parameter gradients
            optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)

            train_duration.append(time.time())
            for j in range(J):
                bm_list.append(self.generate_bm())
                training_path_list.append(self.generate_path(bm_list[j]))
                # self.draw(self.training_path_list[j])

                # U[j, :] = self.generate_stopping_time_factors_from_path(training_path_list[j])
                h = self.generate_stopping_time_factors_from_path(training_path_list[j])
                U[j, :] = h[:, 0]
                individual_payoffs[m].append(self.calculate_payoffs(U[j, :], training_path_list[j], self.Model.getg, self.t))
            mean = torch.sum(torch.stack(individual_payoffs[m])) / len(individual_payoffs[m])
            average_payoff.append(mean)

            loss = -average_payoff[m]
            loss.backward()
            # TODO: deactivate requires grad
            # loss = torch.norm(U)
            # loss.backward()
            t = time.time()
            optimizer.step()
            self.net_net_duration[-1] += time.time() - t
            train_duration[m] = time.time() - train_duration[m]

            # validation
            val_individual_payoffs.append([])
            U = torch.empty(L, self.N + 1)
            val_duration.append(time.time())

            tau_list = []
            for l in range(L):
                h = self.generate_stopping_time_factors_from_path(val_path_list[l])
                U[l, :] = h[:, 0]
                val_individual_payoffs[m].append(self.calculate_payoffs(U[l, :], val_path_list[l], self.Model.getg, self.t))

                for_debugging1 = val_path_list[l]
                for_debugging2 = h[:, 0]
                for_debugging3 = val_individual_payoffs[m][l]

                # part 2
                tau_set = np.zeros(self.N + 1)
                for n in range(tau_set.size):
                    h1 = torch.sum(U[l, 0:n + 1]).item()
                    h2 = 1 - U[l, n].item()
                    h3 = sum(U[l, 0:n]) >= 1 - U[l, n]
                    tau_set[n] = torch.sum(U[l, 0:n + 1]).item() >= 1 - U[l, n].item()
                tau_list.append(np.argmax(tau_set))  # argmax returns the first "True" entry
                for_debugging4 = tau_list[l]

            another_list = []
            for l2 in range(L):
                actual_stopping_time = np.zeros(self.N + 1)
                actual_stopping_time[tau_list[l2]] = 1
                another_list.append(self.calculate_payoffs(actual_stopping_time, val_path_list[l2], self.Model.getg, self.t).item())

            val_value_list.append(torch.sum(torch.stack(val_individual_payoffs[m])) / len(val_individual_payoffs[m]))
            val_duration[m] = time.time() - val_duration[m]
            log.info("After \t%s iterations the continuous value is\t %s and the discrete value is \t%s" % (m, round(val_value_list[m].item(), 3), round(sum(another_list) / L, 3)))

            scheduler.step()  # TODO:verify

            if m == 25:
                assert True

        return individual_payoffs, average_payoff, val_value_list, train_duration, val_duration, self.net_net_duration

    def train(self):
        assert True

    def validate(self):
        assert True

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
                out[m, n + 1] = scipy.stats.norm.rvs(loc=out[m, n], scale=self.t[n + 1] - self.t[n])

        return out

    def generate_path(self, bm):
        out = np.zeros((self.Model.getd(), self.N + 1))
        out[:, 0] = self.Model.getxi()
        for n in range(self.N):
            h = out[:, n]
            part2 = self.Model.getmu(out[:, n]) * (self.t[n + 1] - self.t[n])
            part3 = self.Model.getsigma(out[:, n]) @ (bm[:, n + 1] - bm[:, n])
            out[:, n + 1] = out[:, n] + part2 + part3

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
        assert torch.sum(z).item() == pytest.approx(1, 0.00001)  # TODO: solve this better

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
