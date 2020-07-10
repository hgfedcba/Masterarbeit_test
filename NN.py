import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import scipy
import math
from scipy import stats
import logging as log
import time


class Net(nn.Module):
    def __init__(self, d):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.fc3 = nn.Linear(d, 1)

    def forward(self, y):
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = torch.sigmoid(self.fc3(y))
        return y


class NN:
    def __init__(self, config, Model):
        print("NN init wird aufgerufen")
        self.gamma = config.gamma  # Lernrate
        self.nu = config.N * (2 * config.d + 1) * (config.d + 1)
        self.N = config.N
        self.d = config.d
        self.u = []
        self.Model = Model
        self.gamma = config.gamma
        self.t = self.generate_partition(self.Model.getT())  # 0=t_0<...<t_N=T


    def define_nets(self):
        self.u = []
        for n in range(self.N):
            self.u.append(Net(self.d))

    """
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        inputs = torch.randn(2, 100)

        for epoch in range(100):  # loop over the dataset multiple times

            running_loss = 0.0
            for i in range(1):
                # get the inputs; data is a list of [inputs, labels]
                input = inputs

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(input)
                loss = torch.norm(outputs - torch.ones(100))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                print(running_loss)
                running_loss = 0.0
    """

    def optimization(self, M, J, L):
        np.random.seed(1337)
        torch.manual_seed(1337)

        self.define_nets()

        individual_payoffs = []
        average_payoff = []

        params = []
        for k in range(len(self.u)):
            params += list(self.u[k].parameters())
        optimizer = optim.SGD(params, lr=self.gamma, momentum=0.9)

        val_bm_list = []
        val_path_list = []
        val_value_list = []
        val_individual_payoffs = []

        for l in range(L):
            val_bm_list.append(self.generate_bm())
            val_path_list.append(self.generate_path(val_bm_list[l]))

        path_list = []

        for m in range(M):
            m_th_iteration_time = time.time()
            bm_list = []
            path_list = []
            individual_payoffs.append([])
            U = torch.empty(J, self.N + 1)

            # zero the parameter gradients
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)

            training_time = time.time()
            for j in range(J):
                bm_list.append(self.generate_bm())
                path_list.append(self.generate_path(bm_list[j]))
                # self.draw(self.path_list[j])

                # U[j, :] = self.generate_stopping_time_factors_from_path(path_list[j])
                h = self.generate_stopping_time_factors_from_path(path_list[j])
                U[j, :] = h[:, 0]
                individual_payoffs[m].append(self.calculate_payoffs(U[j, :], path_list[j], self.Model.getg, self.t))
            mean = torch.sum(torch.stack(individual_payoffs[m])) / len(individual_payoffs[m])
            average_payoff.append(mean)

            loss = -average_payoff[m]
            loss.backward()
            # loss = torch.norm(U)
            # loss.backward()
            optimizer.step()
            log.info("time for training is %s seconds" % (time.time() - training_time))

            # validation
            val_individual_payoffs.append([])
            U = torch.empty(L, self.N + 1)
            val_time = time.time()
            for l in range(L):
                h = self.generate_stopping_time_factors_from_path(val_path_list[l])
                U[l, :] = h[:, 0]
                val_individual_payoffs[m].append(self.calculate_payoffs(U[l, :], val_path_list[l], self.Model.getg, self.t))
            val_value_list.append(torch.sum(torch.stack(val_individual_payoffs[m])) / len(val_individual_payoffs[m]))
            # update weights
            log.info("time for validation is %s seconds" % (time.time() - val_time))
            log.info("time for %s-th iteration is %s seconds" % (m, time.time() - m_th_iteration_time))
        return path_list, individual_payoffs, average_payoff, val_value_list


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
                h.append(self.u[n](x[n]))
            else:
                h.append(1)
            # max = torch.max(torch.tensor([h1, h2]))
            # U[n] = max * (torch.ones(1) - sum)
            U.append(h[n] * (torch.ones(1) - sum[n]))

        z = torch.stack(U)

        assert torch.sum(z).item() == 1

        # w = torch.unsqueeze(z, 0)

        return z

        sum[0] = 0
        # x[:, 0] = torch.tensor(x_input[:, 0], requires_grad=True)
        h[0] = self.u[0](x[:, 0])
        U[0] = h[0] * (torch.ones(1) - sum[0])

        sum[1] = torch.sum(U[0:1])  # 0...n-1
        # x[:, 1] = torch.tensor(x_input[:, 1], requires_grad=True)
        h[1] = self.u[1](x[:, 1])
        # max = torch.max(torch.tensor([h1, h2]))
        # U[n] = max * (torch.ones(1) - sum)
        U[1] = h[1] * (torch.ones(1) - sum[1])

        # sum[2] = torch.sum(U[0:2])  # 0...n-1
        sum[2] = U[0] + U[1]

        # x[:, 1] = torch.tensor(x_input[:, 1], requires_grad=True)
        h[2] = self.u[2](x[:, 2])

        # max = torch.max(torch.tensor([h1, h2]))
        # U[n] = max * (torch.ones(1) - sum)
        U[2] = h[2] * (torch.ones(1) - sum[2])

        return U[2]

    """def generate_stopping_time_factors_from_path(self, x_input):
        local_N = x_input.shape[1]
        U = torch.empty(local_N)
        sum = torch.empty(local_N)
        # sum = []
        x = torch.empty(self.d, local_N, requires_grad=True)
        # x = torch.from_numpy(x_input) doesn't work for some reason

        h = torch.empty(self.N + 1)

        h1 = torch.empty(self.N + 1)
        h1[self.N] = 1
        
        for n in range(local_N):
            sum.append(torch.sum(U[0:n]))  # 0...n-1
            x[:, n] = torch.tensor(x_input[:, n])
            if n < self.N:
                h1[n] = self.u[n](x[:, n])
            # max = torch.max(torch.tensor([h1, h2]))
            # U[n] = max * (torch.ones(1) - sum)
            U[n] = h1[n] * (torch.ones(1) - sum[n])
        
        assert torch.sum(U).item() == 1

        return U
        
        sum[0] = 0
        # x[:, 0] = torch.tensor(x_input[:, 0], requires_grad=True)
        h[0] = self.u[0](x[:, 0])
        U[0] = h[0] * (torch.ones(1) - sum[0])

        sum[1] = torch.sum(U[0:1])  # 0...n-1
        # x[:, 1] = torch.tensor(x_input[:, 1], requires_grad=True)
        h[1] = self.u[1](x[:, 1])
        # max = torch.max(torch.tensor([h1, h2]))
        # U[n] = max * (torch.ones(1) - sum)
        U[1] = h[1] * (torch.ones(1) - sum[1])

        # sum[2] = torch.sum(U[0:2])  # 0...n-1
        sum[2] = U[0] + U[1]

        # x[:, 1] = torch.tensor(x_input[:, 1], requires_grad=True)
        h[2] = self.u[2](x[:, 2])

        # max = torch.max(torch.tensor([h1, h2]))
        # U[n] = max * (torch.ones(1) - sum)
        U[2] = h[2] * (torch.ones(1) - sum[2])

        return U[2]
    """

    def calculate_payoffs(self, U, x, g, t):
        sum = torch.zeros(1)
        for n in range(self.N + 1):
            sum += U[n] * g(t[n], x[:, n])
        return sum

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

    def test2(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # an affine operation: y = Wx + b
                self.fc1 = nn.Linear(1, 1)
                self.fc2 = nn.Linear(1, 1)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = torch.sigmoid(x)
                return x

        net = Net()
        print(net(torch.ones(1) * -1000))
        print(net(torch.zeros(1)))
        print(net(torch.ones(1)))
        print(net(torch.ones(1) * 100))
        print("Hii")

    def rectified_minimum(self, x, const, factor):
        return torch.min(x, const + x * (factor + 100))

    def rectified_max(self, x, const, factor):
        return torch.max(x, const + x * (factor + 100))
