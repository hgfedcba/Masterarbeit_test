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
        internal_neurons = 300
        self.fc1 = nn.Linear(d, internal_neurons)
        self.fc2 = nn.Linear(internal_neurons, internal_neurons)
        self.fc3 = nn.Linear(internal_neurons, 1)

    def forward(self, y):
        y = F.selu(self.fc1(y))
        y = F.selu(self.fc2(y))
        y = torch.sigmoid(self.fc3(y))
        return y


class NN:
    def __init__(self, config, Model, log):

        self.log = log
        self.gamma = config.gamma  # Lernrate
        self.nu = config.N * (2 * config.d + 1) * (config.d + 1)
        self.N = config.N
        self.d = config.d
        self.u = []
        self.Model = Model
        self.gamma = config.gamma
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
        optimizer = optim.SGD(params, lr=self.gamma, momentum=0.9)

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
            if m + 1 % 100 == 0 and self.gamma > 0.001:
                self.gamma /= 10
                optimizer = optim.SGD(params, lr=self.gamma, momentum=0.9)
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

            if m == 25:
                assert True

        return individual_payoffs, average_payoff, val_value_list, train_duration, val_duration, self.net_net_duration


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

    def test3(self):
        class Net(nn.Module):
            def __init__(self, d):
                super(Net, self).__init__()
                # an affine operation: y = Wx + b
                self.fc1 = nn.Linear(d, 100)
                self.fc2 = nn.Linear(100, 100)
                self.fc4 = nn.Linear(100, 1)

            def forward(self, y):
                """
                y = F.relu(self.fc1(y))
                y = F.relu(self.fc2(y))
                y = torch.sigmoid(self.fc3(y))
                """
                y = F.relu(self.fc1(y))
                y = F.relu(self.fc2(y))
                y = self.fc4(y)
                return y

        # u = Net(1)
        u = nn.Sequential(nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1))

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
                x[j] = u(h)
            help = x
            plt.ylim([0, 1])
            plot(t, x, linewidth=4)
            xlabel('x', fontsize=16)
            ylabel('u(x)', fontsize=16)
            grid(True)
            pdf.savefig(c_fig)

            pdf.close()
            plt.close(c_fig)

        def f(x):
            # return x * x / 64 - 5 * x / 4 + 25
            if x > 40:
                return torch.zeros(1)
            else:
                return torch.ones(1) * 40 - x

        # optimizer = optim.Adam(u.parameters(), lr=0.01)
        optimizer = optim.Adam(u.parameters(), lr=0.00001)

        for m in range(500):
            optimizer.zero_grad()
            batchsize = 32
            x = torch.rand(batchsize) * 20 + 30
            x.requires_grad = True
            loss = torch.zeros(batchsize)
            for j in range(batchsize):
                loss[j] = torch.norm(u(x[j] * torch.ones(1)) - f(x[j]))
            # print("x = " + str(x.item()) + ",\tu(x) = " + str(u(x).item()) + ",\tf(x) = " + str(f(x).item()) + ",\tloss = " + str(loss.item()))
            loss_sum = torch.sum(loss)
            loss_sum.backward()
            optimizer.step()
            if m % 20 == 0:
                out(m)

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



    def rectified_minimum(self, x, const, factor):
        return torch.min(x, const + x * (factor + 100))

    def rectified_max(self, x, const, factor):
        return torch.max(x, const + x * (factor + 100))
