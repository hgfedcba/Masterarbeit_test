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
