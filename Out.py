import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfp
import torch


class Output:

    def __init__(self, output_location):
        self.NN = 0
        self.config = 0
        self.Model = 0
        self.average_payoff = []
        self.val_value_list = []
        self.train_duration = []
        self.val_duration = []
        self.net_net_duration = []
        self.output_location = output_location

        import os

        print("Current Working Directory ", os.getcwd())

        try:
            # Change the current working Directory
            os.chdir(output_location)
            print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")
        print("Current Working Directory ", os.getcwd())

        # Check if New path exists
        if os.path.exists(output_location):
            # Change the current working Directory
            os.chdir(output_location)
        else:
            print("Can't change the Current Working Directory")

        print("Current Working Directory ", os.getcwd())

    def create_pdf(self):
        a = min(self.config.xi, self.config.xi + self.Model.getT() * self.Model.getmu(1))
        b = max(self.config.xi, self.config.xi + self.Model.getT() * self.Model.getmu(1))
        a = a - 1.5 * self.Model.getT() * self.Model.getsigma(1)
        b = b + 1.5 * self.Model.getT() * self.Model.getsigma(1)

        # TODO: copy graph so i only use a copy when it was still open
        pdf = pdfp.PdfPages("graph.pdf")

        t = np.linspace(a, b, 20)
        l = len(self.NN.u) - 1
        x = np.zeros((l, t.shape[0]))
        for k in range(l):
            c_fig = plt.figure(k)

            for j in range(len(t)):
                h = torch.tensor(np.ones(1) * t[j], dtype=torch.float32)
                x[k][j] = self.NN.u[k](h)
            help = x[k][:]
            plt.ylim([0, 1])
            plot(t, x[k][:], linewidth=4)
            xlabel('x', fontsize=16)
            ylabel('u_%s(x)' % k, fontsize=16)
            grid(True)
            pdf.savefig(c_fig)
            plt.close(c_fig)

        pdf.close()

        """
        def _plot_fig(train_results, valid_results, model_name):
        colors = ["red", "blue", "green"]
        xs = np.arange(1, train_results.shape[1]+1)
        plt.figure()
        legends = []
        for i in range(train_results.shape[0]):
            plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
            plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
            legends.append("train-%d"%(i+1))
            legends.append("valid-%d"%(i+1))
        plt.xlabel("Epoch")
        plt.ylabel("Normalized Gini")
        plt.title("%s"%model_name)
        plt.legend(legends)
        plt.savefig("./fig/%s.png"%model_name)
        plt.close()
    """

        """
        t = np.linspace(0.0, self.Model.getT(), self.N + 1)
        for k in range(self.Model.getd()):
            plot(t, x[k])
        xlabel('t', fontsize=16)
        ylabel('x', fontsize=16)
        grid(True)
        show()
        """
        """
        fig = plt.figure()
        x = np.arange(10)
        y = 2.5 * np.sin(x / 20 * np.pi)
        yerr = np.linspace(0.05, 0.2, 10)

        plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')

        plt.errorbar(x, y + 2, yerr=yerr, uplims=True, label='uplims=True')

        plt.errorbar(x, y + 1, yerr=yerr, uplims=True, lolims=True,
                     label='uplims=True, lolims=True')

        upperlimits = [True, False] * 5
        lowerlimits = [False, True] * 5
        plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits,
                     label='subsets of uplims and lolims')

        plt.legend(loc='lower right')
        pdf = pdfp.PdfPages("output.pdf")
        pdf.savefig(fig)
        pdf.savefig(fig)
        pdf.close()
        """
        """
        t = np.arange(0.0, 2.0, 0.01)
        s1 = np.sin(2 * np.pi * t)
        s2 = np.sin(4 * np.pi * t)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, s1)
        plt.subplot(212)
        plt.plot(t, 2 * s1)

        plt.figure(2)
        plt.plot(t, s2)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(t, s2, 's')
        ax = plt.gca()
        ax.set_xticklabels([])

        plt.show()
        """
