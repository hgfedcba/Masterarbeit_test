import datetime
import time
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfp
import torch
import os
import NN


class Output:

    def __init__(self, output_location):
        self.NN = 0
        self.config = 0
        self.Model = 0
        self.T = 0
        self.N = 0
        self.average_payoff = []
        self.val_value_list = []
        self.train_duration = []
        self.val_duration = []
        self.net_net_duration = []
        self.output_location = output_location
        self.stock_price_partition = 0

        # print("Current Working Directory ", os.getcwd())

        try:
            # Change the current working Directory
            os.chdir(output_location)
            # print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")
        # print("Current Working Directory ", os.getcwd())
        """
        # Check if New path exists
        if os.path.exists(output_location):
            # Change the current working Directory
            os.chdir(output_location)
        else:
            print("Can't change the Current Working Directory")
        """

    def generate_stock_price_partition(self):
        # TODO:properly
        """
        a = min(self.config.xi, self.config.xi + self.Model.getT() * self.Model.getmu(np.ones(self.config.d)))
        b = max(self.config.xi, self.config.xi + self.Model.getT() * self.Model.getmu(np.ones(self.config.d)))
        a = a - 1.5 * self.Model.getT() * self.Model.getsigma(np.ones(self.config.d))
        b = b + 1.5 * self.Model.getT() * self.Model.getsigma(np.ones(self.config.d))
        """
        self.stock_price_partition = np.linspace(20, 50, 31)

    def create_net_pdf(self, name):
        # TODO: copy graph so i only use a copy when it was still open
        pdf = pdfp.PdfPages(name)

        t = self.stock_price_partition
        l = len(self.NN.u)
        x = np.zeros((l, t.shape[0]))
        for k in range(l):
            c_fig = plt.figure(k)

            for j in range(len(t)):
                h = torch.tensor(np.ones(self.config.d) * t[j], dtype=torch.float32)
                x[k][j] = self.NN.u[k](h)
            help = x[k][:]
            plt.ylim([0, 1])  # ylim-max is 0.5 since u is small anyway
            plot(t, x[k][:], linewidth=4)
            xlabel('x', fontsize=16)
            ylabel('u_%s(x)' % k, fontsize=16)
            grid(True)
            pdf.savefig(c_fig)
            plt.close(c_fig)

        pdf.close()

    def generate_metric_pdf(self, test, iteration_number):
        # TODO: Stop at stopping time
        # TODO: sensible y-lim

        pdf = pdfp.PdfPages("Metrics" + str(iteration_number) + ".pdf")

        self.generate_stock_price_partition()
        if self.config.d == 1:
            plot_number_paths = int(time.time())
            self.draw_point_graph(self.val_path_list, plot_number_paths)
            fig1 = plt.figure(plot_number_paths)

            for k in range(len(self.val_path_list)):
                stop_point = np.argmax(self.best_result.stopping_times[k])
                plt.scatter(self.Model.get_time_partition(self.N)[stop_point], self.val_path_list[k].flatten()[stop_point], marker='o')
            plt.ylim([0, 100])  # TODO:dynamic
            pdf.savefig(fig1)
            plt.close(fig1)

        # TODO: different time partitions
        l = len(self.train_duration)
        t_val = np.linspace(0, l - 1, num=int(l / self.config.validation_frequency))
        t = np.linspace(0, l - 1, num=l)
        plot_number_value = int(time.time())
        fig2 = plt.figure(plot_number_value)

        self.draw_point_graph_with_given_partition([np.array(self.val_continuous_value_list)], t_val, plot_number_value)
        self.draw_point_graph_with_given_partition([np.array(self.val_discrete_value_list)], t_val, plot_number_value)
        plt.legend(["cont_value", "disc_value"])
        plt.axvline(self.best_result.m, color="red")
        plt.axhline(self.config.other_computation, color='black')

        pdf.savefig(fig2)
        plt.close(fig2)

        plot_number_time = int(time.time())
        fig3 = plt.figure(plot_number_time)

        self.draw_point_graph_with_given_partition([np.array(self.train_duration)], t, plot_number_time)
        self.draw_point_graph_with_given_partition([np.array(self.val_duration)], t_val, plot_number_time)
        self.draw_point_graph_with_given_partition([np.array(self.net_net_duration)], t, plot_number_time)
        plt.legend(["train", "val", "net_net"])
        plt.axvline(self.best_result.m, color="red")

        pdf.savefig(fig3)
        plt.close(fig3)

        # TODO: sum time
        # TODO: oben zeit unten iteration
        # TODO: relative weight update

        pdf.close()
        """
        if test and False:
            show()
        else:
            self.save_fig_in_pdf(plt.figure(plot_number_paths), "Paths" + str(iteration_number) + ".pdf")
        """

    def save_fig_in_pdf(self, fig, name):
        pdf = pdfp.PdfPages(name)
        pdf.savefig(fig)

        pdf.close()

    # TODO: They are not generically working, just for what i am using them for
    def draw_in_given_plot(self, x, plot_number, partition):
        plt.figure(plot_number)
        # TODO: Better
        t = partition
        for l in range(len(x)):
            h = x[l].flatten()
            plot(t, x[l].flatten())
        xlabel('t', fontsize=16)
        ylabel('x', fontsize=16)

    def draw_point_graph(self, x, plot_number=0, step_size=1):
        if plot_number == 0:
            h = time.time()
            plot_number = int(h)
        self.draw_in_given_plot(x, plot_number, self.Model.get_time_partition(self.N, step_size))
        plt.figure(plot_number)
        grid(True)
        # show()
        # plt.close(fig)

        return plot_number

    def draw_point_graph_with_given_partition(self, x, partition, plot_number=0):
        if plot_number == 0:
            h = time.time()
            plot_number = int(h)
        plt.figure(plot_number)
        t = partition
        for l in range(len(x)):
            h = x[l].flatten()
            plot(t, x[l].flatten())
        xlabel('t', fontsize=16)
        ylabel('x', fontsize=16)
        plt.figure(plot_number)
        grid(True)
        # show()
        # plt.close(fig)

        return plot_number

    def draw_function(self, f, plot_number=0):
        if plot_number == 0:
            h = time.time()
            plot_number = int(h)
        t = self.stock_price_partition
        x = []
        for c in t:
            h = torch.tensor([c], dtype=torch.float32)
            x.append(f(h))

        plt.figure(plot_number)
        plot(t, x)
        xlabel('x', fontsize=16)
        ylabel('f(x)', fontsize=16)
        plt.ylim([0, 1])
        grid(True)
        show()
        # plt.close(fig)

        return plot_number


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
