import time
import math
import random


# noinspection SpellCheckingInspection
class MathematicalModel:
    def __init__(self, T, d, mu, sigma, g, xi):
        print("Model init wird aufgerufen")
        self.__T = T  # Time Horizon
        self.__d = d  # Dimension of the underlying Problem
        self.__internal_mu = mu  # drift coefficient
        self.__internal_sigma = sigma  # standard deviation of returns
        self.__internal_g = g  # objective function
        self.__xi = xi  # Startwert

    def getT(self):
        return self.__T

    def getd(self):
        return self.__d

    def getxi(self):
        return self.__xi

    def getmu(self, x):
        self.assert_x_in_Rd(x, self.getd())

        out = self.__internal_mu(x)
        
        self.assert_x_in_Rd(out, self.getd())

        return out

    def getsigma(self, x):
        self.assert_x_in_Rd(x, self.getd())

        out = self.__internal_sigma(x)

        if self.getd() > 1:
            assert type(out).__module__ == 'numpy'
            assert out.shape[0] == self.getd()
            assert out.shape[1] == self.getd()
        else:
            assert isinstance(out, (int, float)) or out.size == 1

        return out

    def getg(self, t, x):
        assert 0 <= t <= self.getT()
        self.assert_x_in_Rd(x, self.getd())

        out = self.__internal_g(t, x)

        # torch...
        # self.assert_x_in_Rd(out, 1)

        return out


    def setmu(self, mu):
        self.__internal_mu = mu

    def setsigma(self, sigma):
        self.__internal_sigma = sigma


    def assert_x_in_Rd(self, x, d):
        if d > 1:
            assert type(x).__module__ == 'numpy'
            assert x.size == d
        else:
            assert isinstance(x, (int, float)) or x.size == 1
