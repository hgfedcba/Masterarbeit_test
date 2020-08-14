class BestResult:
    def __init__(self):
        self.val_error_cont = -1
        self.val_error_disc = -1
        self.m = 0
        self.paths = 0

    def update(self, NN, m, val_error_cont, val_error_disc, stopping_times, time_to_best_result):
        self.NN = NN
        self.m = m
        self.val_error_cont = val_error_cont
        self.val_error_disc = val_error_disc
        self.stopping_times = stopping_times
        self.time_to_best_result = time_to_best_result

    def final_validation(self):
        return self.NN.validate([], self.paths, [], [], [])

    """
    self.log = log
    self.lr = config.lr  # Lernrate
    self.lr_sheduler_breakpoints = config.lr_sheduler_breakpoints
    self.nu = config.N * (2 * config.d + 1) * (config.d + 1)
    self.N = config.N
    self.d = config.d
    self.u = []
    self.Model = Model
    self.t = config.time_partition
    self.net_net_duration = []

    self.internal_neurons = config.internal_neurons
    self.activation1 = config.activation1
    self.activation2 = config.activation2
    self.optimizer = config.optimizer

    self.validation_frequency = config.validation_frequency
    self.antithetic_variables = config.antithetic_variables

    self.out = out
    """
