class BestResults:
    def __init__(self, log):
        self.cont_best_result = IndividualBestResult()
        self.disc_best_result = IndividualBestResult()
        self.paths_for_final_val = 0
        self.log = log

    def process_current_iteration(self, NN, m, val_cont_value, val_disc_value, stopping_times, total_time_used):
        if val_cont_value > self.cont_best_result.cont_value or (val_cont_value == self.cont_best_result.cont_value and val_disc_value > self.cont_best_result.disc_value):
            self.cont_best_result.update(NN, m, val_cont_value, val_disc_value, stopping_times, total_time_used)
            self.log.info("This is a new cont best!!!!!")

        if val_disc_value > self.disc_best_result.disc_value or (val_disc_value == self.disc_best_result.disc_value and val_cont_value > self.disc_best_result.cont_value):
            self.disc_best_result.update(NN, m, val_cont_value, val_disc_value, stopping_times, total_time_used)
            self.log.info("This is a new disc best!!!!!")

    def get_m_max(self):
        return max(self.cont_best_result.m, self.disc_best_result.m)

    def get_max_time_to_best_result(self):
        return max(self.cont_best_result.time_to_best_result, self.disc_best_result.time_to_best_result)

    def final_validation(self):
        return self.cont_best_result.final_validation(self.paths_for_final_val), self.disc_best_result.final_validation(self.paths_for_final_val)


class IndividualBestResult:
    def __init__(self):
        self.cont_value = -1
        self.disc_value = -1
        self.m = 0
        self.NN = None
        self.stopping_times = None
        self.time_to_best_result = None

    def update(self, NN, m, val_cont_value, val_disc_value, stopping_times, time_to_best_result):
        self.NN = NN
        self.m = m
        self.cont_value = val_cont_value
        self.disc_value = val_disc_value
        self.stopping_times = stopping_times
        self.time_to_best_result = time_to_best_result

    def final_validation(self, paths):
        return self.NN.validate([], paths, [], [], [])

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
