class SoftmaxTemperature:
    @staticmethod
    def visit_softmax_temperature_fn(trained_steps,training_steps):
        if trained_steps < 0.25 * training_steps:
            return 1.0
        elif trained_steps < 0.5 * training_steps:
            return 0.75
        elif trained_steps < 0.75 * training_steps:
            return 0.5
        elif training_steps - trained_steps <= 10:
            return 0 
        return 0.25
