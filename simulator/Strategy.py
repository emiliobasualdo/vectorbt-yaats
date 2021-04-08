import vectorbt as vbt

class Strategy:
    def optimize(self, candles, windows, **kwargs) -> vbt.Portfolio:
        pass

    def simulate(self, candles, best_fast_windows, best_slow_windows, **kwargs) -> vbt.Portfolio:
        pass

    def __str__(self):
        return self.__class__.__name__