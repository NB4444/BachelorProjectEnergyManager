import humanize
from matplotlib import pyplot


class Plot(object):
    @staticmethod
    def free():
        # Free up memory
        pyplot.draw()
        pyplot.clf()
        pyplot.close("all")

    @staticmethod
    def humanize_number(value):
        return humanize.intcomma(value)

    @staticmethod
    def j_to_wh(value):
        return value / 3600

    @staticmethod
    def to_percentage(value):
        return value * 100

    @staticmethod
    def ns_to_s(value):
        return value / 1e9

    @staticmethod
    def humanize_size(value):
        return humanize.naturalsize(value)

    @staticmethod
    def parse_number_list(value):
        return [number for number in value.split(",")]

    def __init__(self, title: str):
        self.title = title

    def plot(self):
        self.on_plot()

    def on_plot(self):
        raise NotImplementedError()
