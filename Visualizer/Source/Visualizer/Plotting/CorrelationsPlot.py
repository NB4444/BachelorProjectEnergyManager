import seaborn
from matplotlib import pyplot

from Visualizer.Plotting.FigurePlot import FigurePlot


class CorrelationsPlot(FigurePlot):
    def __init__(self, title: str, correlations):
        super().__init__(title=title)

        self.correlations = correlations
        self.figure = pyplot.figure(figsize=(100, 50))
        self.axes = self.figure.add_subplot(1, 1, 1)

    def on_plot_figure(self, figure, axes):
        seaborn.heatmap(self.correlations, annot=True, cmap=pyplot.cm.Reds)

        # Make everything fit
        # pyplot.tight_layout()
