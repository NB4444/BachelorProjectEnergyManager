from typing import Any, List

from Visualizer.Plotting.FigurePlot import FigurePlot


class HistogramPlot(FigurePlot):
    def __init__(self, title: str, values: List[Any], bins: int = 10, x_label: str = "x", grid: bool = True):
        super().__init__(title=title)

        self.values = values
        self.x_label = x_label
        self.grid = grid
        self.bins = bins

    def on_plot_figure(self, figure, axes):
        # Enable grid
        if self.grid:
            axes.grid()

        # Set the labels
        axes.set(xlabel=self.x_label, ylabel="Frequency")

        if len(self.values) > 0:
            axes.hist(self.values, bins=self.bins)
