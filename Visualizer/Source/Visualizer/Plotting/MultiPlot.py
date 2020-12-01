from typing import List

from matplotlib import pyplot, gridspec

from Visualizer.Plotting.FigurePlot import FigurePlot
from Visualizer.Plotting.Plot import Plot


class MultiPlot(FigurePlot):
    def __init__(self, title: str, plots: List[Plot]):
        super().__init__(title=title)

        self.plots = plots
        self.figure = pyplot.figure(figsize=(30, 30))
        self.axes = self.figure.add_subplot(1, 1, 1)

    def plot(self):
        super().plot()

    def on_plot_figure(self, figure):
        # Create the figure axes
        grid_spec = gridspec.GridSpec(len(self.plots) + 1, 1)

        # Create style cycler
        # style_cycler = (cycler.cycler(color=list("rgb")) + cycler.cycler(linestyle=['-', '--', '-.']))

        # Create the plots
        index = 1
        previous_axes = None
        for plot in self.plots:
            # Create the plot
            axes = figure.add_subplot(grid_spec[index]) if previous_axes is None else figure.add_subplot(
                grid_spec[index], sharex=previous_axes)
            index += 1
            previous_axes = axes

            # Cycle line styles
            # axes.set_prop_cycle(style_cycler)

            plot.figure = figure
            plot.axes = axes
            plot.plot()

        # Make everything fit
        # pyplot.tight_layout()
