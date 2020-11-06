import matplotlib
from matplotlib import pyplot

from Visualizer.Plotting.Plot import Plot


class FigurePlot(Plot):
    def __init__(self, title: str):
        super().__init__(title)

        # Keep track of annotations
        self.annotations = list()

        self.figure = None
        self.axes = None

    def on_plot(self):
        self.figure = pyplot.figure(figsize=(15, 5)) if self.figure is None else self.figure
        self.axes = self.figure.add_subplot(1, 1, 1) if self.axes is None else self.axes

        # Set the labels
        if self.axes != False:
            self.axes.set(title=self.title)

        self.on_plot_figure(self.figure, self.axes)

    def on_plot_figure(self, figure, axes):
        raise NotImplementedError()

    def save(self, output_directory: str = None):
        self.plot()

        # Make everything fit
        pyplot.tight_layout()

        # Save the subplot
        if output_directory is not None:
            self.axes.figure.canvas.draw()
            extent = self._determine_extent(
                self.axes.get_xticklabels() + self.axes.get_yticklabels() + self.annotations + [self.axes, self.axes.title, self.axes.get_xaxis().get_label(), self.axes.get_yaxis().get_label(),
                                                                                                self.axes.get_legend()]).transformed(
                self.figure.dpi_scale_trans.inverted())

            # Make everything fit
            pyplot.tight_layout()
            self.figure.savefig(f"{output_directory}/{self.title}.png", bbox_inches=extent)

    def _determine_extent(elements, padding=(0.0, 0.0)):
        return matplotlib.transforms.Bbox.union([element.get_window_extent() for element in elements]).expanded(1.0 + padding[0], 1.0 + padding[1])
