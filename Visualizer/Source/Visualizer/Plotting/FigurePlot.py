import matplotlib
from matplotlib import pyplot

from Visualizer.Plotting.Plot import Plot


class FigurePlot(Plot):
    def __init__(self, title: str):
        super().__init__(title)

        # Keep track of annotations
        self.annotations = list()

        self.figure = pyplot.figure(figsize=(15, 5))

    def on_plot(self):
        self.on_plot_figure(self.figure)

    def on_plot_figure(self, figure):
        raise NotImplementedError()

    def save(self, output_directory: str = None):
        self.plot()

        # Make everything fit
        self.figure.tight_layout()

        # Save the subplot
        if output_directory is not None:
            self.figure.canvas.draw()

            # Make everything fit
            pyplot.tight_layout()
            self.figure.savefig(f"{output_directory}/{self.title}.png", bbox_inches=self.extent)

    @property
    def extent(self):
        padding = (0.0, 0.0)
        elements = self.annotations

        self.figure.canvas.draw()

        return matplotlib.transforms.Bbox.union(
            [element.get_window_extent() for element in elements if element is not None]).expanded(1.0 + padding[0],
                                                                                                   1.0 + padding[
                                                                                                       1]).transformed(
            self.figure.dpi_scale_trans.inverted())
