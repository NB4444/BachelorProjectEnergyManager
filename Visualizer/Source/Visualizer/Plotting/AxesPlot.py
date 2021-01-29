import collections
import math
import matplotlib
import mplcursors
from matplotlib import dates, pyplot
from mpl_toolkits.mplot3d import Axes3D
from mplcursors import Selection
from natsort import natsorted
from random import Random
from typing import Any, List, OrderedDict

from Visualizer.Plotting.FigurePlot import FigurePlot


class AxesPlot(FigurePlot):
    def __init__(self, title: str, plot_series: OrderedDict[str, OrderedDict[Any, Any]],
                 x_label: str = "x", y_label: str = "y",
                 z_label: str = "z", legend: bool = True,
                 grid: bool = True,
                 format_x_as_date: bool = False, random_colors: bool = False, labels: List[Any] = None):
        super().__init__(title=title)

        self.plot_series = collections.OrderedDict()
        for name, data in natsorted(plot_series.items()):
            self.plot_series[name] = data
        self.format_x_as_date = format_x_as_date
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.legend = legend
        self.grid = grid
        self.random_colors = random_colors
        self.labels = labels

        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None
        if self.is_3d:
            self.axes = Axes3D(self.figure)
        else:
            self.axes = self.figure.add_subplot(1, 1, 1)

            # Set the labels
            self.axes.set(title=self.title)

    def on_plot_figure(self, figure):
        axes = self.axes

        # Format dates
        if self.format_x_as_date:
            axes.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M:%S"))
            figure.autofmt_xdate()
            # pyplot.xticks(rotation=45)

        # Make the x-axis visible
        # axes.xaxis.set_tick_params(which="both", labelbottom=True)
        # pyplot.setp(axes.get_xticklabels(), visible=True)

        # Enable grid
        if self.grid:
            axes.grid()

        # Set the labels
        axes.set_xlabel(self.x_label)
        axes.set_ylabel(self.y_label)
        if self.is_3d:
            axes.set_zlabel(self.z_label)

        # Collect the values
        x_values = []
        y_values = []
        z_values = []
        for _, x_y_z_values in self.plot_series.items():
            for x_value, y_z_values in x_y_z_values.items():
                x_values.append(x_value)
                if self.is_3d:
                    for y_value, z_value in y_z_values.items():
                        y_values.append(y_value)
                        z_values.append(z_value)
                else:
                    y_values.append(y_z_values)

        # Set the ranges
        def set_ranges(setter, self_minimum, self_maximum, values):
            if len(values) > 0 and (isinstance(values[0], int) or isinstance(values[0], float)):
                minimum = min(values) if len(values) > 0 else 0
                maximum = max(values) if len(values) > 0 else 0
                setter(((minimum - abs(0.1 * minimum)) if minimum < 0 else 0) if self_minimum is None else self_minimum,
                       (maximum + abs(0.1 * maximum)) if self_maximum is None else self_maximum)

        set_ranges(axes.set_xlim, self.x_min, self.x_max, x_values)
        set_ranges(axes.set_ylim, self.y_min, self.y_max, y_values)
        if self.is_3d:
            set_ranges(axes.set_zlim, self.z_min, self.z_max, z_values)

        # Set the colors
        colors = [pyplot.get_cmap("gist_rainbow")(1. * i / len(self.plot_series)) for i in range(len(self.plot_series))]
        if self.random_colors:
            Random(0).shuffle(colors)
        axes.set_prop_cycle(color=colors)

        self.on_plot_series(figure, axes)

        # def onpick(event):
        #     ind = event.ind[0]
        #     x, y, z = event.artist._offsets3d
        #     os.write(1, repr(x[ind]).encode() + b'\n')
        #     os.write(1, repr(y[ind]).encode() + b'\n')
        #     os.write(1, repr(z[ind]).encode() + b'\n')
        #
        # figure.canvas.mpl_connect("pick_event", onpick)

        # Add cursors
        cursor = mplcursors.cursor(axes)

        if self.labels is not None:
            @cursor.connect("add")
            def on_add(selection: Selection):
                if not self.is_3d:
                    selection.annotation.set_text(self.labels[selection.target.index])

        # Enable legend
        if self.legend:
            max_columns = 1
            axes.legend(bbox_to_anchor=(1.15, 1), loc="upper left",
                        ncol=min(math.ceil(len(self.plot_series) / 10), max_columns))

        # Make everything fit
        if not self.is_3d:
            self.figure.tight_layout()

    def on_plot_series(self, figure, axes):
        raise NotImplementedError()

    @property
    def is_3d(self):
        return all(
            all(isinstance(value, dict) for key, value in values.items()) for _, values in self.plot_series.items())

    @property
    def extent(self):
        padding = (0.0, 0.0)
        elements = self.axes.get_xticklabels() + self.axes.get_yticklabels() + self.annotations + [self.axes,
                                                                                                   self.axes.title,
                                                                                                   self.axes.get_xaxis().get_label(),
                                                                                                   self.axes.get_yaxis().get_label(),
                                                                                                   self.axes.get_legend()]

        self.figure.canvas.draw()

        return matplotlib.transforms.Bbox.union(
            [element.get_window_extent() for element in elements if element is not None]).expanded(1.0 + padding[0],
                                                                                                   1.0 + padding[
                                                                                                       1]).transformed(
            self.figure.dpi_scale_trans.inverted())
