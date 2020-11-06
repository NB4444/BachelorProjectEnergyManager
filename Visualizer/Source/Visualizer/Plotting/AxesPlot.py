import collections
import math
from typing import Dict, Any

import mplcursors as mplcursors
from matplotlib import dates

from Visualizer.Plotting.FigurePlot import FigurePlot


class AxesPlot(FigurePlot):
    def __init__(self, title: str, plot_series: Dict[str, Dict[Any, Any]], x_label: str = "x", y_label: str = "y", legend: bool = True,
                 grid: bool = True,
                 format_x_as_date: bool = False):
        super().__init__(title=title)

        self.plot_series = collections.OrderedDict(sorted(plot_series.items()))
        self.format_x_as_date = format_x_as_date
        self.x_label = x_label
        self.y_label = y_label
        self.legend = legend
        self.grid = grid
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

    def on_plot_figure(self, figure, axes):
        # Format dates
        if self.format_x_as_date:
            axes.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M:%S"))
            figure.autofmt_xdate()
            # pyplot.xticks(rotation=45)

        # Make the x-axis visible
        axes.xaxis.set_tick_params(which="both", labelbottom=True)
        # pyplot.setp(axes.get_xticklabels(), visible=True)

        # Enable grid
        if self.grid:
            axes.grid()

        # Set the labels
        axes.set(xlabel=self.x_label, ylabel=self.y_label)

        # Collect the values
        x_values = []
        y_values = []
        for _, series_values in self.plot_series.items():
            for timestamp, value in series_values.items():
                x_values.append(timestamp)
                y_values.append(value)

        # Set the ranges
        x_min = min(x_values) if len(x_values) > 0 else 0
        x_max = max(x_values) if len(x_values) > 0 else 0
        axes.set_xlim(x_min if self.x_min is None else self.x_min, x_max if self.x_max is None else self.x_max)
        y_min = min(y_values) if len(y_values) > 0 else 0
        y_max = max(y_values) if len(y_values) > 0 else 0
        axes.set_ylim(((y_min - abs(0.1 * y_min)) if y_min < 0 else 0) if self.y_min is None else self.y_min, (y_max + abs(0.1 * y_max)) if self.y_max is None else self.y_max)

        self.on_plot_series(figure, axes)

        # Enable legend
        if self.legend:
            axes.legend(bbox_to_anchor=(1.15, 1), loc="upper left", ncol=math.ceil(len(self.plot_series) / 10))

        # Enable labels
        mplcursors.cursor(hover=True)

    def on_plot_series(self, figure, axes):
        raise NotImplementedError()
