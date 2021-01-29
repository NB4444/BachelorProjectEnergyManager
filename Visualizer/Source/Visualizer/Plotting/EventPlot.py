import matplotlib
import numpy
from datetime import datetime
from matplotlib import pyplot
from matplotlib.dates import date2num, AutoDateFormatter, AutoDateLocator
from random import Random
from typing import List, Tuple

from Visualizer.Plotting.FigurePlot import FigurePlot


class EventPlot(FigurePlot):
    def __init__(self, title: str, events: List[Tuple[str, datetime, datetime]],
                 random_colors: bool = False):
        super().__init__(title=title)

        self.events = events
        self.random_colors = random_colors

        self.axes = self.figure.add_subplot(1, 1, 1)

        # Set the labels
        self.axes.set(title=self.title)

    def on_plot_figure(self, figure):
        axes = self.axes

        # Define the y offset
        y_offset = 0.3

        # Disable the y axis
        # axes.get_yaxis().set_visible(False)
        # axes.set_aspect(1)

        # Set the x label
        axes.set_xlabel("Timestamp")

        # Enable grid
        axes.grid()

        # Format dates
        # axes.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        # pyplot.xticks(rotation=45)

        # First collect the event data
        event_names = []
        minimum_timestamp = None
        maximum_timestamp = None
        for (event, start_timestamp, end_timestamp) in self.events:
            if minimum_timestamp is None or start_timestamp < minimum_timestamp:
                minimum_timestamp = start_timestamp
            if maximum_timestamp is None or end_timestamp > maximum_timestamp:
                maximum_timestamp = end_timestamp
            event_names.append(event)
        event_names = sorted(list(set(event_names)))
        if minimum_timestamp is not None and maximum_timestamp is not None:
            axes.set_xlim(date2num(minimum_timestamp), date2num(maximum_timestamp))
            axes.set_ylim(0, len(event_names))

        # Set the ticks
        axes.set_yticks([event_id for event_id, _ in enumerate(event_names)], minor=False)
        axes.set_yticklabels([])

        # Set the colors
        colors = [pyplot.get_cmap("gist_rainbow")(1. * i / len(event_names)) for i in range(len(event_names))]
        if self.random_colors:
            Random(0).shuffle(colors)
        axes.set_prop_cycle(color=colors)

        # Create the event patterns
        for event, start_timestamp, end_timestamp in self.events:
            event_id = event_names.index(event)
            x_range = [date2num(start_timestamp), date2num(end_timestamp)]
            y_range = numpy.array([event_id, event_id])
            y_range_2 = y_range + 1

            # Create the rectangle
            # axes.add_patch(
            #     Rectangle((date2num(start_timestamp), event_id), date2num(end_timestamp) - date2num(start_timestamp), 1,
            #               color=colors[event_id]))

            axes.fill_between(x_range, y_range, y2=y_range_2, color=colors[event_id])

            def average(values):
                return sum(values) / len(values)

            # Add the annotation
            average_x = average(x_range)
            average_y = average([y_range[0], y_range_2[0]])
            # axes.text(average_x, average_y, event, horizontalalignment="center", verticalalignment="center")

        # Assign date locator / formatter to the x-axis to get proper labels
        locator = AutoDateLocator(minticks=3)
        formatter = AutoDateFormatter(locator)
        axes.xaxis.set_major_locator(locator)
        axes.xaxis.set_major_formatter(formatter)

        # Create the labels
        for event_id, event_name in enumerate(event_names):
            self.annotations.append(
                axes.annotate(event_name, xy=(0, event_id + y_offset), xytext=(-125, 0),
                              xycoords=("axes fraction", "data"),
                              textcoords="offset points"))

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
