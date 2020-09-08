import collections
import math
from typing import List, Any

import humanize
import matplotlib
import pandas
import seaborn
from matplotlib import pyplot, dates, gridspec

# Define conversion functions
j_to_wh = lambda value: value / 3600
to_percentage = lambda value: value * 100
ns_to_s = lambda value: value / 1e9
humanize_number = lambda value: humanize.intcomma(value)
humanize_size = lambda value: humanize.naturalsize(value)
parse_number_list = lambda value: [number for number in value.split(",")]


def determine_extent(elements, padding=(0.0, 0.0)):
    return matplotlib.transforms.Bbox.union([element.get_window_extent() for element in elements]).expanded(1.0 + padding[0], 1.0 + padding[1])


def plot_timeseries(title, plot_series, figure=None, axes=None, x_label="Timestamp", y_label="Value", legend=True, grid=True, format_x_as_date=False, show_final_values=True,
                    output_directory: str = None):
    if figure is None:
        figure = pyplot.figure(figsize=(15, 5))

    if axes is None:
        axes = figure.add_subplot(1, 1, 1)

    # Check if there is data to plot
    if len(plot_series) == 0 or all([len(series_values) == 0 for _, series_values in plot_series.items()]):
        # axes.text(0.5, 0.5, "No data", verticalalignment="center", horizontalalignment="center", fontfamily="monospace")

        return figure

    # Sort the labels for the legend
    plot_series = collections.OrderedDict(sorted(plot_series.items()))

    # Collect the values
    x_values = []
    y_values = []
    for _, series_values in plot_series.items():
        for timestamp, value in series_values.items():
            x_values.append(timestamp)
            y_values.append(value)

    # Set the ranges
    x_min = min(x_values) if len(x_values) > 0 else 0
    x_max = max(x_values) if len(x_values) > 0 else 0
    axes.set_xlim(x_min, x_max)
    y_min = min(y_values) if len(y_values) > 0 else 0
    y_max = max(y_values) if len(y_values) > 0 else 0
    axes.set_ylim(((y_min - abs(0.1 * y_min)) if y_min < 0 else 0), (y_max + abs(0.1 * y_max)))

    # Set the data
    annotations = list()
    for (series_name, series_style), series_values in plot_series.items():
        if len(series_values) > 0:
            axes.plot(series_values.keys(), series_values.values(), series_style, label=series_name)

            if show_final_values and len(series_values.values()) > 0:
                final_value = list(series_values.values())[-1]
                annotations.append(pyplot.annotate(f"{humanize_number(final_value)}", xy=(1, final_value), xytext=(8, 0), xycoords=("axes fraction", "data"), textcoords="offset points"))

    # Set the labels
    axes.set(xlabel=x_label, ylabel=y_label, title=title)

    # Format dates
    if format_x_as_date:
        axes.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        figure.autofmt_xdate()
        # pyplot.xticks(rotation=45)

    # Make the x-axis visible
    axes.xaxis.set_tick_params(which="both", labelbottom=True)
    # pyplot.setp(axes.get_xticklabels(), visible=True)

    # Enable grid
    if grid:
        axes.grid()

    # Enable legend
    if legend:
        axes.legend(bbox_to_anchor=(1.15, 1), loc="upper left", ncol=math.ceil(len(plot_series) / 10))

    # Make everything fit
    pyplot.tight_layout()

    # Save the subplot
    if output_directory is not None:
        axes.figure.canvas.draw()
        extent = determine_extent(
            axes.get_xticklabels() + axes.get_yticklabels() + annotations + [axes, axes.title, axes.get_xaxis().get_label(), axes.get_yaxis().get_label(), axes.get_legend()]).transformed(
            figure.dpi_scale_trans.inverted())

        # Make everything fit
        pyplot.tight_layout()
        figure.savefig(f"{output_directory}/{title}.png", bbox_inches=extent)

    return figure


def plot_timeseries_overview(summary_plot_generator, plot_generators, figure=None, output_directory: str = None):
    if figure is None:
        figure = pyplot.figure(figsize=(30, 30))

    # Create the figure axes
    grid_spec = gridspec.GridSpec(len(plot_generators) + 1, 1)
    axes = figure.add_subplot(grid_spec[0])

    # Create the information summary
    summary_plot_generator(figure, axes)

    # Create style cycler
    # style_cycler = (cycler.cycler(color=list("rgb")) + cycler.cycler(linestyle=['-', '--', '-.']))

    # Create the plots
    index = 1
    previous_axes = None
    for plot_generator in plot_generators:
        # Create the plot
        axes = figure.add_subplot(grid_spec[index]) if previous_axes is None else figure.add_subplot(grid_spec[index], sharex=previous_axes)
        index += 1
        previous_axes = axes

        # Cycle line styles
        # axes.set_prop_cycle(style_cycler)

        plot_generator(figure=figure, axes=axes, output_directory=output_directory)

    # Make everything fit
    pyplot.tight_layout()

    # Save the plot
    if output_directory is not None:
        figure.savefig(f"{output_directory}/Summary.png")

    return figure


def plot_table(data: List[Any], columns: List[str], maximum_column_width: int = None, maximum_columns: int = None, minimum_rows: int = None, maximum_rows: int = 50):
    pandas.options.display.max_colwidth = maximum_column_width
    pandas.options.display.max_columns = maximum_columns
    pandas.options.display.min_rows = minimum_rows
    pandas.options.display.max_rows = maximum_rows

    table = pandas.DataFrame(data, columns=columns)
    table.infer_objects()
    table

    return table


def plot_dictionary(data, figure=None, axes=None, output_directory: str = None):
    if figure is None:
        figure = pyplot.figure(figsize=(15, 5))

    if axes is None:
        axes = figure.add_subplot(1, 1, 1)

    axes.axis("off")
    axes.invert_yaxis()
    text = ""
    for name, value in sorted(data.items()):
        text += "{0:>100}  {1:<50}\n".format(str(name), str(value))
    axes.text(0.5, 0.5, text, verticalalignment="center", horizontalalignment="center", fontfamily="monospace")

    # Make everything fit
    pyplot.tight_layout()

    # Save the plot
    if output_directory is not None:
        figure.savefig(f"{output_directory}/Summary.png")

    return figure


def plot_correlations(correlations, figure=None, axes=None, output_directory: str = None):
    if figure is None:
        figure = pyplot.figure(figsize=(60, 60))

    if axes is None:
        axes = figure.add_subplot(1, 1, 1)

    seaborn.heatmap(correlations, annot=True, cmap=pyplot.cm.Reds)

    # Make everything fit
    pyplot.tight_layout()

    # Save the plot
    if output_directory is not None:
        figure.savefig(f"{output_directory}/Correlations.png")


def free():
    # Free up memory
    pyplot.draw()
    pyplot.clf()
    pyplot.close("all")
