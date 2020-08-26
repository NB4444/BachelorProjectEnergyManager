import argparse
import collections
import os
import shutil
from typing import OrderedDict, Any, Tuple

from matplotlib import pyplot, dates, gridspec

from Visualizer.Testing.Test import Test
from Visualizer.Testing.TestResults import TestResults


def parse_arguments():
    parser = argparse.ArgumentParser(prog="Visualizer", description="Visualize EnergyManager test results.")

    input_group = parser.add_argument_group("input")
    input_group.add_argument(
        "--database",
        "-d",
        metavar="FILE",
        action="store",
        help="Specifies the database to use.",
        type=str,
        required=True
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output-directory",
        "-o",
        metavar="DIRECTORY",
        action="store",
        help="Output directory to use for storing results.",
        type=str,
        required=True
    )

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()

    for test in Test.load_all(arguments.database):
        # Determine the output directory for the current test
        output_directory = f"{arguments.output_directory}/{test.name}"

        # Make sure the directory exists and is empty
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        os.makedirs(output_directory)

        def collect_values(test_results: TestResults, monitor: str, name: str, type, modifier=lambda value: value):
            values: OrderedDict[int, type] = collections.OrderedDict()
            for timestamp, results in test_results.monitor_results[monitor].items():
                values[timestamp] = modifier(type(results[name]))

            return values


        def plot(test: Test, test_results: TestResults, size: Tuple[int, int]):
            title = test.name
            x_label = "Timestamp"
            legend = True
            grid = True
            format_x_as_date = True
            show_final_values = True
            x_range: Tuple[Any, Any] = None
            y_range: Tuple[Any, Any] = None
            j_to_wh = lambda value: value / 3600
            to_percentage = lambda value: value * 100
            ns_to_s = lambda value: value / 1e9
            plots = {
                "Clock Rate (Hz)": {
                    ("CPU Core", ""): collect_values(test_results, "CPUMonitor", "coreClockRate", int),
                    ("GPU Core", ""): collect_values(test_results, "GPUMonitor", "coreClockRate", int),
                    ("GPU Memory", ""): collect_values(test_results, "GPUMonitor", "memoryClockRate", int),
                    ("GPU SM", ""): collect_values(test_results, "GPUMonitor", "streamingMultiprocessorClockRate", int),
                },
                "Energy Consumption (J)": {
                    ("CPU", ""): collect_values(test_results, "CPUMonitor", "energyConsumption", float),
                    ("GPU", ""): collect_values(test_results, "GPUMonitor", "energyConsumption", float),
                    ("Node", ""): collect_values(test_results, "NodeMonitor", "energyConsumption", float),
                },
                "Energy Consumption (Wh)": {
                    ("CPU", ""): collect_values(test_results, "CPUMonitor", "energyConsumption", float, j_to_wh),
                    ("GPU", ""): collect_values(test_results, "GPUMonitor", "energyConsumption", float, j_to_wh),
                    ("Node", ""): collect_values(test_results, "NodeMonitor", "energyConsumption", float, j_to_wh),
                },
                "Fan Speed (RPM)": {
                    ("GPU", ""): collect_values(test_results, "GPUMonitor", "fanSpeed", float),
                },
                "Power Consumption (W)": {
                    ("CPU", ""): collect_values(test_results, "CPUMonitor", "powerConsumption", float),
                    ("GPU", ""): collect_values(test_results, "GPUMonitor", "powerConsumption", float),
                    ("Node", ""): collect_values(test_results, "NodeMonitor", "powerConsumption", float),
                },
                "Timespan (s)": {
                    ("CPU Guest Nice", ""): collect_values(test_results, "CPUMonitor", "guestNiceTimespan", float, ns_to_s),
                    ("CPU Guest", ""): collect_values(test_results, "CPUMonitor", "guestTimespan", float, ns_to_s),
                    ("CPU IO Wait", ""): collect_values(test_results, "CPUMonitor", "ioWaitTimespan", float, ns_to_s),
                    ("CPU Idle", ""): collect_values(test_results, "CPUMonitor", "idleTimespan", float, ns_to_s),
                    ("CPU Interrupts", ""): collect_values(test_results, "CPUMonitor", "interruptsTimespan", float, ns_to_s),
                    ("CPU Nice", ""): collect_values(test_results, "CPUMonitor", "niceTimespan", float, ns_to_s),
                    ("CPU Soft Interrupts", ""): collect_values(test_results, "CPUMonitor", "softInterruptsTimespan", float, ns_to_s),
                    ("CPU Steal", ""): collect_values(test_results, "CPUMonitor", "stealTimespan", float, ns_to_s),
                    ("CPU System", ""): collect_values(test_results, "CPUMonitor", "systemTimespan", float, ns_to_s),
                    ("CPU User", ""): collect_values(test_results, "CPUMonitor", "userTimespan", float, ns_to_s),
                    ("Runtime", ""): collect_values(test_results, "NodeMonitor", "runtime", float, ns_to_s),
                },
                "Temperature (C)": {
                    ("GPU", ""): collect_values(test_results, "GPUMonitor", "temperature", float),
                },
                "Utilization Rate (%)": {
                    ("CPU Core", ""): collect_values(test_results, "CPUMonitor", "coreUtilizationRate", float, to_percentage),
                    ("GPU Core", ""): collect_values(test_results, "GPUMonitor", "coreUtilizationRate", float, to_percentage),
                    ("GPU Memory", ""): collect_values(test_results, "GPUMonitor", "memoryUtilizationRate", float, to_percentage),
                }
            }

            # Create the figure
            grid_spec = gridspec.GridSpec(len(plots) + 1, 1)
            figure = pyplot.figure(figsize=size)

            # Create the information summary
            axes = figure.add_subplot(grid_spec[0])
            axes.axis("off")
            axes.invert_yaxis()
            axes.text(0, 0, f"GPU Name = {list(collect_values(test_results, 'GPUMonitor', 'name', str).items())[0][1]}", verticalalignment="top")

            # Create the plots
            index = 1
            previous_axes = None
            for plot_name, plot_series in plots.items():
                # Create the plot
                axes = figure.add_subplot(grid_spec[index]) if previous_axes is None else figure.add_subplot(grid_spec[index], sharex=previous_axes)
                index += 1
                previous_axes = axes

                # Collect the values
                x_values = []
                y_values = []
                for _, series_values in plot_series.items():
                    for timestamp, value in series_values.items():
                        x_values.append(timestamp)
                        y_values.append(value)

                # Set the ranges
                axes.set_xlim(min(x_values) if x_range is None else x_range[0], max(x_values) if x_range is None else x_range[1])
                y_min = min(y_values)
                y_max = max(y_values)
                axes.set_ylim(((y_min - abs(0.1 * y_min)) if y_min < 0 else 0) if y_range is None else y_range[0], (y_max + abs(0.1 * y_max)) if y_range is None else y_range[1])

                # Set the data
                for (series_name, series_style), series_values in plot_series.items():
                    axes.plot(series_values.keys(), series_values.values(), series_style, label=series_name)

                    if show_final_values:
                        final_value = list(series_values.values())[-1]
                        pyplot.annotate(f"{final_value:n}", xy=(1, final_value), xytext=(8, 0), xycoords=("axes fraction", "data"), textcoords="offset points")

                # Set the labels
                axes.set(xlabel=x_label, ylabel=plot_name, title=title)

                # Format dates
                if format_x_as_date:
                    axes.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d %H:%M:%S"))
                    figure.autofmt_xdate()
                    # pyplot.xticks(rotation=45)

                # Enable grid
                if grid:
                    axes.grid()

                # Enable legend
                if legend:
                    pyplot.legend()

                # Make everything fit
                pyplot.tight_layout()

            figure.savefig(f"{output_directory}/{title}.png")


        # Plot some graphs
        # TODO: Add per-core series
        plot(test, test.test_results, (10, 30))
