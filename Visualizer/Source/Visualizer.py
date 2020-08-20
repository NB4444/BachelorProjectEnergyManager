import argparse
import collections
import os
import shutil
from typing import Dict, OrderedDict, Any, Tuple

from matplotlib import pyplot, dates

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

    for test_results in TestResults.load_all(arguments.database):
        # Determine the output directory for the current test
        output_directory = f"{arguments.output_directory}/{test_results.test}"

        # Make sure the directory exists and is empty
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        os.makedirs(output_directory)


        def plot(title: str, x_label: str, y_label: str, plots: Dict[Tuple[str, str], OrderedDict[int, Any]], legend: bool = True, grid: bool = True, format_x_as_date: bool = True, show_final_values: bool = True, x_range: Tuple[Any, Any] = None, y_range: Tuple[Any, Any] = None):
            # Set the ranges
            x_values = [x for x_values in [plots[plot].keys() for plot in plots] for x in x_values]
            y_values = [y for y_values in [plots[plot].values() for plot in plots] for y in y_values]
            pyplot.xlim(min(x_values) if x_range is None else x_range[0], max(x_values) if x_range is None else x_range[1])
            pyplot.ylim(min(y_values) if y_range is None else y_range[0], max(y_values) if y_range is None else y_range[1])

            # Create the plot
            figure, axes = pyplot.subplots()

            # Set the data
            for (plot_name, plot_style), plot_values in plots.items():
                axes.plot(plot_values.keys(), plot_values.values(), plot_style, label=plot_name)

                if show_final_values:
                    final_value = list(plot_values.values())[-1]
                    pyplot.annotate(f"{final_value:n}", xy=(1, final_value), xytext=(8, 0), xycoords=("axes fraction", "data"), textcoords="offset points")

            # Set the labels
            axes.set(xlabel=x_label, ylabel=y_label, title=title)

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


        def collect_values(test_results: TestResults, monitor: str, name: str, type, modifier=lambda value: value):
            values: OrderedDict[int, type] = collections.OrderedDict()
            for timestamp, results in test_results.monitor_results[monitor].items():
                values[timestamp] = modifier(type(results[name]))

            return values


        def plot_core_clock_rates(test_results: TestResults):
            plot("Core Clock Rates (Hz)", "Timestamp", "Core Clock Rate (Hz)", {
                ("CPU", "b-x"): collect_values(test_results, "CPUMonitor", "coreClockRate", int),
                ("GPU", "g-+"): collect_values(test_results, "GPUMonitor", "coreClockRate", int)
            })


        def plot_energy_consumption(test_results: TestResults):
            plot("Energy Consumption (J)", "Timestamp", "Energy Consumption (J)", {
                ("Node", "r-o"): collect_values(test_results, "NodeMonitor", "energyConsumption", float),
                ("CPU", "b-x"): collect_values(test_results, "CPUMonitor", "energyConsumption", float),
                ("GPU", "g-+"): collect_values(test_results, "GPUMonitor", "energyConsumption", float)
            })

            plot("Energy Consumption (Wh)", "Timestamp", "Energy Consumption (Wh)", {
                ("Node", "r-o"): collect_values(test_results, "NodeMonitor", "energyConsumption", float, lambda value: value / 3600),
                ("CPU", "b-x"): collect_values(test_results, "CPUMonitor", "energyConsumption", float, lambda value: value / 3600),
                ("GPU", "g-+"): collect_values(test_results, "GPUMonitor", "energyConsumption", float, lambda value: value / 3600)
            })


        def plot_power_consumption(test_results: TestResults):
            plot("Power Consumption (W)", "Timestamp", "Power Consumption (W)", {
                ("Node", "r-o"): collect_values(test_results, "NodeMonitor", "powerConsumption", float),
                ("CPU", "b-x"): collect_values(test_results, "CPUMonitor", "powerConsumption", float),
                ("GPU", "g-+"): collect_values(test_results, "GPUMonitor", "powerConsumption", float)
            })


        def plot_runtime(test_results: TestResults):
            plot("Runtime (s)", "Timestamp", "Runtime (s)", {
                ("Runtime", "b-x"): collect_values(test_results, "NodeMonitor", "runtime", float)
            })


        def plot_utilization_rates(test_results: TestResults):
            plot("Utilization Rates (%)", "Timestamp", "Utilization Rate (%)", {
                ("GPU Core", "g-x"): collect_values(test_results, "GPUMonitor", "coreUtilizationRate", int),
                ("GPU Memory", "g-o"): collect_values(test_results, "GPUMonitor", "memoryUtilizationRate", int)
            }, True, True, True, True, None, (0, 100))


        # Plot some graphs
        plot_core_clock_rates(test_results)
        plot_energy_consumption(test_results)
        plot_power_consumption(test_results)
        plot_runtime(test_results)
        plot_utilization_rates(test_results)
