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


        def plot(title: str, x_label: str, y_label: str, plots: Dict[str, OrderedDict[int, Any]], legend: bool = True, grid: bool = True, format_x_as_date: bool = True, show_final_values: bool = True, x_range: Tuple[Any, Any] = None, y_range: Tuple[Any, Any] = None):
            # Set the ranges
            x_values = [x for x_values in [plots[plot].keys() for plot in plots] for x in x_values]
            y_values = [y for y_values in [plots[plot].values() for plot in plots] for y in y_values]
            pyplot.xlim(min(x_values) if x_range is None else x_range[0], max(x_values) if x_range is None else x_range[1])
            pyplot.ylim(min(y_values) if y_range is None else y_range[0], max(y_values) if y_range is None else y_range[1])

            # Create the plot
            figure, axes = pyplot.subplots()

            # Set the data
            for plot_name, plot_values in plots.items():
                axes.plot(plot_values.keys(), plot_values.values(), label=plot_name)

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


        def plot_core_clock_rates(test_results: TestResults):
            cpu_core_clock_rates: OrderedDict[int, int] = collections.OrderedDict()
            for timestamp, results in test_results.monitor_results["CPUMonitor"].items():
                cpu_core_clock_rates[timestamp] = int(results["coreClockRate"]);

            gpu_core_clock_rates: OrderedDict[int, int] = collections.OrderedDict()
            for timestamp, results in test_results.monitor_results["GPUMonitor"].items():
                gpu_core_clock_rates[timestamp] = int(results["coreClockRate"]);

            plot("Core Clock Rates", "Timestamp", "Core Clock Rate (Hz)", {
                "CPU": cpu_core_clock_rates,
                "GPU": gpu_core_clock_rates
            })


        def plot_power_consumption(test_results: TestResults):
            gpu_power_consumption: OrderedDict[int, int] = collections.OrderedDict()
            for timestamp, results in test_results.monitor_results["GPUMonitor"].items():
                gpu_power_consumption[timestamp] = float(results["powerConsumption"]);

            plot("Power Consumption", "Timestamp", "Power Consumption (W)", {
                "GPU": gpu_power_consumption,
            })


        def plot_runtime(test_results: TestResults):
            gpu_runtime: OrderedDict[int, int] = collections.OrderedDict()
            for timestamp, results in test_results.monitor_results["CPUMonitor"].items():
                gpu_runtime[timestamp] = int(results["runtime"]);

            plot("Runtime", "Timestamp", "Runtime (s)", {
                "Runtime": gpu_runtime,
            })


        def plot_total_power_consumption(test_results: TestResults):
            gpu_total_power_consumption: OrderedDict[int, float] = collections.OrderedDict()
            for timestamp, results in test_results.monitor_results["GPUMonitor"].items():
                gpu_total_power_consumption[timestamp] = float(results["totalPowerConsumption"]);

            plot("Total Power Consumption", "Timestamp", "Total Power Consumption (J)", {
                "GPU": gpu_total_power_consumption,
            })


        def plot_utilization_rates(test_results: TestResults):
            gpu_core_utilization_rate: OrderedDict[int, int] = collections.OrderedDict()
            gpu_memory_utilization_rate: OrderedDict[int, int] = collections.OrderedDict()
            for timestamp, results in test_results.monitor_results["GPUMonitor"].items():
                gpu_core_utilization_rate[timestamp] = int(results["coreUtilizationRate"]);
                gpu_memory_utilization_rate[timestamp] = int(results["memoryUtilizationRate"]);

            plot("Utilization Rates", "Timestamp", "Utilization Rate(%)", {
                "GPU Core": gpu_core_utilization_rate,
                "GPU Memory": gpu_memory_utilization_rate
            }, True, True, True, True, None, (0, 100))


        # Plot some graphs
        plot_core_clock_rates(test_results)
        plot_power_consumption(test_results)
        plot_runtime(test_results)
        plot_total_power_consumption(test_results)
        plot_utilization_rates(test_results)
