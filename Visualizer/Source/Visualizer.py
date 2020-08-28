import argparse
import collections
import math
import multiprocessing
import os
import shutil
from typing import OrderedDict, Any, Tuple

import cycler
import humanize
import matplotlib
from matplotlib import pyplot, dates, gridspec

from Visualizer.Testing.Test import Test
from Visualizer.Testing.TestResults import TestResults


def parse_arguments():
    parser = argparse.ArgumentParser(prog="Visualizer", description="Visualize EnergyManager test results.")

    parser.add_argument(
        "--update",
        "-u",
        action="store_true",
        help="Updates existing results (may take a long time).",
        required=False
    )

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


def collect_values(test_results: TestResults, monitor: str, name: str, type, modifier=lambda value: value):
    values: OrderedDict[int, type] = collections.OrderedDict()
    try:
        for timestamp, results in test_results.monitor_results[monitor].items():
            try:
                values[timestamp] = modifier(type(results[name]))
            except:
                pass
    except:
        pass

    return values


def collect_indexed_values(test_results: TestResults, monitor: str, name: str, type, modifier=lambda value: value):
    index = 0

    results = list()

    while True:
        values = collect_values(test_results, monitor, name + str(index), type, modifier)

        if len(values) == 0:
            break
        else:
            results.append(values)
            index += 1

    return results


def collect_indexed_series(test_results: TestResults, series_name_prefix: str, monitor: str, name: str, type, modifier=lambda value: value):
    series = {}

    for index, values in enumerate(collect_indexed_values(test_results, monitor, name, type, modifier)):
        series[(f"{series_name_prefix} {index}", "")] = values

    return series


def collect_summarized_indexed_values(test_results: TestResults, summarized_series_name: str, series_name_prefix: str, monitor: str, summarized_name: str, name: str, type, modifier=lambda value: value):
    series = {(f"{summarized_series_name}", ""): collect_values(test_results, monitor, summarized_name, type, modifier)}
    series.update(collect_indexed_series(test_results, series_name_prefix, monitor, name, type, modifier))

    return series


def collect_constant_value(test_results: TestResults, monitor: str, name: str, type, modifier=lambda value: value):
    values = collect_values(test_results, monitor, name, type, modifier).items()
    return list(values)[-1][1] if len(values) > 0 else None


def determine_extent(elements, padding=(0.0, 0.0)):
    return matplotlib.transforms.Bbox.union([element.get_window_extent() for element in elements]).expanded(1.0 + padding[0], 1.0 + padding[1])


def plot_test(test: Test, test_results: TestResults, size: Tuple[int, int], output_directory: str):
    title = test.name
    x_label = "Timestamp"
    legend = True
    grid = True
    format_x_as_date = False
    show_final_values = True
    x_range: Tuple[Any, Any] = None
    y_range: Tuple[Any, Any] = None

    # Define conversion functions
    j_to_wh = lambda value: value / 3600
    to_percentage = lambda value: value * 100
    ns_to_s = lambda value: value / 1e9
    humanize_number = lambda value: humanize.intcomma(value)
    humanize_size = lambda value: humanize.naturalsize(value)
    parse_number_list = lambda value: [number for number in value.split(",")]

    summary = {
        "GPU Brand": collect_constant_value(test_results, "GPUMonitor", "brand", str),
        "GPU Compute Capability Major Version": collect_constant_value(test_results, "GPUMonitor", "computeCapabilityMajorVersion", int),
        "GPU Compute Capability Minor Version": collect_constant_value(test_results, "GPUMonitor", "computeCapabilityMinorVersion", int),
        # "GPU Memory Bandwidth (B/s)": collect_constant_value(test_results, "GPUMonitor", "memoryBandwidth", float, humanize_size),
        "GPU Memory Size (B)": collect_constant_value(test_results, "GPUMonitor", "memorySize", int, humanize_size),
        # "GPU Multiprocessor Count": collect_constant_value(test_results, "GPUMonitor", "multiprocessorCount", int, humanize_number),
        "GPU Name": collect_constant_value(test_results, "GPUMonitor", "name", str),
        "GPU PCIe Link Width (B)": collect_constant_value(test_results, "GPUMonitor", "pciELinkWidth", int, humanize_size),
        "GPU Default Power Limit (W)": collect_constant_value(test_results, "GPUMonitor", "defaultPowerLimit", float, humanize_number),
        "GPU Supported Core Clock Rates (Hz)": collect_constant_value(test_results, "GPUMonitor", "supportedCoreClockRates", str, parse_number_list),
        "GPU Supported Memory Clock Rates (Hz)": collect_constant_value(test_results, "GPUMonitor", "supportedMemoryClockRates", str, parse_number_list),
        "GPU Default Auto Boosted Clocks Enabled": collect_constant_value(test_results, "GPUMonitor", "defaultAutoBoostedClocksEnabled", bool),
    }

    clock_rate_plot = collect_summarized_indexed_values(test_results, "CPU", "CPU Core", "CPUMonitor", "coreClockRate", "coreClockRateCore", int)
    clock_rate_plot.update(collect_summarized_indexed_values(test_results, "CPU Maximum", "CPU Maximum Core", "CPUMonitor", "maximumCoreClockRate", "maximumCoreClockRateCore", int))
    clock_rate_plot.update({
        ("GPU Core", ""): collect_values(test_results, "GPUMonitor", "coreClockRate", int),
        # ("GPU Core Maximum", ""): collect_values(test_results, "GPUMonitor", "maximumCoreClockRate", int),
        ("GPU Memory", ""): collect_values(test_results, "GPUMonitor", "memoryClockRate", int),
        # ("GPU Memory Maximum", ""): collect_values(test_results, "GPUMonitor", "maximumMemoryClockRate", int),
        ("GPU SM", ""): collect_values(test_results, "GPUMonitor", "streamingMultiprocessorClockRate", int),
    })

    energy_consumption_j_plot = collections.OrderedDict({
        ("CPU", ""): collect_values(test_results, "CPUMonitor", "energyConsumption", float),
        ("GPU", ""): collect_values(test_results, "GPUMonitor", "energyConsumption", float),
        ("Node", ""): collect_values(test_results, "NodeMonitor", "energyConsumption", float),
    })

    energy_consumption_wh_plot = {
        ("CPU", ""): collect_values(test_results, "CPUMonitor", "energyConsumption", float, j_to_wh),
        ("GPU", ""): collect_values(test_results, "GPUMonitor", "energyConsumption", float, j_to_wh),
        ("Node", ""): collect_values(test_results, "NodeMonitor", "energyConsumption", float, j_to_wh),
    }

    fan_speed_plot = collect_summarized_indexed_values(test_results, "GPU", "GPU Fan", "GPUMonitor", "fanSpeed", "fanSpeedFan", float, to_percentage)

    memory_consumption_plot = {
        ("GPU Kernel Dynamic Shared", ""): collect_values(test_results, "GPUMonitor", "kernelDynamicSharedMemorySize", int),
        ("GPU Kernel Static Shared", ""): collect_values(test_results, "GPUMonitor", "kernelStaticSharedMemorySize", int),
        # ("GPU", ""): collect_values(test_results, "GPUMonitor", "memorySize", int),
        # ("GPU Free", ""): collect_values(test_results, "GPUMonitor", "memoryFreeSize", int),
        ("GPU Used", ""): collect_values(test_results, "GPUMonitor", "memoryUsedSize", int),
        ("GPU PCIe Link", ""): collect_values(test_results, "GPUMonitor", "pciELinkWidth", int),
        # ("RAM", ""): collect_values(test_results, "NodeMonitor", "memorySize", int),
        # ("RAM Free", ""): collect_values(test_results, "NodeMonitor", "freeMemorySize", int),
        ("RAM Used", ""): collect_values(test_results, "NodeMonitor", "usedMemorySize", int),
        ("RAM Shared", ""): collect_values(test_results, "NodeMonitor", "sharedMemorySize", int),
        ("RAM Buffer", ""): collect_values(test_results, "NodeMonitor", "bufferMemorySize", int),
        # ("Swap", ""): collect_values(test_results, "NodeMonitor", "swapMemorySize", int),
        # ("Swap Free", ""): collect_values(test_results, "NodeMonitor", "freeSwapMemorySize", int),
        ("Swap Used", ""): collect_values(test_results, "NodeMonitor", "usedSwapMemorySize", int),
        # ("High", ""): collect_values(test_results, "NodeMonitor", "highMemorySize", int),
        # ("High Free", ""): collect_values(test_results, "NodeMonitor", "freeHighMemorySize", int),
        ("High Used", ""): collect_values(test_results, "NodeMonitor", "usedHighMemorySize", int),
    }

    power_consumption_plot = {
        ("CPU", ""): collect_values(test_results, "CPUMonitor", "powerConsumption", float),
        ("GPU", ""): collect_values(test_results, "GPUMonitor", "powerConsumption", float),
        ("GPU Power Limit", ""): collect_values(test_results, "GPUMonitor", "powerLimit", float),
        ("GPU Enforced Power Limit", ""): collect_values(test_results, "GPUMonitor", "powerLimit", float),
        ("Node", ""): collect_values(test_results, "NodeMonitor", "powerConsumption", float),
    }

    processes_plot = {
        ("Processes", ""): collect_values(test_results, "NodeMonitor", "processCount", int)
    }

    switches_plot = {
        ("GPU Auto Boosted Clocks", ""): collect_values(test_results, "GPUMonitor", "autoBoostedClocksEnabled", bool),
    }

    temperature_plot = {
        ("CPU", ""): collect_values(test_results, "CPUMonitor", "temperature", float),
        ("GPU", ""): collect_values(test_results, "GPUMonitor", "temperature", float),
    }

    timespan_plot = collect_summarized_indexed_values(test_results, "CPU Guest Nice", "CPU Guest Nice Core", "CPUMonitor", "guestNiceTimespan", "guestNiceTimespanCore", float, to_percentage)
    timespan_plot.update(collect_summarized_indexed_values(test_results, "CPU Guest", "CPU Guest Core", "CPUMonitor", "guestTimespan", "guestTimespanCore", float, to_percentage))
    timespan_plot.update(collect_summarized_indexed_values(test_results, "CPU IO Wait", "CPU IO Wait Core", "CPUMonitor", "ioWaitTimespan", "ioWaitTimespanCore", float, to_percentage))
    timespan_plot.update(collect_summarized_indexed_values(test_results, "CPU Idle", "CPU Idle Core", "CPUMonitor", "idleTimespan", "idleTimespanCore", float, to_percentage))
    timespan_plot.update(collect_summarized_indexed_values(test_results, "CPU Interrupts", "CPU Interrupts Core", "CPUMonitor", "interruptsTimespan", "interruptsTimespanCore", float, to_percentage))
    timespan_plot.update(collect_summarized_indexed_values(test_results, "CPU Nice", "CPU Nice Core", "CPUMonitor", "niceTimespan", "niceTimespanCore", float, to_percentage))
    timespan_plot.update(collect_summarized_indexed_values(test_results, "CPU Soft Interrupts", "CPU Soft Interrupts Core", "CPUMonitor", "softInterruptsTimespan", "softInterruptsTimespanCore", float, to_percentage))
    timespan_plot.update(collect_summarized_indexed_values(test_results, "CPU Steal", "CPU Steal Core", "CPUMonitor", "stealTimespan", "stealTimespanCore", float, to_percentage))
    timespan_plot.update(collect_summarized_indexed_values(test_results, "CPU System", "CPU System Core", "CPUMonitor", "systemTimespan", "systemTimespanCore", float, to_percentage))
    timespan_plot.update(collect_summarized_indexed_values(test_results, "CPU User", "CPU User Core", "CPUMonitor", "userTimespan", "userTimespanCore", float, to_percentage))
    timespan_plot[("Runtime", "")] = collect_values(test_results, "NodeMonitor", "runtime", float, ns_to_s)

    utilization_rate_plot = collect_summarized_indexed_values(test_results, "CPU", "CPU Core", "CPUMonitor", "coreUtilizationRate", "coreUtilizationRateCore", float, to_percentage)
    utilization_rate_plot.update({
        ("GPU Core", ""): collect_values(test_results, "GPUMonitor", "coreUtilizationRate", float, to_percentage),
        ("GPU Memory", ""): collect_values(test_results, "GPUMonitor", "memoryUtilizationRate", float, to_percentage),
    })

    plots = {
        "Clock Rate (Hz)": clock_rate_plot,
        "Energy Consumption (J)": energy_consumption_j_plot,
        "Energy Consumption (Wh)": energy_consumption_wh_plot,
        "Fan Speed (%)": fan_speed_plot,
        "Memory Consumption (B)": memory_consumption_plot,
        "Power Consumption (W)": power_consumption_plot,
        "Processes": processes_plot,
        "Switches (Boolean)": switches_plot,
        "Temperature (C)": temperature_plot,
        "Timespan (s)": timespan_plot,
        "Utilization Rate (%)": utilization_rate_plot,
    }

    # Create the figure
    grid_spec = gridspec.GridSpec(len(plots) + 1, 1)
    figure = pyplot.figure(figsize=size)

    # Create the information summary
    axes = figure.add_subplot(grid_spec[0])
    axes.axis("off")
    axes.invert_yaxis()
    summary_text = ""
    for name, value in sorted(summary.items()):
        summary_text += "{0:>100}  {1:<50}\n".format(str(name), str(value))
    axes.text(0, 0, summary_text, verticalalignment="top", fontfamily="monospace")

    # Create style cycler
    # style_cycler = (cycler.cycler(color=list("rgb")) + cycler.cycler(linestyle=['-', '--', '-.']))

    # Create the plots
    index = 1
    previous_axes = None
    for plot_name, plot_series_unordered in plots.items():
        # Sort the labels for the legend
        plot_series = collections.OrderedDict(sorted(plot_series_unordered.items()))

        # Create the plot
        axes = figure.add_subplot(grid_spec[index]) if previous_axes is None else figure.add_subplot(grid_spec[index], sharex=previous_axes)
        index += 1
        previous_axes = axes

        # Cycle line styles
        # axes.set_prop_cycle(style_cycler)

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
        axes.set_xlim(x_min if x_range is None else x_range[0], x_max if x_range is None else x_range[1])
        y_min = min(y_values) if len(y_values) > 0 else 0
        y_max = max(y_values) if len(y_values) > 0 else 0
        axes.set_ylim(((y_min - abs(0.1 * y_min)) if y_min < 0 else 0) if y_range is None else y_range[0], (y_max + abs(0.1 * y_max)) if y_range is None else y_range[1])

        # Set the data
        annotations = list()
        for (series_name, series_style), series_values in plot_series.items():
            if len(series_values) > 0:
                axes.plot(series_values.keys(), series_values.values(), series_style, label=series_name)

                if show_final_values and len(series_values.values()) > 0:
                    final_value = list(series_values.values())[-1]
                    annotations.append(pyplot.annotate(f"{humanize_number(final_value)}", xy=(1, final_value), xytext=(8, 0), xycoords=("axes fraction", "data"), textcoords="offset points"))

        # Set the labels
        axes.set(xlabel=x_label, ylabel=plot_name, title=title)

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

        axes.figure.canvas.draw()
        extent = determine_extent(axes.get_xticklabels() + axes.get_yticklabels() + annotations + [axes, axes.title, axes.get_xaxis().get_label(), axes.get_yaxis().get_label(), axes.get_legend()]).transformed(figure.dpi_scale_trans.inverted())

        # Save the subplot
        figure.savefig(f"{output_directory}/{plot_name}.png", bbox_inches=extent)

    # Make everything fit
    pyplot.tight_layout()

    # Save the summary
    figure.savefig(f"{output_directory}/Summary.png")

    # Free up memory
    pyplot.draw()
    pyplot.clf()
    pyplot.close("all")


if __name__ == '__main__':
    arguments = parse_arguments()

    # Load the tests
    tests = Test.load_all(arguments.database)

    # Set up statistics variable
    processed_tests = multiprocessing.Value("i", 0)


    def initialize(arguments):
        global processed_tests
        processed_tests = arguments


    def process_test(test: Test, test_count: int, output_directory: str, update: bool):
        global processed_tests

        with processed_tests.get_lock():
            processed_tests.value += 1

            print(f"Processing test {processed_tests.value}/{test_count}: {test.id} - {test.name}...")

        # Determine the output directory for the current test
        test_output_directory = f"{output_directory}/{test.id} - {test.name}"

        # Check if the results already exists
        if os.path.exists(test_output_directory):
            # If so, check if we need to update them
            if update:
                print(f"Results exist, updating...")

                # Delete the old results
                shutil.rmtree(test_output_directory)
            else:
                print(f"Results exist, skipping...")

                # Go to the next test
                return

        # Create a directory to hold the results
        os.makedirs(test_output_directory)

        # Plot some graphs
        plot_test(test, test.test_results, (30, 30), test_output_directory)


    # Create a thread pool and process all threads
    with multiprocessing.Pool(initializer=initialize, initargs=(processed_tests,)) as pool:
        pool.starmap(process_test, [(test, len(tests), arguments.output_directory, arguments.update) for test in tests], chunksize=1)
