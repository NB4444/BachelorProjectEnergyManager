# TODO: Fix function and graph names so that they represent x value type, then y value type and then z value type and not in the random orders they are now
import collections
from functools import cached_property
from typing import List, OrderedDict

from matplotlib import pyplot
from numpy import array, mean, median

from Visualizer.Monitoring.Persistence.ProfilerSession import ProfilerSession
from Visualizer.Plotting.HistogramPlot import HistogramPlot
from Visualizer.Plotting.Plot import Plot
from Visualizer.Plotting.ScatterPlot import ScatterPlot
from Visualizer.Plotting.TablePlot import TablePlot
from Visualizer.Utility import Containers


class DataSet(object):
    def __init__(self, data: List[ProfilerSession]):
        self.data = data

    @cached_property
    def table_plot(self):
        return TablePlot(title="Profiler Sessions",
                         table=[[profiler_session.id, profiler_session.label, profiler_session.profile] for
                                profiler_session in self.data],
                         columns=["ID", "Label", "Profile"])

    @cached_property
    def summary(self):
        return Containers.unique_dict_of_lists(
            Containers.merge_dictionaries([profiler_session.summary for profiler_session in self.data]))

    @cached_property
    def monitor_data(self):
        return Containers.unique_dict_of_lists(Containers.merge_dictionaries(
            [variables for profiler_session in self.data for monitor_session in profiler_session.monitor_sessions for
             _, variables in monitor_session.namespaced_monitor_data.items()]))

    def clock_rate_histogram_plots(self, bins=30, title_filter: str = ""):
        return [HistogramPlot(title=title, values=values, bins=bins, x_label="Clock Rate (Hertz)") for title, values in
                self.monitor_data.items() if
                title.endswith("coreClockRate") and title_filter in title]

    def utilization_rate_histogram_plots(self, bins=30, title_filter: str = ""):
        return [HistogramPlot(title=title, values=[Plot.to_percentage(value) for value in values], bins=bins, x_label="Utilization Rate (%)") for title, values
                in self.monitor_data.items() if
                title.endswith("coreUtilizationRate") and title_filter in title]

    def energy_consumption_histogram_plot(self, bins=30):
        return HistogramPlot(title="Energy Consumption",
                             values=[profiler_session.total_energy_consumption for profiler_session in self.data],
                             bins=bins, x_label="Energy Consumption (Joules)")

    @cached_property
    def minimum_energy_consumption_profiler_session(self):
        optimal_energy_consumption: ProfilerSession = None

        for profiler_session in self.data:
            energy_consumption = profiler_session.total_energy_consumption

            if optimal_energy_consumption is None or energy_consumption < optimal_energy_consumption.total_energy_consumption:
                optimal_energy_consumption = profiler_session

        return optimal_energy_consumption

    @cached_property
    def mean_energy_consumption(self):
        return mean(array([profiler_session.total_energy_consumption for profiler_session in self.data]))

    @cached_property
    def median_energy_consumption(self):
        return median(array([profiler_session.total_energy_consumption for profiler_session in self.data]))

    def runtime_histogram_plot(self, bins=30):
        return HistogramPlot(title="Runtime",
                             values=[Plot.ns_to_s(profiler_session.total_runtime) for profiler_session in self.data],
                             bins=bins, x_label="Runtime (Seconds)")

    @cached_property
    def minimum_runtime_profiler_session(self):
        optimal_runtime: ProfilerSession = None

        for profiler_session in self.data:
            runtime = profiler_session.total_runtime

            if optimal_runtime is None or runtime < optimal_runtime.total_runtime:
                optimal_runtime = profiler_session

        return optimal_runtime

    @cached_property
    def mean_runtime(self):
        return mean(array([profiler_session.total_runtime for profiler_session in self.data]))

    @cached_property
    def median_runtime(self):
        return median(array([profiler_session.total_runtime for profiler_session in self.data]))

    def flops_histogram_plot(self, bins=30):
        return HistogramPlot(title="FLOPs", values=[profiler_session.total_flops for profiler_session in self.data],
                             bins=bins)

    @cached_property
    def maximum_flops_profiler_session(self):
        optimal_flops: ProfilerSession = None

        for profiler_session in self.data:
            flops = profiler_session.total_flops

            if optimal_flops is None or flops > optimal_flops.total_flops:
                optimal_flops = profiler_session

        return optimal_flops

    @cached_property
    def mean_flops(self):
        return mean(array([profiler_session.total_flops for profiler_session in self.data]))

    @cached_property
    def median_flops(self):
        return median(array([profiler_session.total_flops for profiler_session in self.data]))

    def runtime_vs_energy_consumption_vs_combined_clock_rates(self, normalized=True):
        data: OrderedDict[str, OrderedDict[int, int]] = collections.OrderedDict({})
        for profiler_session in self.data:
            profile = "Runs"
            if profile not in data:
                data[profile] = collections.OrderedDict()

            if normalized:
                data[profile][
                    profiler_session.total_runtime / self.minimum_runtime_profiler_session.total_runtime * 100] = profiler_session.total_energy_consumption / self.minimum_energy_consumption_profiler_session.total_energy_consumption * 100
            else:
                data[profile][Plot.ns_to_s(profiler_session.total_runtime)] = profiler_session.total_energy_consumption

        return data

    def energy_consumption_vs_runtime_scatter_plot(self, normalized=True):
        plot_series = self.runtime_vs_energy_consumption_vs_combined_clock_rates(normalized)

        values = []
        for profiler_session in self.data:
            if "maximumCPUClockRate" in profiler_session.profile:
                values.append(
                    profiler_session.profile["maximumCPUClockRate"] + profiler_session.profile["maximumGPUClockRate"])
        max_value = max(values, default=None)
        min_value = min(values, default=None)

        return ScatterPlot(
            title="Runtime vs. Energy Consumption vs. Combined Clock Rates", plot_series=plot_series,
            x_label="Runtime (" + ("% of optimal" if normalized else "Seconds") + ")",
            y_label="Energy Consumption (" + ("% of optimal" if normalized else "Joules") + ")",
            colors=[pyplot.get_cmap("gist_rainbow")((value - min_value) / (max_value - min_value)) for value in
                    values] if len(values) > 0 else None,
            labels=[profiler_session.plot_label for profiler_session in self.data]
        )

    def energy_consumption_vs_flops(self, normalized=True):
        data: OrderedDict[str, OrderedDict[int, int]] = collections.OrderedDict({})
        for profiler_session in self.data:
            profile = "Runs"
            if profile not in data:
                data[profile] = collections.OrderedDict()

            if normalized:
                data[profile][
                    profiler_session.total_flops / self.maximum_flops_profiler_session.total_flops * 100] = profiler_session.total_energy_consumption / self.minimum_energy_consumption_profiler_session.total_energy_consumption * 100
            else:
                data[profile][profiler_session.total_flops] = profiler_session.total_energy_consumption

        return data

    def energy_consumption_vs_flops_scatter_plot(self, normalized=True):
        plot_series = self.energy_consumption_vs_flops(normalized)

        values = []
        for profiler_session in self.data:
            if "maximumCPUClockRate" in profiler_session.profile:
                values.append(
                    profiler_session.profile["maximumCPUClockRate"] + profiler_session.profile["maximumGPUClockRate"])
        max_value = max(values)
        min_value = min(values)

        return ScatterPlot(
            title="Energy Consumption vs. FLOPs", plot_series=plot_series,
            x_label="FLOPs (" + ("% of optimal" if normalized else "Operations") + ")",
            y_label="Energy Consumption (" + ("% of optimal" if normalized else "Joules") + ")",
            colors=[pyplot.get_cmap("gist_rainbow")((value - min_value) / (max_value - min_value)) for value in
                    values] if len(values) > 0 else None,
            labels=[profiler_session.plot_label for profiler_session in self.data]
        )

    @cached_property
    def core_clock_rate_vs_gpu_clock_rate_vs_energy_consumption(self):
        data: OrderedDict[str, OrderedDict[int, OrderedDict[int, int]]] = collections.OrderedDict({})
        for profiler_session in self.data:
            profile = "Runs"
            if profile not in data:
                data[profile] = collections.OrderedDict()

            core_clock_rate = profiler_session.profile["maximumCPUClockRate"]
            if core_clock_rate not in data[profile]:
                data[profile][core_clock_rate] = collections.OrderedDict()

            gpu_clock_rate = profiler_session.profile["maximumGPUClockRate"]
            data[profile][core_clock_rate][gpu_clock_rate] = profiler_session.total_energy_consumption

        return data

    @cached_property
    def core_clock_rate_vs_gpu_clock_rate_vs_energy_consumption_scatter_plot(self):
        plot_series = self.core_clock_rate_vs_gpu_clock_rate_vs_energy_consumption

        values = []
        for series_name, x_y_z_values in plot_series.items():
            for x_value, y_z_values in x_y_z_values.items():
                for y_value, z_value in y_z_values.items():
                    values.append(x_value + y_value)
        max_value = max(values, default=None)
        min_value = min(values, default=None)

        return ScatterPlot(
            title="Core Frequency vs. GPU Frequency vs. Energy Consumption",
            plot_series=plot_series,
            x_label="Core Clock Rate (Hertz)", y_label="GPU Clock Rate (Hertz)",
            z_label="Energy Consumption (Joules)",
            colors=[pyplot.get_cmap("gist_rainbow")((value - min_value) / (max_value - min_value)) for value in
                    values] if len(values) > 0 else None,
            labels=[profiler_session.plot_label for profiler_session in self.data]
        )

    @cached_property
    def core_clock_rate_vs_gpu_clock_rate_vs_runtime(self):
        data: OrderedDict[str, OrderedDict[int, OrderedDict[int, int]]] = collections.OrderedDict({})
        for profiler_session in self.data:
            profile = "Runs"
            if profile not in data:
                data[profile] = collections.OrderedDict()

            core_clock_rate = profiler_session.profile["maximumCPUClockRate"]
            if core_clock_rate not in data[profile]:
                data[profile][core_clock_rate] = collections.OrderedDict()

            gpu_clock_rate = profiler_session.profile["maximumGPUClockRate"]
            data[profile][core_clock_rate][gpu_clock_rate] = Plot.ns_to_s(profiler_session.total_runtime)

        return data

    @cached_property
    def core_clock_rate_vs_gpu_clock_rate_vs_runtime_scatter_plot(self):
        plot_series = self.core_clock_rate_vs_gpu_clock_rate_vs_runtime

        values = []
        for series_name, x_y_z_values in plot_series.items():
            for x_value, y_z_values in x_y_z_values.items():
                for y_value, z_value in y_z_values.items():
                    values.append(x_value + y_value)
        max_value = max(values, default=None)
        min_value = min(values, default=None)

        return ScatterPlot(
            title="Core Frequency vs. GPU Frequency vs. Runtime", plot_series=plot_series,
            x_label="Core Clock Rate (Hertz)",
            y_label="GPU Clock Rate (Hertz)", z_label="Runtime (Seconds)",
            colors=[pyplot.get_cmap("gist_rainbow")((value - min_value) / (max_value - min_value)) for value in
                    values] if len(values) > 0 else None,
            labels=[profiler_session.plot_label for profiler_session in self.data]
        )
