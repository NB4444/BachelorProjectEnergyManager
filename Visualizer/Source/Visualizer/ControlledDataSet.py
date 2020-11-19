import collections
from enum import Enum
from typing import Any, OrderedDict

from matplotlib import pyplot

from Visualizer.DataSet import DataSet
from Visualizer.Plotting.Plot import Plot
from Visualizer.Plotting.ScatterPlot import ScatterPlot


class ControlComparison(Enum):
    MEAN = 0
    MEDIAN = 1
    OPTIMAL = 2


class ControlledDataSet(object):
    def __init__(self, data_set: DataSet, control_data_set: DataSet):
        self.data_set = data_set
        self.control_data_set = control_data_set

    def energy_savings_vs_runtime_increase(self, control_comparison=ControlComparison.MEAN, normalized=True):
        control_energy_consumption = None
        control_runtime = None
        if control_comparison == ControlComparison.MEAN:
            control_energy_consumption = self.control_data_set.mean_energy_consumption
            control_runtime = self.control_data_set.mean_runtime
        elif control_comparison == ControlComparison.MEDIAN:
            control_energy_consumption = self.control_data_set.median_energy_consumption
            control_runtime = self.control_data_set.median_runtime
        elif control_comparison == ControlComparison.OPTIMAL:
            control_energy_consumption = self.control_data_set.minimum_energy_consumption_profiler_session.total_energy_consumption
            control_runtime = self.control_data_set.minimum_runtime_profiler_session.total_runtime

        data: OrderedDict[str, OrderedDict[Any, Any]] = collections.OrderedDict({})
        for profiler_session in self.data_set.data:
            energy_savings = control_energy_consumption - profiler_session.total_energy_consumption
            runtime_increase = profiler_session.total_runtime - control_runtime

            profile = "Runs"
            if profile not in data:
                data[profile] = collections.OrderedDict()

            if normalized:
                data[profile][
                    runtime_increase / control_runtime * 100] = energy_savings / control_energy_consumption * 100
            else:
                data[profile][Plot.ns_to_s(runtime_increase)] = energy_savings

        return data

    def energy_savings_vs_runtime_increase_plot(self, control_comparison=ControlComparison.MEAN, normalized=True):
        plot_series = self.energy_savings_vs_runtime_increase(control_comparison, normalized)

        values = []
        for profiler_session in self.data_set.data:
            if "maximumCPUClockRate" in profiler_session.profile:
                values.append(
                    profiler_session.profile["maximumCPUClockRate"] + profiler_session.profile["maximumGPUClockRate"])
        max_value = max(values, default=None)
        min_value = min(values, default=None)

        return ScatterPlot(
            title="Energy Savings vs. Runtime Increase", plot_series=plot_series,
            x_label="Runtime Increase (" + ("% of optimal" if normalized else "Seconds") + ")",
            y_label="Energy Savings (" + ("% of optimal" if normalized else "Joules") + ")",
            colors=[pyplot.get_cmap("gist_rainbow")((value - min_value) / (max_value - min_value)) for value in
                    values] if len(values) > 0 else None,
            labels=[profiler_session.plot_label for profiler_session in self.data_set.data]
        )

    def energy_savings_vs_flops_decrease(self, control_comparison=ControlComparison.MEAN, normalized=True):
        control_energy_consumption = None
        control_flops = None
        if control_comparison == ControlComparison.MEAN:
            control_energy_consumption = self.control_data_set.mean_energy_consumption
            control_flops = self.control_data_set.mean_flops
        elif control_comparison == ControlComparison.MEDIAN:
            control_energy_consumption = self.control_data_set.median_energy_consumption
            control_flops = self.control_data_set.median_flops
        elif control_comparison == ControlComparison.OPTIMAL:
            control_energy_consumption = self.control_data_set.minimum_energy_consumption_profiler_session.total_energy_consumption
            control_flops = self.control_data_set.maximum_flops_profiler_session.total_flops

        data: OrderedDict[str, OrderedDict[Any, Any]] = collections.OrderedDict({})
        for profiler_session in self.data_set.data:
            energy_savings = control_energy_consumption - profiler_session.total_energy_consumption
            flops_decrease = control_flops - profiler_session.total_flops

            profile = "Runs"
            if profile not in data:
                data[profile] = collections.OrderedDict()

            if normalized:
                data[profile][flops_decrease / control_flops * 100] = energy_savings / control_energy_consumption * 100
            else:
                data[profile][flops_decrease] = energy_savings

        return data

    def energy_savings_vs_flops_decrease_plot(self, control_comparison=ControlComparison.MEAN, normalized=True):
        plot_series = self.energy_savings_vs_flops_decrease(control_comparison, normalized)

        values = []
        for profiler_session in self.data_set.data:
            if "maximumCPUClockRate" in profiler_session.profile:
                values.append(
                    profiler_session.profile["maximumCPUClockRate"] + profiler_session.profile["maximumGPUClockRate"])
        max_value = max(values, default=None)
        min_value = min(values, default=None)

        return ScatterPlot(
            title="Energy Savings vs. FLOPs Decrease", plot_series=plot_series,
            x_label="FLOPs Decrease (" + ("% of optimal" if normalized else "Operations") + ")",
            y_label="Energy Savings (" + ("% of optimal" if normalized else "Joules") + ")",
            colors=[pyplot.get_cmap("gist_rainbow")((value - min_value) / (max_value - min_value)) for value in
                    values] if len(values) > 0 else None,
            labels=[profiler_session.plot_label for profiler_session in self.data_set.data]
        )

    def core_clock_rate_vs_gpu_clock_rate_vs_energy_savings(self, control_comparison=ControlComparison.MEAN):
        control_energy_consumption = None
        if control_comparison == ControlComparison.MEAN:
            control_energy_consumption = self.control_data_set.mean_energy_consumption
        elif control_comparison == ControlComparison.MEDIAN:
            control_energy_consumption = self.control_data_set.median_energy_consumption
        elif control_comparison == ControlComparison.OPTIMAL:
            control_energy_consumption = self.control_data_set.minimum_energy_consumption_profiler_session.total_energy_consumption

        data: OrderedDict[
            str, OrderedDict[int, OrderedDict[int, int]]] = collections.OrderedDict({})
        for profiler_session in self.data_set.data:
            savings = control_energy_consumption - profiler_session.total_energy_consumption

            profile = "Saves Energy" if savings >= 0 else "Costs Energy"
            if profile not in data:
                data[profile] = collections.OrderedDict()

            core_clock_rate = profiler_session.profile["maximumCPUClockRate"]
            if core_clock_rate not in data[profile]:
                data[profile][core_clock_rate] = collections.OrderedDict()

            gpu_clock_rate = profiler_session.profile["maximumGPUClockRate"]
            data[profile][core_clock_rate][gpu_clock_rate] = savings

        return data

    def core_clock_rate_vs_gpu_clock_rate_vs_energy_savings_scatter_plot(self,
                                                                         control_comparison=ControlComparison.MEAN):
        return ScatterPlot(
            title="Core Frequency vs. GPU Frequency vs. Energy Savings",
            plot_series=self.core_clock_rate_vs_gpu_clock_rate_vs_energy_savings(control_comparison),
            x_label="Core Clock Rate (Hertz)",
            y_label="GPU Clock Rate (Hertz)",
            z_label="Energy Savings (Joules)",
            labels=[profiler_session.plot_label for profiler_session in self.data_set.data]
        )

    def core_clock_rate_vs_gpu_clock_rate_vs_runtime_increase(self, control_comparison=ControlComparison.MEAN):
        control_runtime = None
        if control_comparison == ControlComparison.MEAN:
            control_runtime = self.control_data_set.mean_runtime
        elif control_comparison == ControlComparison.MEDIAN:
            control_runtime = self.control_data_set.median_runtime
        elif control_comparison == ControlComparison.OPTIMAL:
            control_runtime = self.control_data_set.minimum_runtime_profiler_session.total_runtime

        data: OrderedDict[
            str, OrderedDict[int, OrderedDict[int, int]]] = collections.OrderedDict({})
        for profiler_session in self.data_set.data:
            increase = Plot.ns_to_s(profiler_session.total_runtime - control_runtime)

            profile = "Saves Time" if increase < 0 else "Costs Time"
            if profile not in data:
                data[profile] = collections.OrderedDict()

            core_clock_rate = profiler_session.profile["maximumCPUClockRate"]
            if core_clock_rate not in data[profile]:
                data[profile][core_clock_rate] = collections.OrderedDict()

            gpu_clock_rate = profiler_session.profile["maximumGPUClockRate"]
            data[profile][core_clock_rate][gpu_clock_rate] = increase

        return data

    def core_clock_rate_vs_gpu_clock_rate_vs_runtime_increase_scatter_plot(self,
                                                                           control_comparison=ControlComparison.MEAN):
        return ScatterPlot(
            title="Core Frequency vs. GPU Frequency vs. Runtime Increase",
            plot_series=self.core_clock_rate_vs_gpu_clock_rate_vs_runtime_increase(control_comparison),
            x_label="Core Clock Rate (Hertz)",
            y_label="GPU Clock Rate (Hertz)",
            z_label="Runtime Increase (Seconds)",
            labels=[profiler_session.plot_label for profiler_session in self.data_set.data]
        )
