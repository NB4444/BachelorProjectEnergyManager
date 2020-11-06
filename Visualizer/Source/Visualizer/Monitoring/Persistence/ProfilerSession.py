from functools import cached_property
from typing import Dict, List, Any

from Visualizer.Monitoring.Persistence.MonitorData import MonitorData
from Visualizer.Persistence.Entity import Entity
from Visualizer.Plotting.DictionaryPlot import DictionaryPlot
from Visualizer.Plotting.MultiPlot import MultiPlot
from Visualizer.Plotting.Plot import Plot
from Visualizer.Plotting.ScatterPlot import ScatterPlot
from Visualizer.Plotting.TablePlot import TablePlot
from Visualizer.Plotting.TimeseriesPlot import TimeseriesPlot


class ProfilerSession(Entity):
    @classmethod
    def load_all(cls, database_file: str):
        return cls._load(database_file)

    @classmethod
    def load_by_id(cls, database_file: str, id: int):
        return cls._load(database_file, f"id = {id}")[0]

    @classmethod
    def _load(cls, database_file: str, conditions: str = None):
        Entity.database_file = database_file

        profiler_sessions = []
        for row in cls._select("ProfilerSession", ["id", "label"], conditions):
            id = row[0]
            label = row[1]

            profiler_session = ProfilerSession(
                database_file,
                id,
                label
            )
            profiler_sessions.append(profiler_session)

        return profiler_sessions

    @classmethod
    def table(cls, profiler_sessions: List["ProfilerSession"]):
        return TablePlot(title="Profiler Sessions", table=[[profiler_session.id, profiler_session.label, profiler_session.profile] for profiler_session in profiler_sessions],
                         columns=["ID", "Label", "Profile"])

    @classmethod
    def optimal_energy_consumption(cls, profiler_sessions: List["ProfilerSession"]):
        optimal_energy_consumption: ProfilerSession = None

        for profiler_session in profiler_sessions:
            energy_consumption = profiler_session.total_energy_consumption

            if optimal_energy_consumption is None or energy_consumption < optimal_energy_consumption.total_energy_consumption:
                optimal_energy_consumption = profiler_session

        return optimal_energy_consumption

    @classmethod
    def optimal_runtime(cls, profiler_sessions: List["ProfilerSession"]):
        optimal_runtime: ProfilerSession = None

        for profiler_session in profiler_sessions:
            runtime = profiler_session.total_runtime

            if optimal_runtime is None or runtime < optimal_runtime.total_runtime:
                optimal_runtime = profiler_session

        return optimal_runtime

    @classmethod
    def optimal_flops(cls, profiler_sessions: List["ProfilerSession"]):
        optimal_flops: ProfilerSession = None

        for profiler_session in profiler_sessions:
            flops = profiler_session.total_flops

            if optimal_flops is None or flops > optimal_flops.total_flops:
                optimal_flops = profiler_session

        return optimal_flops

    @classmethod
    def energy_consumption_vs_runtime(cls, profiler_sessions: List["ProfilerSession"], normalized=True):
        # Find the optimal values
        optimal_energy_consumption = cls.optimal_energy_consumption(profiler_sessions).total_energy_consumption
        optimal_runtime = cls.optimal_runtime(profiler_sessions).total_runtime

        data: Dict[str, Dict[Any, Any]] = {}
        for profiler_session in profiler_sessions:
            profile = f"{profiler_session.profile['minimumCPUClockRate']} - {profiler_session.profile['minimumGPUClockRate']}"
            if profile not in data:
                data[profile] = {}

            if normalized:
                data[profile][profiler_session.total_runtime / optimal_runtime * 100] = profiler_session.total_energy_consumption / optimal_energy_consumption * 100
            else:
                data[profile][profiler_session.total_runtime] = profiler_session.total_energy_consumption

        return data

    @classmethod
    def energy_consumption_vs_runtime_plot(cls, profiler_sessions: List["ProfilerSession"], normalized=True):
        return ScatterPlot(title="Energy Consumption vs. Runtime", plot_series=cls.energy_consumption_vs_runtime(profiler_sessions, normalized),
                           x_label="Runtime" + " (% of optimal)" if normalized else "",
                           y_label="Energy Consumption" + " (% of optimal)" if normalized else "")

    @classmethod
    def energy_consumption_vs_flops(cls, profiler_sessions: List["ProfilerSession"], normalized=True):
        # Find the optimal values
        optimal_energy_consumption = cls.optimal_energy_consumption(profiler_sessions)
        optimal_flops = cls.optimal_flops(profiler_sessions)

        data: Dict[str, Dict[Any, Any]] = {}
        for profiler_session in profiler_sessions:
            profile = f"{profiler_session.profile['minimumCPUClockRate']} - {profiler_session.profile['minimumGPUClockRate']}"
            if profile not in data:
                data[profile] = {}

            if normalized:
                data[profile][profiler_session.total_flops / optimal_flops * 100] = profiler_session.total_energy_consumption / optimal_energy_consumption * 100
            else:
                data[profile][profiler_session.total_flops] = profiler_session.total_energy_consumption

        return data

    @classmethod
    def energy_consumption_vs_flops_plot(cls, profiler_sessions: List["ProfilerSession"], normalized=True):
        return ScatterPlot(title="Energy Consumption vs. FLOPs", plot_series=cls.energy_consumption_vs_flops(profiler_sessions, normalized),
                           x_label="FLOPs (" + ("% of optimal" if normalized else "Operations") + ")",
                           y_label="Energy Consumption (" + "% of optimal" if normalized else "Joules" + ")")

    @classmethod
    def energy_savings_vs_runtime_increase(cls, profiler_sessions: List["ProfilerSession"], normalized=True):
        # Find the optimal values
        optimal_runtime = cls.optimal_runtime(profiler_sessions)

        data: Dict[str, Dict[Any, Any]] = {}
        for profiler_session in profiler_sessions:
            energy_savings = optimal_runtime.total_energy_consumption - profiler_session.total_energy_consumption
            runtime_increase = profiler_session.total_runtime - optimal_runtime.total_runtime

            profile = f"{profiler_session.profile['minimumCPUClockRate']} - {profiler_session.profile['minimumGPUClockRate']}"
            if profile not in data:
                data[profile] = {}

            if normalized:
                data[profile][runtime_increase / optimal_runtime.total_runtime * 100] = energy_savings / optimal_runtime.total_energy_consumption * 100
            else:
                data[profile][Plot.ns_to_s(runtime_increase)] = energy_savings

        return data

    @classmethod
    def energy_savings_vs_runtime_increase_plot(cls, profiler_sessions: List["ProfilerSession"], normalized=True):
        return ScatterPlot(title="Energy Savings vs. Runtime Increase", plot_series=cls.energy_savings_vs_runtime_increase(profiler_sessions, normalized),
                           x_label="Runtime Increase (" + ("% of optimal" if normalized else "Seconds") + ")",
                           y_label="Energy Savings (" + ("% of optimal" if normalized else "Joules") + ")")

    @classmethod
    def energy_savings_vs_flops_decrease(cls, profiler_sessions: List["ProfilerSession"], normalized=True):
        # Find the optimal values
        optimal_flops = cls.optimal_flops(profiler_sessions)

        data: Dict[str, Dict[Any, Any]] = {}
        for profiler_session in profiler_sessions:
            energy_savings = optimal_flops.total_energy_consumption - profiler_session.total_energy_consumption
            flops_decrease = optimal_flops.total_flops - profiler_session.total_flops

            profile = f"{profiler_session.profile['minimumCPUClockRate']} - {profiler_session.profile['minimumGPUClockRate']}"
            if profile not in data:
                data[profile] = {}

            if normalized:
                data[profile][flops_decrease / optimal_flops.total_runtime * 100] = energy_savings / optimal_flops.total_energy_consumption * 100
            else:
                data[profile][flops_decrease] = energy_savings

        return data

    @classmethod
    def energy_savings_vs_flops_decrease_plot(cls, profiler_sessions: List["ProfilerSession"], normalized=True):
        return ScatterPlot(title="Energy Savings vs. Runtime Increase", plot_series=cls.energy_savings_vs_runtime_increase(profiler_sessions, normalized),
                           x_label="Runtime Increase" + " (% of optimal)" if normalized else "",
                           y_label="Energy Savings" + " (% of optimal)" if normalized else "")

    def __init__(self, database_file: str, id: int, label: str):
        super().__init__(database_file)

        self.id = id
        self.label = label

    @cached_property
    def profile(self):
        profile = {}
        for row in self._select("ProfilerSessionProfile", ["argument", "value"], f"profilerSessionID = {self.id}"):
            argument = row[0]
            value = row[1]
            profile[argument] = value

        return profile

    @cached_property
    def monitor_data(self):
        return MonitorData.load_by_profiler_session(self.database_file, self)

    def get_monitor_data_by_monitor_name(self, monitor_name: str):
        matching_monitors = [monitor_data for monitor_data in self.monitor_data if monitor_data.monitor_name == monitor_name]
        if len(matching_monitors) > 0:
            return matching_monitors[0]
        else:
            print(f"Could not find monitor with name {monitor_name} for profiler session with ID {self.id}")
            raise NotImplementedError

    @cached_property
    def cpu_monitor(self):
        return self.get_monitor_data_by_monitor_name("CPUMonitor")

    @cached_property
    def gpu_monitor(self):
        return self.get_monitor_data_by_monitor_name("GPUMonitor")

    @cached_property
    def node_monitor(self):
        return self.get_monitor_data_by_monitor_name("NodeMonitor")

    @property
    def summary(self):
        return {
            "Label": self.label,
            "GPU Brand": self.gpu_monitor.collect_constant_value("brand", str),
            "GPU Compute Capability Major Version": self.gpu_monitor.collect_constant_value("computeCapabilityMajorVersion", int),
            "GPU Compute Capability Minor Version": self.gpu_monitor.collect_constant_value("computeCapabilityMinorVersion", int),
            # "GPU Memory Bandwidth (B/s)": self.gpu_monitor.collect_constant_value("memoryBandwidth", float, Plot.humanize_size),
            "GPU Memory Size (B)": self.gpu_monitor.collect_constant_value("memorySize", int, Plot.humanize_size),
            # "GPU Multiprocessor Count": self.gpu_monitor.collect_constant_value("multiprocessorCount", int, Plot.humanize_number),
            "GPU Name": self.gpu_monitor.collect_constant_value("name", str),
            "GPU PCIe Link Width (B)": self.gpu_monitor.collect_constant_value("pciELinkWidth", int, Plot.humanize_size),
            "GPU Default Power Limit (W)": self.gpu_monitor.collect_constant_value("defaultPowerLimit", float, Plot.humanize_number),
            "GPU Supported Core Clock Rates (Hz)": self.gpu_monitor.collect_constant_value("supportedCoreClockRates", str, Plot.parse_number_list),
            "GPU Supported Memory Clock Rates (Hz)": self.gpu_monitor.collect_constant_value("supportedMemoryClockRates", str, Plot.parse_number_list),
            "GPU Default Auto Boosted Clocks Enabled": self.gpu_monitor.collect_constant_value("defaultAutoBoostedClocksEnabled", bool)
        }

    @property
    def summary_plot(self):
        return DictionaryPlot(title="Summary", dictionary=self.summary)

    @property
    def overview_plot(self):
        return MultiPlot(title="Overview", plots=[
            self.clock_rate_plot(),
            self.energy_consumption_plot(),
            self.fan_speed_plot,
            self.memory_consumption_plot(),
            self.power_consumption_plot(),
            self.processes_plot,
            self.switches_plot,
            self.temperature_plot,
            self.timespan_plot,
            self.utilization_rate_plot,
            self.kernel_coordinates_plot,
            # self.correlations_plot,
        ])

    @property
    def clock_rate(self):
        clock_rate = self.cpu_monitor.collect_summarized_indexed_values("CPU", "CPU Core", "coreClockRate", "coreClockRateCore", int)
        clock_rate.update({
            "GPU Core": self.gpu_monitor.collect_values("coreClockRate", int),
            "GPU Memory": self.gpu_monitor.collect_values("memoryClockRate", int),
            "GPU SM": self.gpu_monitor.collect_values("streamingMultiprocessorClockRate", int)
        })

        return clock_rate

    @property
    def clock_rate_limits(self):
        clock_rate_limits = self.cpu_monitor.collect_summarized_indexed_values("CPU Maximum", "CPU Maximum Core", "maximumCoreClockRate", "maximumCoreClockRateCore",
                                                                               int)
        clock_rate_limits.update({
            "GPU Core Maximum": self.gpu_monitor.collect_values("maximumCoreClockRate", int),
            "GPU Memory Maximum": self.gpu_monitor.collect_values("GPUMonitor", "maximumMemoryClockRate", int)
        })

        return clock_rate_limits

    def clock_rate_plot(self, plot_limits=True):
        series = self.clock_rate
        if plot_limits:
            series.update(self.clock_rate_limits)

        return TimeseriesPlot(title="Clock Rate", plot_series=series, y_label="Clock Rate (Hz)")

    def energy_consumption(self, modifier=lambda value: value):
        return {
            "CPU": self.cpu_monitor.collect_values("energyConsumption", float, modifier),
            "GPU": self.gpu_monitor.collect_values("energyConsumption", float, modifier),
            "Node": self.node_monitor.collect_values("energyConsumption", float, modifier)
        }

    def energy_consumption_plot(self, unit_string="J", modifier=lambda value: value):
        return TimeseriesPlot(title="Energy Consumption", plot_series=self.energy_consumption(modifier), y_label=f"Energy Consumption ({unit_string})")

    @property
    def total_energy_consumption(self):
        _, variables = list(self.node_monitor.monitor_data.items())[-1]
        return variables["energyConsumption"]

    @property
    def fan_speed(self):
        return self.gpu_monitor.collect_summarized_indexed_values("GPU", "GPU Fan", "fanSpeed", "fanSpeedFan", float, Plot.to_percentage)

    @property
    def fan_speed_plot(self):
        return TimeseriesPlot(title="Fan Speed", plot_series=self.fan_speed, y_label="Fan Speed (%)")

    @property
    def total_flops(self):
        _, variables = list(self.node_monitor.monitor_data.items())[-1]
        return variables["flops"]

    @property
    def memory_consumption(self):
        return {
            "GPU Free": self.gpu_monitor.collect_values("memoryFreeSize", int),
            "GPU Used": self.gpu_monitor.collect_values("memoryUsedSize", int),
            "RAM Free": self.node_monitor.collect_values("freeMemorySize", int),
            "RAM Used": self.node_monitor.collect_values("usedMemorySize", int),
            "Swap Free": self.node_monitor.collect_values("freeSwapMemorySize", int),
            "Swap Used": self.node_monitor.collect_values("usedSwapMemorySize", int),
            "High Free": self.node_monitor.collect_values("freeHighMemorySize", int),
            "High Used": self.node_monitor.collect_values("usedHighMemorySize", int)
        }

    @property
    def memory_sizes(self):
        return {
            "GPU Kernel Dynamic Shared": self.gpu_monitor.collect_values("kernelDynamicSharedMemorySize", int),
            "GPU Kernel Static Shared": self.gpu_monitor.collect_values("kernelStaticSharedMemorySize", int),
            "GPU": self.gpu_monitor.collect_values("memorySize", int),
            "GPU PCIe Link": self.gpu_monitor.collect_values("pciELinkWidth", int),
            "RAM": self.node_monitor.collect_values("memorySize", int),
            "RAM Shared": self.node_monitor.collect_values("sharedMemorySize", int),
            "RAM Buffer": self.node_monitor.collect_values("bufferMemorySize", int),
            "Swap": self.node_monitor.collect_values("swapMemorySize", int),
            "High": self.node_monitor.collect_values("highMemorySize", int)
        }

    def memory_consumption_plot(self, plot_sizes=True):
        series = self.memory_consumption
        if plot_sizes:
            series.update(self.memory_sizes)

        return TimeseriesPlot(title="Memory Consumption", plot_series=series, y_label="Memory Consumption (B)")

    @property
    def power_consumption(self):
        return {
            "CPU": self.cpu_monitor.collect_values("powerConsumption", float),
            "GPU": self.gpu_monitor.collect_values("powerConsumption", float),
            "Node": self.node_monitor.collect_values("powerConsumption", float)
        }

    @property
    def power_limits(self):
        return {
            "GPU Power Limit": self.gpu_monitor.collect_values("powerLimit", float),
            "GPU Enforced Power Limit": self.gpu_monitor.collect_values("powerLimit", float)
        }

    def power_consumption_plot(self, plot_limits=True):
        series = self.power_consumption
        if plot_limits:
            series.update(self.power_limits)

        return TimeseriesPlot(title="Power Consumption", plot_series=series, y_label="Power Consumption (W)")

    @property
    def processes(self):
        return {
            "Processes": self.node_monitor.collect_values("processCount", int)
        }

    @property
    def processes_plot(self):
        return TimeseriesPlot(title="Processes", plot_series=self.processes, y_label="Processes")

    @property
    def total_runtime(self):
        _, variables = list(self.node_monitor.monitor_data.items())[-1]
        return variables["runtime"]

    @property
    def switches(self):
        return {
            "GPU Auto Boosted Clocks": self.gpu_monitor.collect_values("autoBoostedClocksEnabled", bool)
        }

    @property
    def switches_plot(self):
        return TimeseriesPlot(title="Switches", plot_series=self.switches, y_label="Switches (bool)")

    @property
    def temperature(self):
        return {
            "CPU": self.cpu_monitor.collect_values("temperature", float),
            "GPU": self.gpu_monitor.collect_values("temperature", float)
        }

    @property
    def temperature_plot(self):
        return TimeseriesPlot(title="Temperature", plot_series=self.temperature, y_label="Temperature (C)")

    @property
    def timespan(self):
        timespan = self.cpu_monitor.collect_summarized_indexed_values("CPU Guest Nice", "CPU Guest Nice Core", "guestNiceTimespan", "guestNiceTimespanCore", float, Plot.to_percentage)
        timespan.update(self.cpu_monitor.collect_summarized_indexed_values("CPU Guest", "CPU Guest Core", "guestTimespan", "guestTimespanCore", float, Plot.to_percentage))
        timespan.update(self.cpu_monitor.collect_summarized_indexed_values("CPU IO Wait", "CPU IO Wait Core", "ioWaitTimespan", "ioWaitTimespanCore", float, Plot.to_percentage))
        timespan.update(self.cpu_monitor.collect_summarized_indexed_values("CPU Idle", "CPU Idle Core", "idleTimespan", "idleTimespanCore", float, Plot.to_percentage))
        timespan.update(self.cpu_monitor.collect_summarized_indexed_values("CPU Interrupts", "CPU Interrupts Core", "interruptsTimespan", "interruptsTimespanCore", float, Plot.to_percentage))
        timespan.update(self.cpu_monitor.collect_summarized_indexed_values("CPU Nice", "CPU Nice Core", "niceTimespan", "niceTimespanCore", float, Plot.to_percentage))
        timespan.update(
            self.cpu_monitor.collect_summarized_indexed_values("CPU Soft Interrupts", "CPU Soft Interrupts Core", "softInterruptsTimespan", "softInterruptsTimespanCore", float, Plot.to_percentage))
        timespan.update(self.cpu_monitor.collect_summarized_indexed_values("CPU Steal", "CPU Steal Core", "stealTimespan", "stealTimespanCore", float, Plot.to_percentage))
        timespan.update(self.cpu_monitor.collect_summarized_indexed_values("CPU System", "CPU System Core", "systemTimespan", "systemTimespanCore", float, Plot.to_percentage))
        timespan.update(self.cpu_monitor.collect_summarized_indexed_values("CPU User", "CPU User Core", "userTimespan", "userTimespanCore", float, Plot.to_percentage))
        timespan[("Runtime", "")] = self.node_monitor.collect_values("runtime", float, Plot.ns_to_s)

        return timespan

    @property
    def timespan_plot(self):
        return TimeseriesPlot(title="Timespan", plot_series=self.timespan, y_label="Timespan (s)")

    @property
    def utilization_rate(self):
        utilization_rate = self.cpu_monitor.collect_summarized_indexed_values("CPU", "CPU Core", "coreUtilizationRate", "coreUtilizationRateCore", float, Plot.to_percentage)
        utilization_rate.update({
            "GPU Core": self.gpu_monitor.collect_values("coreUtilizationRate", float, Plot.to_percentage),
            "GPU Memory": self.gpu_monitor.collect_values("memoryUtilizationRate", float, Plot.to_percentage)
        })

        return utilization_rate

    @property
    def utilization_rate_plot(self):
        return TimeseriesPlot(title="Utilization Rate", plot_series=self.utilization_rate, y_label="Utilization Rate (%)")

    @property
    def kernel_coordinates(self):
        return {
            "Block X": self.gpu_monitor.collect_values("kernelBlockX", int),
            "Block Y": self.gpu_monitor.collect_values("kernelBlockY", int),
            "Block Z": self.gpu_monitor.collect_values("kernelBlockZ", int),
            "Grid X": self.gpu_monitor.collect_values("kernelGridX", int),
            "Grid Y": self.gpu_monitor.collect_values("kernelGridY", int),
            "Grid Z": self.gpu_monitor.collect_values("kernelGridZ", int)
        }

    @property
    def kernel_coordinates_plot(self):
        return TimeseriesPlot(title="Kernel Coordinates", plot_series=self.kernel_coordinates, y_label="Coordinate")
