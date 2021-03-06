from datetime import datetime
from functools import cached_property
from typing import Dict, List, Tuple

from Visualizer.Monitoring.Persistence.MonitorSession import MonitorSession
from Visualizer.Persistence.Entity import Entity
from Visualizer.Plotting.DictionaryPlot import DictionaryPlot
from Visualizer.Plotting.EventPlot import EventPlot
from Visualizer.Plotting.MultiPlot import MultiPlot
from Visualizer.Plotting.Plot import Plot
from Visualizer.Plotting.TimeseriesPlot import TimeseriesPlot
from Visualizer.Utility.Parsing import determine_type


class ProfilerSession(Entity):
    @classmethod
    def load_all(cls, database_file: str):
        return cls._load(database_file)

    @classmethod
    def load_by_id(cls, database_file: str, id: int):
        return cls._load(database_file, f"id = {id}")[0]

    @classmethod
    def _load(cls, database_file: str, conditions: str = None):
        profiler_sessions = []
        for row in Entity(database_file)._select("ProfilerSession", ["id", "label"], conditions):
            id = row[0]
            label = row[1]

            profiler_session = ProfilerSession(
                database_file,
                id,
                label
            )
            profiler_sessions.append(profiler_session)

        return profiler_sessions

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
            profile[argument] = determine_type(value)(value)

        return profile

    @cached_property
    def monitor_sessions(self):
        return MonitorSession.load_by_profiler_session(self.database_file, self)

    def get_monitor_session_by_monitor_name(self, monitor_name: str):
        return [monitor_session for monitor_session in self.monitor_sessions if
                monitor_session.monitor_name == monitor_name]

    @cached_property
    def cpu_monitors(self):
        return self.get_monitor_session_by_monitor_name("CPUMonitor")

    @cached_property
    def cpu_core_monitors(self):
        return self.get_monitor_session_by_monitor_name("CPUCoreMonitor")

    @cached_property
    def gpu_monitors(self):
        return self.get_monitor_session_by_monitor_name("GPUMonitor")

    @cached_property
    def node_monitor(self):
        node_monitors = self.get_monitor_session_by_monitor_name("NodeMonitor")
        return node_monitors[0] if len(node_monitors) > 0 else None

    @cached_property
    def energy_monitor(self):
        energy_monitors = self.get_monitor_session_by_monitor_name("EnergyMonitor")
        return energy_monitors[0] if len(energy_monitors) > 0 else None

    @cached_property
    def ear_monitor(self):
        return self.get_monitor_session_by_monitor_name("EARMonitor")[0]

    @property
    def monitor_data_correlations_plot(self):
        return MonitorSession.correlations_plot(self.monitor_sessions)

    def plot_label(self, use_ear=False):
        result = f"ID={self.id}\n"
        result += "\n"
        if "minimumCPUClockRate" in self.profile:
            result += f"Core Minimum Frequency={Plot.humanize_number(self.profile['minimumCPUClockRate'])}\n"
        if "maximumCPUClockRate" in self.profile:
            result += f"Core Maximum Frequency={Plot.humanize_number(self.profile['maximumCPUClockRate'])}\n"
        if "minimumGPUClockRate" in self.profile:
            result += f"GPU Minimum Frequency={Plot.humanize_number(self.profile['minimumGPUClockRate'])}\n"
        if "maximumGPUClockRate" in self.profile:
            result += f"GPU Maximum Frequency={Plot.humanize_number(self.profile['maximumGPUClockRate'])}\n"
        result += f"Energy Consumption={Plot.humanize_number(self.total_energy_consumption(use_ear))} J\n"
        result += f"Runtime={Plot.ns_to_s(self.total_runtime)} s"

        return result

    @property
    def summary(self):
        return {
            "Label": self.label,
            "GPUs": [{
                "Brand": gpu_monitor.get_value("brand", str),
                "Compute Capability Major Version": gpu_monitor.get_value("computeCapabilityMajorVersion", int),
                "Compute Capability Minor Version": gpu_monitor.get_value("computeCapabilityMinorVersion", int),
                # "GPU Memory Bandwidth (B/s)": gpu_monitor.get_value("memoryBandwidth", float, Plot.humanize_size),
                "Memory Size (B)": gpu_monitor.get_value("memorySize", int, Plot.humanize_size),
                # "GPU Multiprocessor Count": gpu_monitor.get_value("multiprocessorCount", int, Plot.humanize_number),
                "Name": gpu_monitor.get_value("name", str),
                "PCIe Link Width (B)": gpu_monitor.get_value("pciELinkWidth", int, Plot.humanize_size),
                "Default Power Limit (W)": gpu_monitor.get_value("defaultPowerLimit", float, Plot.humanize_number),
                "Supported Core Clock Rates (Hz)": gpu_monitor.get_value("supportedCoreClockRates", str,
                                                                         Plot.parse_number_list),
                "Supported Memory Clock Rates (Hz)": gpu_monitor.get_value("supportedMemoryClockRates", str,
                                                                           Plot.parse_number_list),
                "Default Auto Boosted Clocks Enabled": gpu_monitor.get_value("defaultAutoBoostedClocksEnabled", bool)
            } for gpu_monitor in self.gpu_monitors]
        }

    @property
    def summary_dictionary_plot(self):
        return DictionaryPlot(title="Summary", dictionary=self.summary)

    @property
    def overview_multi_plot(self):
        return MultiPlot(title="Overview", plots=[
            self.clock_rate_timeseries_plot(),
            self.energy_consumption_timeseries_plot(),
            self.fan_speed_timeseries_plot,
            self.memory_consumption_timeseries_plot(),
            self.power_consumption_timeseries_plot(),
            self.processes_timeseries_plot,
            self.switches_timeseries_plot,
            self.temperature_timeseries_plot,
            self.timespan_timeseries_plot,
            self.utilization_rate_timeseries_plot,
            self.kernel_coordinates_timeseries_plot,
            # self.monitor_data_correlations_plot,
        ])

    def clock_rate(self, use_ear=False):
        clock_rate = {}

        for cpu_monitor in self.cpu_monitors:
            cpu_id = cpu_monitor.get_value("id", int)
            clock_rate.update({f"CPU {cpu_id}": cpu_monitor.get_values("coreClockRate", int)})

        for cpu_core_monitor in self.cpu_core_monitors:
            cpu_id = cpu_core_monitor.get_value("cpuID", int)
            core_id = cpu_core_monitor.get_value("coreID", int)
            clock_rate.update({f"CPU {cpu_id} Core {core_id}": cpu_core_monitor.get_values("coreClockRate", int)})

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            clock_rate.update({
                f"GPU {gpu_id} Core": gpu_monitor.get_values("coreClockRate", int),
                f"GPU {gpu_id} Memory": gpu_monitor.get_values("memoryClockRate", int),
                f"GPU {gpu_id} SM": gpu_monitor.get_values("streamingMultiprocessorClockRate", int)
            })

        if use_ear:
            clock_rate.update({
                "EAR CPU Clock Rate": self.ear_monitor.get_values("cpuCoreClockRate", float),
            })

        return clock_rate

    @property
    def clock_rate_limits(self):
        clock_rate_limits = {}

        for cpu_monitor in self.cpu_monitors:
            cpu_id = cpu_monitor.get_value("id", int)
            clock_rate_limits.update({f"CPU {cpu_id} Maximum": cpu_monitor.get_values("maximumCoreClockRate", int)})
            clock_rate_limits.update(
                {f"CPU {cpu_id} Current Maximum": cpu_monitor.get_values("currentMaximumCoreClockRate", int)})
            clock_rate_limits.update({f"CPU {cpu_id} Minimum": cpu_monitor.get_values("minimumCoreClockRate", int)})
            clock_rate_limits.update(
                {f"CPU {cpu_id} Current Minimum": cpu_monitor.get_values("currentMinimumCoreClockRate", int)})

        for cpu_core_monitor in self.cpu_core_monitors:
            cpu_id = cpu_core_monitor.get_value("cpuID", int)
            core_id = cpu_core_monitor.get_value("coreID", int)
            clock_rate_limits.update(
                {f"CPU {cpu_id} Core {core_id} Maximum": cpu_core_monitor.get_values("maximumCoreClockRate", int)})
            clock_rate_limits.update({
                f"CPU {cpu_id} Core {core_id} Current Maximum": cpu_core_monitor.get_values(
                    "currentMaximumCoreClockRate", int)
            })
            clock_rate_limits.update(
                {f"CPU {cpu_id} Core {core_id} Minimum": cpu_core_monitor.get_values("minimumCoreClockRate", int)})
            clock_rate_limits.update({
                f"CPU {cpu_id} Core {core_id} Current Minimum": cpu_core_monitor.get_values(
                    "currentMinimumCoreClockRate", int)
            })

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            clock_rate_limits.update({
                f"GPU {gpu_id} Core Maximum": gpu_monitor.get_values("maximumCoreClockRate", int),
                f"GPU {gpu_id} Core Current Maximum": gpu_monitor.get_values("currentMaximumCoreClockRate", int),
                f"GPU {gpu_id} Core Current Minimum": gpu_monitor.get_values("currentMinimumCoreClockRate", int),
                f"GPU {gpu_id} Memory Maximum": gpu_monitor.get_values("maximumMemoryClockRate", int)
            })

        return clock_rate_limits

    def clock_rate_timeseries_plot(self, plot_limits=True, use_ear=False):
        series = self.clock_rate(use_ear)
        if plot_limits:
            series.update(self.clock_rate_limits)

        return TimeseriesPlot(title="Clock Rate", plot_series=series, y_label="Clock Rate (Hertz)")

    @property
    def cpu_clock_rate_string(self):
        result = "CPU"
        if "minimumCPUClockRate" in self.profile or "maximumCPUClockRate" in self.profile:
            result += "@{"

            if "minimumCPUClockRate" in self.profile:
                result += Plot.humanize_number(self.profile["minimumCPUClockRate"])
            if "maximumCPUClockRate" in self.profile:
                result += Plot.humanize_number(self.profile["maximumCPUClockRate"])

            result += "}"

        return result

    @property
    def gpu_clock_rate_string(self):
        result = "GPU"
        if "minimumGPUClockRate" in self.profile or "maximumGPUClockRate" in self.profile:
            result += "@{"

            if "minimumGPUClockRate" in self.profile:
                result += Plot.humanize_number(self.profile["minimumGPUClockRate"])
            if "maximumGPUClockRate" in self.profile:
                result += Plot.humanize_number(self.profile["maximumGPUClockRate"])

            result += "}"

        return result

    def energy_consumption(self, modifier=lambda value: value, use_ear=False):
        energy_consumption = {}

        for cpu_monitor in self.cpu_monitors:
            cpu_id = cpu_monitor.get_value("id", int)
            energy_consumption.update({f"CPU {cpu_id}": cpu_monitor.get_values("energyConsumption", float, modifier)})

        for cpu_core_monitor in self.cpu_core_monitors:
            cpu_id = cpu_core_monitor.get_value("cpuID", int)
            core_id = cpu_core_monitor.get_value("coreID", int)
            energy_consumption.update(
                {f"CPU {cpu_id} Core {core_id}": cpu_core_monitor.get_values("energyConsumption", float, modifier)})

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            energy_consumption.update({f"GPU {gpu_id}": gpu_monitor.get_values("energyConsumption", float, modifier)})

        energy_consumption.update({
            "Node": self.node_monitor.get_values("energyConsumption", float, modifier),
        })

        if use_ear:
            energy_consumption.update({
                "EAR Node": self.ear_monitor.get_values("energyConsumption", float, modifier)
            })

        return energy_consumption

    def energy_consumption_timeseries_plot(self, unit_string="J", modifier=lambda value: value, use_ear=False):
        return TimeseriesPlot(title="Energy Consumption", plot_series=self.energy_consumption(modifier, use_ear),
                              y_label=f"Energy Consumption ({unit_string})")

    def total_energy_consumption(self, use_ear=False):
        return (self.node_monitor if not use_ear else self.ear_monitor).get_last_value("energyConsumption", float)

    @property
    def events(self):
        """
        Gets the events per GPU monitor.
        :return: The events.
        """
        events: Dict[MonitorSession, List[Tuple[str, datetime, datetime]]] = {}

        for gpu_monitor in self.gpu_monitors:
            events[gpu_monitor] = []

            # Process the event lists
            event_items = list(gpu_monitor.get_values("events", str).items())
            for index, (timestamp, event_list) in enumerate(event_items):
                for event in event_list.split(","):
                    event_name = event.split("(")[0]
                    event_type = event.split("(")[1].split(")")[0]

                    def find_exit():
                        if event_type == "ENTER":
                            # Find the matching exit event
                            for exit_index, (exit_timestamp, exit_event_list) in enumerate(event_items[index + 1:]):
                                for exit_event in exit_event_list.split(","):
                                    exit_event_name = exit_event.split("(")[0]
                                    exit_event_type = exit_event.split("(")[1].split(")")[0]

                                    if exit_event_name == event_name and exit_event_type == "EXIT":
                                        events[gpu_monitor].append((event_name, timestamp, exit_timestamp))
                                        return

                    find_exit()

        return events

    def events_event_plot(self):
        return [EventPlot(title=f"Events (GPU {gpu_monitor.get_value('id', int)})", events=events) for
                gpu_monitor, events in self.events.items()]

    @property
    def fan_speed(self):
        fan_speed = {}

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            fan_speed.update(
                gpu_monitor.get_summarized_indexed_value_series(f"GPU {gpu_id}", f"GPU {gpu_id} Fan", "fanSpeed",
                                                                "fanSpeedFan", float, Plot.to_percentage))

        return fan_speed

    @property
    def fan_speed_timeseries_plot(self):
        return TimeseriesPlot(title="Fan Speed", plot_series=self.fan_speed, y_label="Fan Speed (%)")

    @property
    def instructions_per_cycle(self):
        utilization_rate = {}

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            utilization_rate.update({
                f"GPU {gpu_id}": gpu_monitor.get_values("instructionsPerCycle", float),
            })

        return utilization_rate

    @property
    def instructions_per_cycle_timeseries_plot(self):
        return TimeseriesPlot(title="Instructions Per Cycle", plot_series=self.instructions_per_cycle,
                              y_label="Instructions")

    @property
    def total_flops(self):
        _, variables = list(self.node_monitor.monitor_data.items())[-1]
        return variables["flops"]

    @property
    def memory_consumption(self):
        memory_consumption = {}

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            memory_consumption.update({
                f"GPU {gpu_id} Free": gpu_monitor.get_values("memoryFreeSize", int),
                f"GPU {gpu_id} Used": gpu_monitor.get_values("memoryUsedSize", int)
            })

        memory_consumption.update({
            "RAM Free": self.node_monitor.get_values("freeMemorySize", int),
            "RAM Used": self.node_monitor.get_values("usedMemorySize", int),
            "Swap Free": self.node_monitor.get_values("freeSwapMemorySize", int),
            "Swap Used": self.node_monitor.get_values("usedSwapMemorySize", int),
            "High Free": self.node_monitor.get_values("freeHighMemorySize", int),
            "High Used": self.node_monitor.get_values("usedHighMemorySize", int)
        })

        return memory_consumption

    @property
    def memory_sizes(self):
        memory_sizes = {}

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            memory_sizes.update({
                f"GPU {gpu_id} Kernel Dynamic Shared": gpu_monitor.get_values("kernelDynamicSharedMemorySize", int),
                f"GPU {gpu_id} Kernel Static Shared": gpu_monitor.get_values("kernelStaticSharedMemorySize", int),
                f"GPU {gpu_id}": gpu_monitor.get_values("memorySize", int),
                f"GPU {gpu_id} PCIe Link": gpu_monitor.get_values("pciELinkWidth", int)
            })

        memory_sizes.update({
            "RAM": self.node_monitor.get_values("memorySize", int),
            "RAM Shared": self.node_monitor.get_values("sharedMemorySize", int),
            "RAM Buffer": self.node_monitor.get_values("bufferMemorySize", int),
            "Swap": self.node_monitor.get_values("swapMemorySize", int),
            "High": self.node_monitor.get_values("highMemorySize", int)
        })

        return memory_sizes

    def memory_consumption_timeseries_plot(self, plot_sizes=True):
        series = self.memory_consumption
        if plot_sizes:
            series.update(self.memory_sizes)

        return TimeseriesPlot(title="Memory Consumption", plot_series=series, y_label="Memory Consumption (B)")

    def power_consumption(self, use_ear=False):
        power_consumption = {}

        for cpu_monitor in self.cpu_monitors:
            cpu_id = cpu_monitor.get_value("id", int)
            power_consumption.update({f"CPU {cpu_id}": cpu_monitor.get_values("powerConsumption", float)})

        for cpu_core_monitor in self.cpu_core_monitors:
            cpu_id = cpu_core_monitor.get_value("cpuID", int)
            core_id = cpu_core_monitor.get_value("coreID", int)
            power_consumption.update(
                {f"CPU {cpu_id} Core {core_id}": cpu_core_monitor.get_values("powerConsumption", float)})

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            power_consumption.update({f"GPU {gpu_id}": gpu_monitor.get_values("powerConsumption", float)})

        power_consumption.update({
            "Node": self.node_monitor.get_values("powerConsumption", float),
        })

        if use_ear:
            power_consumption.update({
                "EAR Node Average": self.ear_monitor.get_values("averagePowerConsumption", float)
            })

        return power_consumption

    @property
    def power_limits(self):
        power_limits = {}

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            power_limits.update({
                f"GPU {gpu_id} Power Limit": gpu_monitor.get_values("powerLimit", float),
                f"GPU {gpu_id} Enforced Power Limit": gpu_monitor.get_values("powerLimit", float)
            })

        return power_limits

    def power_consumption_timeseries_plot(self, plot_limits=True, use_ear=False):
        series = self.power_consumption(use_ear)
        if plot_limits:
            series.update(self.power_limits)

        return TimeseriesPlot(title="Power Consumption", plot_series=series, y_label="Power Consumption (W)")

    @property
    def processes(self):
        return {
            "Processes": self.node_monitor.get_values("processCount", int)
        }

    @property
    def processes_timeseries_plot(self):
        return TimeseriesPlot(title="Processes", plot_series=self.processes, y_label="Processes")

    @property
    def total_runtime(self):
        _, variables = list(self.node_monitor.monitor_data.items())[-1]
        return variables["runtime"]

    @property
    def states(self):
        """
        Gets the states per GPU monitor.
        :return: The states.
        """
        states: List[Tuple[str, datetime, datetime]] = []

        current_state = None
        start_timestamp = None

        # Process the states
        if self.energy_monitor is not None:
            state_items = list(self.energy_monitor.get_values("state", str).items())
            for index, (timestamp, state) in enumerate(state_items):
                # Update the state if it changes or if this is the last state and store the current value
                if state != current_state or index == len(state_items) - 1:
                    # Store the current state
                    if current_state is not None:
                        states.append((current_state, start_timestamp, timestamp))

                    # Update the state
                    current_state = state
                    start_timestamp = timestamp

        return states

    def states_event_plot(self):
        return EventPlot(title="States", events=self.states)

    @property
    def switches(self):
        switches = {}

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            switches.update({
                f"GPU {gpu_id} Auto Boosted Clocks": gpu_monitor.get_values("autoBoostedClocksEnabled", bool)
            })

        return switches

    @property
    def switches_timeseries_plot(self):
        return TimeseriesPlot(title="Switches", plot_series=self.switches, y_label="Switches (bool)")

    @property
    def temperature(self):
        temperature = {}

        for cpu_monitor in self.cpu_monitors:
            cpu_id = cpu_monitor.get_value("id", int)
            temperature.update({f"CPU {cpu_id}": cpu_monitor.get_values("temperature", float)})

        for cpu_core_monitor in self.cpu_core_monitors:
            cpu_id = cpu_core_monitor.get_value("cpuID", int)
            core_id = cpu_core_monitor.get_value("coreID", int)
            temperature.update({f"CPU {cpu_id} Core {core_id}": cpu_core_monitor.get_values("temperature", float)})

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            temperature.update({f"GPU {gpu_id}": gpu_monitor.get_values("temperature", float)})

        return temperature

    @property
    def temperature_timeseries_plot(self):
        return TimeseriesPlot(title="Temperature", plot_series=self.temperature, y_label="Temperature (C)")

    def timespan(self, use_ear=False):
        timespan = {}

        for cpu_monitor in self.cpu_monitors:
            cpu_id = cpu_monitor.get_value("id", int)
            timespan.update({
                f"CPU {cpu_id} Guest Nice": cpu_monitor.get_values("guestNiceTimespan", float, Plot.ns_to_s),
                f"CPU {cpu_id} Guest": cpu_monitor.get_values("guestTimespan", float, Plot.ns_to_s),
                f"CPU {cpu_id} IO Wait": cpu_monitor.get_values("ioWaitTimespan", float, Plot.ns_to_s),
                f"CPU {cpu_id} Idle": cpu_monitor.get_values("idleTimespan", float, Plot.ns_to_s),
                f"CPU {cpu_id} Interrupts": cpu_monitor.get_values("interruptsTimespan", float, Plot.ns_to_s),
                f"CPU {cpu_id} Nice": cpu_monitor.get_values("niceTimespan", float, Plot.ns_to_s),
                f"CPU {cpu_id} Soft Interrupts": cpu_monitor.get_values("softInterruptsTimespan", float,
                                                                        Plot.ns_to_s),
                f"CPU {cpu_id} Steal": cpu_monitor.get_values("stealTimespan", float, Plot.ns_to_s),
                f"CPU {cpu_id} System": cpu_monitor.get_values("systemTimespan", float, Plot.ns_to_s),
                f"CPU {cpu_id} User": cpu_monitor.get_values("userTimespan", float, Plot.ns_to_s)
            })

        for cpu_core_monitor in self.cpu_core_monitors:
            cpu_id = cpu_core_monitor.get_value("cpuID", int)
            core_id = cpu_core_monitor.get_value("coreID", int)
            timespan.update({
                f"CPU {cpu_id} Core {core_id} Guest Nice": cpu_core_monitor.get_values("guestNiceTimespan", float,
                                                                                       Plot.ns_to_s),
                f"CPU {cpu_id} Core {core_id} Guest": cpu_core_monitor.get_values("guestTimespan", float,
                                                                                  Plot.ns_to_s),
                f"CPU {cpu_id} Core {core_id} IO Wait": cpu_core_monitor.get_values("ioWaitTimespan", float,
                                                                                    Plot.ns_to_s),
                f"CPU {cpu_id} Core {core_id} Idle": cpu_core_monitor.get_values("idleTimespan", float,
                                                                                 Plot.ns_to_s),
                f"CPU {cpu_id} Core {core_id} Interrupts": cpu_core_monitor.get_values("interruptsTimespan", float,
                                                                                       Plot.ns_to_s),
                f"CPU {cpu_id} Core {core_id} Nice": cpu_core_monitor.get_values("niceTimespan", float,
                                                                                 Plot.ns_to_s),
                f"CPU {cpu_id} Core {core_id} Soft Interrupts": cpu_core_monitor.get_values("softInterruptsTimespan",
                                                                                            float, Plot.ns_to_s),
                f"CPU {cpu_id} Core {core_id} Steal": cpu_core_monitor.get_values("stealTimespan", float,
                                                                                  Plot.ns_to_s),
                f"CPU {cpu_id} Core {core_id} System": cpu_core_monitor.get_values("systemTimespan", float,
                                                                                   Plot.ns_to_s),
                f"CPU {cpu_id} Core {core_id} User": cpu_core_monitor.get_values("userTimespan", float,
                                                                                 Plot.ns_to_s)
            })

        timespan.update({
            "Runtime": self.node_monitor.get_values("runtime", float, Plot.ns_to_s)
        })

        if use_ear:
            timespan.update({
                "EAR Runtime": self.ear_monitor.get_values("applicationRuntime", float, Plot.ns_to_s)
            })

        return timespan

    def timespan_timeseries_plot(self, use_ear=False):
        return TimeseriesPlot(title="Timespan", plot_series=self.timespan(use_ear), y_label="Timespan (s)")

    @property
    def utilization_rate(self):
        utilization_rate = {}

        for cpu_monitor in self.cpu_monitors:
            cpu_id = cpu_monitor.get_value("id", int)
            utilization_rate.update(
                {f"CPU {cpu_id}": cpu_monitor.get_values("coreUtilizationRate", float, Plot.to_percentage)})

        for cpu_core_monitor in self.cpu_core_monitors:
            cpu_id = cpu_core_monitor.get_value("cpuID", int)
            core_id = cpu_core_monitor.get_value("coreID", int)
            utilization_rate.update({
                f"CPU {cpu_id} Core {core_id}": cpu_core_monitor.get_values(
                    "coreUtilizationRate", float, Plot.to_percentage)
            })

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            utilization_rate.update({
                f"GPU {gpu_id} Core": gpu_monitor.get_values("coreUtilizationRate", float, Plot.to_percentage),
                f"GPU {gpu_id} Memory": gpu_monitor.get_values("memoryUtilizationRate", float, Plot.to_percentage)
            })

        return utilization_rate

    @property
    def utilization_rate_timeseries_plot(self):
        return TimeseriesPlot(title="Utilization Rate", plot_series=self.utilization_rate,
                              y_label="Utilization Rate (%)")

    @property
    def kernel_coordinates(self):
        kernel_coordinates = {}

        for gpu_monitor in self.gpu_monitors:
            gpu_id = gpu_monitor.get_value("id", int)
            kernel_coordinates.update({
                f"GPU {gpu_id} Block X": gpu_monitor.get_values("kernelBlockX", int),
                f"GPU {gpu_id} Block Y": gpu_monitor.get_values("kernelBlockY", int),
                f"GPU {gpu_id} Block Z": gpu_monitor.get_values("kernelBlockZ", int),
                f"GPU {gpu_id} Grid X": gpu_monitor.get_values("kernelGridX", int),
                f"GPU {gpu_id} Grid Y": gpu_monitor.get_values("kernelGridY", int),
                f"GPU {gpu_id} Grid Z": gpu_monitor.get_values("kernelGridZ", int)
            })

        return kernel_coordinates

    @property
    def kernel_coordinates_timeseries_plot(self):
        return TimeseriesPlot(title="Kernel Coordinates", plot_series=self.kernel_coordinates, y_label="Coordinate")
