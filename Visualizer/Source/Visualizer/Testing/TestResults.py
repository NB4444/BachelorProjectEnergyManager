import collections
import datetime
from functools import cached_property
from typing import Dict, OrderedDict, Any

from Visualizer import Plotting
from Visualizer.Persistence.Entity import Entity


class TestResults(Entity):
    def __init__(self, database_file: str, test: "Test"):
        super().__init__(database_file)

        self.test = test

    @cached_property
    def results(self) -> Dict[str, Any]:
        # Retrieve the test results
        test_results: Dict[str, str] = dict()
        for row in self._select("TestResults", ["name", "value"], f"testID = {self.test.id}"):
            # Keep track of the current result
            name = row[0]
            value = row[1]

            # Store the results
            test_results[name] = value

        return test_results

    @cached_property
    def monitor_results(self) -> Dict[str, OrderedDict[datetime.datetime, Dict[str, Any]]]:
        # Retrieve the monitor results
        monitor_results: Dict[str, OrderedDict[datetime.datetime, Dict[str, str]]] = dict()
        for row in self._select("MonitorResults", ["monitor", "timestamp", "name", "value"], f"testID = {self.test.id}", "timestamp ASC"):
            # Keep track of the current result
            monitor = row[0]
            timestamp = datetime.datetime.fromtimestamp(float(row[1]) / 1000.0)
            name = row[2]
            value = row[3]

            # Create the necessary data structures
            if monitor not in monitor_results:
                monitor_results[monitor] = collections.OrderedDict()

            if timestamp not in monitor_results[monitor]:
                monitor_results[monitor][timestamp] = dict()

            # Try to determine the value type
            def determine_type(value):
                for (type, condition) in [
                    (int, int),
                    (float, float),
                    (datetime, lambda value: datetime.datetime.strptime(value, "%Y/%m/%d"))
                ]:
                    try:
                        condition(value)
                        return type
                    except ValueError:
                        continue

                return str

            # Store the results
            monitor_results[monitor][timestamp][name] = determine_type(value)(value)

        return monitor_results

    @cached_property
    def monitor_results_table(self):
        return self.get_monitor_results_table()

    def get_monitor_results_table(self, maximum_column_width: int = None, maximum_columns: int = None, minimum_rows: int = None, maximum_rows: int = 50):
        return Plotting.plot_table(
            data=[[timestamp, monitor_name, name, value] for monitor_name, monitor_data in self.monitor_results.items() for timestamp, data in monitor_data.items() for name, value in data.items()],
            columns=["Timestamp", "Monitor", "Name", "Value"],
            maximum_column_width=maximum_column_width,
            maximum_columns=maximum_columns,
            minimum_rows=minimum_rows,
            maximum_rows=maximum_rows
        )

    @cached_property
    def data_table(self):
        # Transform the data into a better format
        transformed_data: Dict[int, Dict[str, Any]] = collections.OrderedDict()
        for monitor_name, monitor_data in self.monitor_results.items():
            for timestamp, data in monitor_data.items():
                if timestamp not in transformed_data:
                    transformed_data[timestamp] = dict()

                for name, value in data.items():
                    transformed_data[timestamp][f"{monitor_name}.{name}"] = value

        transformed_data = collections.OrderedDict(sorted(transformed_data.items()))

        # Retrieve the columns and rows
        columns = ["Timestamp"] + sorted(list(set([name for _, data in transformed_data.items() for name, _ in data.items()])))
        rows = [[timestamp] + [(data[column] if column in data else float("NaN")) for column in columns[1:]] for timestamp, data in transformed_data.items()]

        return Plotting.plot_table(
            data=rows,
            columns=columns
        ).interpolate(method="linear", limit_direction="both")

    def __collect_values(self, monitor: str, name: str, type, modifier=lambda value: value):
        values: OrderedDict[int, type] = collections.OrderedDict()
        try:
            for timestamp, results in self.monitor_results[monitor].items():
                try:
                    values[timestamp] = modifier(type(results[name]))
                except:
                    pass
        except:
            pass

        return values

    def __collect_indexed_values(self, monitor: str, name: str, type, modifier=lambda value: value):
        index = 0

        results = list()

        while True:
            values = self.__collect_values(monitor, name + str(index), type, modifier)

            if len(values) == 0:
                break
            else:
                results.append(values)
                index += 1

        return results

    def __collect_indexed_series(self, series_name_prefix: str, monitor: str, name: str, type, modifier=lambda value: value):
        series = {}

        for index, values in enumerate(self.__collect_indexed_values(monitor, name, type, modifier)):
            series[(f"{series_name_prefix} {index}", "")] = values

        return series

    def __collect_summarized_indexed_values(self, summarized_series_name: str, series_name_prefix: str, monitor: str, summarized_name: str, name: str, type,
                                            modifier=lambda value: value):
        series = {(f"{summarized_series_name}", ""): self.__collect_values(monitor, summarized_name, type, modifier)}
        series.update(self.__collect_indexed_series(series_name_prefix, monitor, name, type, modifier))

        return series

    def __collect_constant_value(self, monitor: str, name: str, type, modifier=lambda value: value):
        values = self.__collect_values(monitor, name, type, modifier).items()
        return list(values)[-1][1] if len(values) > 0 else None

    def overview_plot(self, output_directory: str = None):
        Plotting.plot_timeseries_overview(summary_plot_generator=self.summary_plot, plot_generators=[
            self.clock_rate_plot,
            self.energy_consumption_plot,
            self.fan_speed_plot,
            self.memory_consumption_plot,
            self.power_consumption_plot,
            self.processes_plot,
            self.switches_plot,
            self.temperature_plot,
            self.timespan_plot,
            self.utilization_rate_plot,
            self.kernel_coordinates_plot,
            # self.correlations_plot,
        ], output_directory=output_directory)

    @cached_property
    def summary(self):
        return {
            "Name": self.test.name,
            "GPU Brand": self.__collect_constant_value("GPUMonitor", "brand", str),
            "GPU Compute Capability Major Version": self.__collect_constant_value("GPUMonitor", "computeCapabilityMajorVersion", int),
            "GPU Compute Capability Minor Version": self.__collect_constant_value("GPUMonitor", "computeCapabilityMinorVersion", int),
            # "GPU Memory Bandwidth (B/s)": self.__collect_constant_value("GPUMonitor", "memoryBandwidth", float, Plotting.humanize_size),
            "GPU Memory Size (B)": self.__collect_constant_value("GPUMonitor", "memorySize", int, Plotting.humanize_size),
            # "GPU Multiprocessor Count": self.__collect_constant_value("GPUMonitor", "multiprocessorCount", int, Plotting.humanize_number),
            "GPU Name": self.__collect_constant_value("GPUMonitor", "name", str),
            "GPU PCIe Link Width (B)": self.__collect_constant_value("GPUMonitor", "pciELinkWidth", int, Plotting.humanize_size),
            "GPU Default Power Limit (W)": self.__collect_constant_value("GPUMonitor", "defaultPowerLimit", float, Plotting.humanize_number),
            "GPU Supported Core Clock Rates (Hz)": self.__collect_constant_value("GPUMonitor", "supportedCoreClockRates", str, Plotting.parse_number_list),
            "GPU Supported Memory Clock Rates (Hz)": self.__collect_constant_value("GPUMonitor", "supportedMemoryClockRates", str, Plotting.parse_number_list),
            "GPU Default Auto Boosted Clocks Enabled": self.__collect_constant_value("GPUMonitor", "defaultAutoBoostedClocksEnabled", bool),
        }

    def summary_plot(self, figure=None, axes=None):
        return Plotting.plot_dictionary(self.summary, figure, axes)

    @cached_property
    def clock_rate(self):
        clock_rate = self.__collect_summarized_indexed_values("CPU", "CPU Core", "CPUMonitor", "coreClockRate", "coreClockRateCore", int)
        clock_rate.update({
            ("GPU Core", ""): self.__collect_values("GPUMonitor", "coreClockRate", int),
            ("GPU Memory", ""): self.__collect_values("GPUMonitor", "memoryClockRate", int),
            ("GPU SM", ""): self.__collect_values("GPUMonitor", "streamingMultiprocessorClockRate", int),
        })

        return clock_rate

    @cached_property
    def clock_rate_limits(self):
        clock_rate_limits = self.__collect_summarized_indexed_values("CPU Maximum", "CPU Maximum Core", "CPUMonitor", "maximumCoreClockRate", "maximumCoreClockRateCore", int)
        clock_rate_limits.update({
            ("GPU Core Maximum", ""): self.__collect_values("GPUMonitor", "maximumCoreClockRate", int),
            ("GPU Memory Maximum", ""): self.__collect_values("GPUMonitor", "maximumMemoryClockRate", int),
        })

        return clock_rate_limits

    def clock_rate_plot(self, plot_limits=True, figure=None, axes=None, output_directory: str = None):
        series = self.clock_rate
        if plot_limits:
            series.update(self.clock_rate_limits)

        return Plotting.plot_timeseries(title="Clock Rate", plot_series=series, figure=figure, axes=axes, y_label="Clock Rate (Hz)", output_directory=output_directory)

    @cached_property
    def energy_consumption(self):
        return self.get_energy_consumption(None)

    def get_energy_consumption(self, modifier):
        return {
            ("CPU", ""): self.__collect_values("CPUMonitor", "energyConsumption", float, modifier),
            ("GPU", ""): self.__collect_values("GPUMonitor", "energyConsumption", float, modifier),
            ("Node", ""): self.__collect_values("NodeMonitor", "energyConsumption", float, modifier),
        }

    def energy_consumption_plot(self, figure=None, axes=None, unit_string="J", modifier=lambda value: value, output_directory: str = None):
        return Plotting.plot_timeseries(title="Energy Consumption", plot_series=self.get_energy_consumption(modifier), figure=figure, axes=axes, y_label=f"Energy Consumption ({unit_string})",
                                        output_directory=output_directory)

    @cached_property
    def fan_speed(self):
        return self.__collect_summarized_indexed_values("GPU", "GPU Fan", "GPUMonitor", "fanSpeed", "fanSpeedFan", float, Plotting.to_percentage)

    def fan_speed_plot(self, figure=None, axes=None, output_directory: str = None):
        return Plotting.plot_timeseries(title="Fan Speed", plot_series=self.fan_speed, figure=figure, axes=axes, y_label="Fan Speed (%)", output_directory=output_directory)

    @cached_property
    def memory_consumption(self):
        return {
            ("GPU Free", ""): self.__collect_values("GPUMonitor", "memoryFreeSize", int),
            ("GPU Used", ""): self.__collect_values("GPUMonitor", "memoryUsedSize", int),
            ("RAM Free", ""): self.__collect_values("NodeMonitor", "freeMemorySize", int),
            ("RAM Used", ""): self.__collect_values("NodeMonitor", "usedMemorySize", int),
            ("Swap Free", ""): self.__collect_values("NodeMonitor", "freeSwapMemorySize", int),
            ("Swap Used", ""): self.__collect_values("NodeMonitor", "usedSwapMemorySize", int),
            ("High Free", ""): self.__collect_values("NodeMonitor", "freeHighMemorySize", int),
            ("High Used", ""): self.__collect_values("NodeMonitor", "usedHighMemorySize", int),
        }

    @cached_property
    def memory_sizes(self):
        return {
            ("GPU Kernel Dynamic Shared", ""): self.__collect_values("GPUMonitor", "kernelDynamicSharedMemorySize", int),
            ("GPU Kernel Static Shared", ""): self.__collect_values("GPUMonitor", "kernelStaticSharedMemorySize", int),
            ("GPU", ""): self.__collect_values("GPUMonitor", "memorySize", int),
            ("GPU PCIe Link", ""): self.__collect_values("GPUMonitor", "pciELinkWidth", int),
            ("RAM", ""): self.__collect_values("NodeMonitor", "memorySize", int),
            ("RAM Shared", ""): self.__collect_values("NodeMonitor", "sharedMemorySize", int),
            ("RAM Buffer", ""): self.__collect_values("NodeMonitor", "bufferMemorySize", int),
            ("Swap", ""): self.__collect_values("NodeMonitor", "swapMemorySize", int),
            ("High", ""): self.__collect_values("NodeMonitor", "highMemorySize", int),
        }

    def memory_consumption_plot(self, plot_sizes=True, figure=None, axes=None, output_directory: str = None):
        series = self.memory_consumption
        if plot_sizes:
            series.update(self.memory_sizes)

        return Plotting.plot_timeseries(title="Memory Consumption", plot_series=series, figure=figure, axes=axes, y_label="Memory Consumption (B)", output_directory=output_directory)

    @cached_property
    def power_consumption(self):
        return {
            ("CPU", ""): self.__collect_values("CPUMonitor", "powerConsumption", float),
            ("GPU", ""): self.__collect_values("GPUMonitor", "powerConsumption", float),
            ("Node", ""): self.__collect_values("NodeMonitor", "powerConsumption", float),
        }

    @cached_property
    def power_limits(self):
        return {
            ("GPU Power Limit", ""): self.__collect_values("GPUMonitor", "powerLimit", float),
            ("GPU Enforced Power Limit", ""): self.__collect_values("GPUMonitor", "powerLimit", float),
        }

    def power_consumption_plot(self, plot_limits=True, figure=None, axes=None, output_directory: str = None):
        series = self.power_consumption
        if plot_limits:
            series.update(self.power_limits)

        return Plotting.plot_timeseries(title="Power Consumption", plot_series=series, figure=figure, axes=axes, y_label="Power Consumption (W)", output_directory=output_directory)

    @cached_property
    def processes(self):
        return {
            ("Processes", ""): self.__collect_values("NodeMonitor", "processCount", int)
        }

    def processes_plot(self, figure=None, axes=None, output_directory: str = None):
        return Plotting.plot_timeseries(title="Processes", plot_series=self.processes, figure=figure, axes=axes, y_label="Processes", output_directory=output_directory)

    @cached_property
    def switches(self):
        return {
            ("GPU Auto Boosted Clocks", ""): self.__collect_values("GPUMonitor", "autoBoostedClocksEnabled", bool),
        }

    def switches_plot(self, figure=None, axes=None, output_directory: str = None):
        return Plotting.plot_timeseries(title="Switches", plot_series=self.switches, figure=figure, axes=axes, y_label="Switches (bool)", output_directory=output_directory)

    @cached_property
    def temperature(self):
        return {
            ("CPU", ""): self.__collect_values("CPUMonitor", "temperature", float),
            ("GPU", ""): self.__collect_values("GPUMonitor", "temperature", float),
        }

    def temperature_plot(self, figure=None, axes=None, output_directory: str = None):
        return Plotting.plot_timeseries(title="Temperature", plot_series=self.temperature, figure=figure, axes=axes, y_label="Temperature (C)", output_directory=output_directory)

    @cached_property
    def timespan(self):
        timespan = self.__collect_summarized_indexed_values("CPU Guest Nice", "CPU Guest Nice Core", "CPUMonitor", "guestNiceTimespan", "guestNiceTimespanCore", float, Plotting.to_percentage)
        timespan.update(self.__collect_summarized_indexed_values("CPU Guest", "CPU Guest Core", "CPUMonitor", "guestTimespan", "guestTimespanCore", float, Plotting.to_percentage))
        timespan.update(self.__collect_summarized_indexed_values("CPU IO Wait", "CPU IO Wait Core", "CPUMonitor", "ioWaitTimespan", "ioWaitTimespanCore", float, Plotting.to_percentage))
        timespan.update(self.__collect_summarized_indexed_values("CPU Idle", "CPU Idle Core", "CPUMonitor", "idleTimespan", "idleTimespanCore", float, Plotting.to_percentage))
        timespan.update(
            self.__collect_summarized_indexed_values("CPU Interrupts", "CPU Interrupts Core", "CPUMonitor", "interruptsTimespan", "interruptsTimespanCore", float, Plotting.to_percentage))
        timespan.update(self.__collect_summarized_indexed_values("CPU Nice", "CPU Nice Core", "CPUMonitor", "niceTimespan", "niceTimespanCore", float, Plotting.to_percentage))
        timespan.update(self.__collect_summarized_indexed_values("CPU Soft Interrupts", "CPU Soft Interrupts Core", "CPUMonitor", "softInterruptsTimespan", "softInterruptsTimespanCore", float,
                                                                 Plotting.to_percentage))
        timespan.update(self.__collect_summarized_indexed_values("CPU Steal", "CPU Steal Core", "CPUMonitor", "stealTimespan", "stealTimespanCore", float, Plotting.to_percentage))
        timespan.update(self.__collect_summarized_indexed_values("CPU System", "CPU System Core", "CPUMonitor", "systemTimespan", "systemTimespanCore", float, Plotting.to_percentage))
        timespan.update(self.__collect_summarized_indexed_values("CPU User", "CPU User Core", "CPUMonitor", "userTimespan", "userTimespanCore", float, Plotting.to_percentage))
        timespan[("Runtime", "")] = self.__collect_values("NodeMonitor", "runtime", float, Plotting.ns_to_s)

        return timespan

    def timespan_plot(self, figure=None, axes=None, output_directory: str = None):
        return Plotting.plot_timeseries(title="Timespan", plot_series=self.timespan, figure=figure, axes=axes, y_label="Timespan (s)", output_directory=output_directory)

    @cached_property
    def utilization_rate(self):
        utilization_rate = self.__collect_summarized_indexed_values("CPU", "CPU Core", "CPUMonitor", "coreUtilizationRate", "coreUtilizationRateCore", float, Plotting.to_percentage)
        utilization_rate.update({
            ("GPU Core", ""): self.__collect_values("GPUMonitor", "coreUtilizationRate", float, Plotting.to_percentage),
            ("GPU Memory", ""): self.__collect_values("GPUMonitor", "memoryUtilizationRate", float, Plotting.to_percentage),
        })

        return utilization_rate

    def utilization_rate_plot(self, figure=None, axes=None, output_directory: str = None):
        return Plotting.plot_timeseries(title="Utilization Rate", plot_series=self.utilization_rate, figure=figure, axes=axes, y_label="Utilization Rate (%)", output_directory=output_directory)

    @cached_property
    def kernel_coordinates(self):
        return {
            ("Block X", ""): self.__collect_values("GPUMonitor", "kernelBlockX", int),
            ("Block Y", ""): self.__collect_values("GPUMonitor", "kernelBlockY", int),
            ("Block Z", ""): self.__collect_values("GPUMonitor", "kernelBlockZ", int),
            ("Grid X", ""): self.__collect_values("GPUMonitor", "kernelGridX", int),
            ("Grid Y", ""): self.__collect_values("GPUMonitor", "kernelGridY", int),
            ("Grid Z", ""): self.__collect_values("GPUMonitor", "kernelGridZ", int),
        }

    def kernel_coordinates_plot(self, figure=None, axes=None, output_directory: str = None):
        return Plotting.plot_timeseries(title="Kernel Coordinates", plot_series=self.kernel_coordinates, figure=figure, axes=axes, y_label="Coordinate", output_directory=output_directory)

    def correlations_plot(self, figure=None, axes=None, output_directory: str = None):
        return Plotting.plot_correlations(self.data_table._get_numeric_data().corr(), figure=figure, axes=axes, output_directory=output_directory)
