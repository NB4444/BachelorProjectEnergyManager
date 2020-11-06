import collections
from datetime import datetime
from typing import Dict, OrderedDict, Any, List

from Visualizer.Persistence.Entity import Entity
from Visualizer.Plotting.CorrelationsPlot import CorrelationsPlot
from Visualizer.Plotting.TablePlot import TablePlot


class MonitorData(Entity):
    @classmethod
    def load_all(cls, database_file: str):
        return cls._load(database_file)

    @classmethod
    def load_by_id(cls, database_file: str, id: int):
        return cls._load(database_file, f"id = {id}")

    @classmethod
    def load_by_profiler_session(cls, database_file: str, profiler_session: "ProfilerSession"):
        monitor_data = cls._load(database_file, f"profilerSessionID = {profiler_session.id}")
        for monitor_datum in monitor_data:
            monitor_datum.profiler_session = profiler_session

        return monitor_data

    @classmethod
    def _load(cls, database_file: str, conditions: str = None):
        Entity.database_file = database_file

        monitor_data: Dict[str, OrderedDict[datetime, Dict[str, Any]]] = {}
        for row in cls._select("MonitorData", ["monitorName", "timestamp", "name", "value"], conditions):
            monitor_name = row[0]
            timestamp = datetime.fromtimestamp(float(row[1]) / 1000.0)
            name = row[2]
            value = row[3]

            if monitor_name not in monitor_data:
                monitor_data[monitor_name] = collections.OrderedDict()

            if timestamp not in monitor_data[monitor_name]:
                monitor_data[monitor_name][timestamp] = {}

            # Try to determine the value type
            def determine_type(value):
                for (type, condition) in [
                    (int, int),
                    (float, float),
                    (datetime, lambda value: datetime.strptime(value, "%Y/%m/%d"))
                ]:
                    try:
                        condition(value)
                        return type
                    except ValueError:
                        continue

                return str

            monitor_data[monitor_name][timestamp][name] = determine_type(value)(value)

        result = []
        for current_monitor_name, current_monitor_data in monitor_data.items():
            result.append(MonitorData(
                database_file,
                current_monitor_data,
                current_monitor_name
            ))

        return result

    def __init__(self, database_file: str, monitor_data: OrderedDict[datetime, Dict[str, Any]], monitor_name: str, profiler_session: "ProfilerSession" = None):
        super().__init__(database_file)

        self.monitor_data = monitor_data
        self.monitor_name = monitor_name
        self.profiler_session = profiler_session

    def table(self):
        return TablePlot(title="Monitor Data", table=[[self.monitor_name, timestamp, name, value] for timestamp, variables in self.monitor_data.items() for name, value in variables.items()],
                         columns=["Monitor Name", "Timestamp", "Name", "Value"])

    @property
    def namespaced_monitor_data(self):
        # Make keys namespaced
        new_data = self.monitor_data.copy()
        for timestamp, variables in new_data.items():
            new_variables: Dict[str, Any] = {}
            for variable_name in variables.keys():
                new_variables[f"{self.monitor_name}.{variable_name}"] = variables[variable_name]
            new_data[timestamp] = new_variables

        return new_data

    @classmethod
    def horizontal_table(cls, monitor_data: List["MonitorData"]):
        combined_monitor_data: OrderedDict[datetime, Dict[str, Any]] = None
        for monitor_datum in monitor_data:
            namespaced_data = monitor_datum.namespaced_monitor_data

            # Append data
            if combined_monitor_data is None:
                combined_monitor_data = namespaced_data
            else:
                combined_monitor_data.update(namespaced_data)

        # Re-order the data
        combined_monitor_data = collections.OrderedDict(sorted(combined_monitor_data.items()))

        columns = ["Timestamp"] + sorted(list(set([name for _, data in combined_monitor_data.items() for name, _ in data.items()])))

        return TablePlot(
            title="Monitor Data",
            table=[[timestamp] + [data[column] if column in data else float("NaN") for column in columns[1:]] for timestamp, data in combined_monitor_data.items()],
            columns=columns,
            interpolate=True
        )

    @classmethod
    def correlations_plot(cls, monitor_data: List["MonitorData"]):
        horizontal_table = cls.horizontal_table(monitor_data)

        return CorrelationsPlot(title="Monitor Variable Correlations", correlations=horizontal_table.pandas_table._get_numeric_data().corr())

    def collect_values(self, name: str, type, modifier=lambda value: value):
        values: OrderedDict[datetime, type] = collections.OrderedDict()
        try:
            for timestamp, results in self.monitor_data.items():
                try:
                    values[timestamp] = modifier(type(results[name]))
                except:
                    pass
        except:
            pass

        return values

    def collect_indexed_values(self, name: str, type, modifier=lambda value: value):
        index = 0

        results = list()

        while True:
            values = self.collect_values(name + str(index), type, modifier)

            if len(values) == 0:
                break
            else:
                results.append(values)
                index += 1

        return results

    def collect_indexed_series(self, series_name_prefix: str, name: str, type, modifier=lambda value: value):
        series = {}

        for index, values in enumerate(self.collect_indexed_values(name, type, modifier)):
            series[f"{series_name_prefix} {index}"] = values

        return series

    def collect_summarized_indexed_values(self, summarized_series_name: str, series_name_prefix: str, summarized_name: str, name: str, type,
                                          modifier=lambda value: value):
        series = {f"{summarized_series_name}": self.collect_values(summarized_name, type, modifier)}
        series.update(self.collect_indexed_series(series_name_prefix, name, type, modifier))

        return series

    def collect_constant_value(self, name: str, type, modifier=lambda value: value):
        values = self.collect_values(name, type, modifier).items()
        return list(values)[-1][1] if len(values) > 0 else None
