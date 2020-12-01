import collections
from datetime import datetime
from functools import cached_property
from typing import Dict, OrderedDict, Any, List, Type

from numpy import median, array, mean
from scipy.stats import mode

from Visualizer.Persistence.Entity import Entity
from Visualizer.Plotting.CorrelationsPlot import CorrelationsPlot
from Visualizer.Plotting.TablePlot import TablePlot
from Visualizer.Utility.Parsing import determine_type


class MonitorSession(Entity):
    @classmethod
    def load_all(cls, database_file: str):
        return cls._load(database_file)

    @classmethod
    def load_by_id(cls, database_file: str, id: int):
        return cls._load(database_file, f"id = {id}")

    @classmethod
    def load_by_profiler_session(cls, database_file: str, profiler_session: "ProfilerSession"):
        monitor_sessions = cls._load(database_file, f"profilerSessionID = {profiler_session.id}")
        for monitor_session in monitor_sessions:
            monitor_session.profiler_session = profiler_session

        return monitor_sessions

    @classmethod
    def _load(cls, database_file: str, conditions: str = None):
        Entity.database_file = database_file

        monitor_sessions = []
        for row in cls._select("MonitorSession", ["id", "monitorName"], conditions):
            id = row[0]
            monitor_name = row[1]

            monitor_sessions.append(MonitorSession(
                database_file,
                id,
                monitor_name
            ))

        return monitor_sessions

    def __init__(self, database_file: str, id: int, monitor_name: str, profiler_session: "ProfilerSession" = None):
        super().__init__(database_file)

        self.id = id
        self.monitor_name = monitor_name
        self.profiler_session = profiler_session

    @cached_property
    def monitor_data(self):
        monitor_data: OrderedDict[datetime, Dict[Any, Any]] = collections.OrderedDict()

        for row in self._select("MonitorData", ["timestamp", "name", "value"], f"monitorSessionID = {self.id}"):
            timestamp = datetime.fromtimestamp(float(row[0]) / 1000.0)
            name = row[1]
            value = row[2]

            if timestamp not in monitor_data:
                monitor_data[timestamp] = {}

            monitor_data[timestamp][name] = determine_type(value)(value)

        return monitor_data

    @cached_property
    def monitor_data_table(self):
        return TablePlot(title="Monitor Data",
                         table=[[self.monitor_name, timestamp, name, value] for timestamp, variables in
                                self.monitor_data.items() for name, value in variables.items()],
                         columns=["Monitor Name", "Timestamp", "Name", "Value"])

    @cached_property
    def namespaced_monitor_data(self):
        physical_id = self.get_value("id", int)
        id = physical_id if physical_id is not None else self.id

        # Make keys namespaced
        new_data = self.monitor_data.copy()
        for timestamp, variables in new_data.items():
            new_variables: Dict[str, Any] = {}
            for variable_name in variables.keys():
                new_variables[f"{self.monitor_name}.{id}.{variable_name}"] = variables[variable_name]
            new_data[timestamp] = new_variables

        return new_data

    @classmethod
    def horizontal_table(cls, monitor_sessions: List["MonitorSession"]):
        combined_monitor_data: OrderedDict[datetime, Dict[str, Any]] = None
        for monitor_session in monitor_sessions:
            namespaced_data = monitor_session.namespaced_monitor_data

            # Append data
            if combined_monitor_data is None:
                combined_monitor_data = namespaced_data
            else:
                combined_monitor_data.update(namespaced_data)

        # Re-order the data
        combined_monitor_data = collections.OrderedDict(sorted(combined_monitor_data.items()))

        columns = ["Timestamp"] + sorted(
            list(set([name for _, data in combined_monitor_data.items() for name, _ in data.items()])))

        return TablePlot(
            title="Monitor Data",
            table=[[timestamp] + [data[column] if column in data else float("NaN") for column in columns[1:]] for
                   timestamp, data in combined_monitor_data.items()],
            columns=columns,
            interpolate=True
        )

    @classmethod
    def correlations_plot(cls, monitor_sessions: List["MonitorSession"]):
        horizontal_table = cls.horizontal_table(monitor_sessions)

        return CorrelationsPlot(title="Monitor Variable Correlations",
                                correlations=horizontal_table.pandas_table._get_numeric_data().corr())

    def get_values(self, name: str, type: Type, modifier=lambda value: value):
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

    def get_first_value(self, name: str, type: Type, modifier=lambda value: value):
        _, variables = list(self.monitor_data.items())[0]
        return modifier(type(variables[name])) if name in variables else None

    def get_last_value(self, name: str, type: Type, modifier=lambda value: value):
        _, variables = list(self.monitor_data.items())[-1]
        return modifier(type(variables[name])) if name in variables else None

    def get_value(self, name: str, type: Type, modifier=lambda value: value):
        return self.get_first_value(name, type, modifier)

    def get_mean_value(self, name: str, type: Type, modifier=lambda value: value):
        return mean(array(self.get_values(name, type, modifier)))

    def get_median_value(self, name: str, type: Type, modifier=lambda value: value):
        return median(array(self.get_values(name, type, modifier)))

    def get_mode_value(self, name: str, type: Type, modifier=lambda value: value):
        return mode(array(self.get_values(name, type, modifier)))

    def get_maximum_value(self, name: str, type: Type, modifier=lambda value: value):
        return max(self.get_value(name, type, modifier))

    def get_minimum_value(self, name: str, type: Type, modifier=lambda value: value):
        return min(self.get_value(name, type, modifier))

    def get_indexed_values(self, name: str, type: Type, modifier=lambda value: value):
        index = 0

        results = list()
        while True:
            values = self.get_values(name + str(index), type, modifier)

            if len(values) == 0:
                break
            else:
                results.append(values)
                index += 1

        return results

    def get_indexed_value_series(self, series_name_prefix: str, name: str, type: Type, modifier=lambda value: value):
        series = {}

        for index, values in enumerate(self.get_indexed_values(name, type, modifier)):
            series[f"{series_name_prefix} {index}"] = values

        return series

    def get_summarized_indexed_value_series(self, summarized_series_name: str, series_name_prefix: str,
                                            summarized_variable_name: str, variable_name: str, type: Type,
                                            modifier=lambda value: value):
        series = {f"{summarized_series_name}": self.get_values(summarized_variable_name, type, modifier)}
        series.update(self.get_indexed_value_series(series_name_prefix, variable_name, type, modifier))

        return series
