import collections
import datetime
from typing import Dict, Tuple, OrderedDict

from Visualizer.Persistence.Entity import Entity


class TestResults(Entity):
    @classmethod
    def load_by_test_id(cls, database_file: str, testID: int):
        Entity.database_file = database_file

        # Keep track of the results
        test_results: Dict[str, str] = dict()
        monitor_results: Dict[str, OrderedDict[datetime.datetime, Dict[str, str]]] = dict()

        # Retrieve the test results
        for row in cls._select("TestResults", ["name", "value"], f"testID = {testID}"):
            # Keep track of the current result
            name = row[0]
            value = row[1]

            # Store the results
            test_results[name] = value

        # Retrieve the monitor results
        for row in cls._select("MonitorResults", ["monitor", "timestamp", "name", "value"], f"testID = {testID}", "timestamp ASC"):
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

            # Store the results
            monitor_results[monitor][timestamp][name] = value

        return TestResults(database_file, test_results, monitor_results)

    def __init__(self, database_file: str, results: Dict[str, str], monitor_results: Dict[str, OrderedDict[datetime.datetime, Dict[str, str]]]):
        super().__init__(database_file)

        self.results = results
        self.monitor_results = monitor_results
