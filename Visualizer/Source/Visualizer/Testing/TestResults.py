import collections
import datetime
from typing import Dict, Tuple, OrderedDict

from Visualizer.Persistence.Entity import Entity


class TestResults(Entity):
    @classmethod
    def load_all(cls, database_file: str):
        Entity.database_file = database_file

        # Keep track of the results
        test_results: Dict[str, Tuple[Dict[str, str], Dict[str, OrderedDict[datetime.datetime, Dict[str, str]]]]] = dict()

        # Retrieve the test results for all tests
        for row in cls.select("TestResults", ["test", "name", "value"]):
            # Keep track of the current result
            test = row[0]
            name = row[1]
            value = row[2]

            # Create the necessary data structures
            if test not in test_results:
                test_results[test] = (dict(), dict())

            # Store the results
            test_results[test][0][name] = value

        # Retrieve the monitor results for all tests
        for row in cls.select("MonitorResults", ["test", "monitor", "timestamp", "name", "value"], None, "timestamp ASC"):
            # Keep track of the current result
            test = row[0]
            monitor = row[1]
            timestamp = datetime.datetime.fromtimestamp(float(row[2]) / 1000.0)
            name = row[3]
            value = row[4]

            # Create the necessary data structures
            if test not in test_results:
                test_results[test] = (dict(), dict())

            if monitor not in test_results[test][1]:
                test_results[test][1][monitor] = collections.OrderedDict()

            if timestamp not in test_results[test][1][monitor]:
                test_results[test][1][monitor][timestamp] = dict()

            # Store the results
            test_results[test][1][monitor][timestamp][name] = value

        return [TestResults(database_file, test, test_results[test][0], test_results[test][1]) for test in test_results]

    def __init__(self, database_file: str, test: str, results: Dict[str, str], monitor_results: Dict[str, OrderedDict[datetime.datetime, Dict[str, str]]]):
        super().__init__(database_file)

        self.test = test
        self.results = results
        self.monitor_results = monitor_results
