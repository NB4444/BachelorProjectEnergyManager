from typing import Dict

from Visualizer.Persistence.Entity import Entity
from Visualizer.Plotting.TablePlot import TablePlot


class TestResults(Entity):
    @classmethod
    def load_all(cls, database_file: str):
        return cls._load(database_file)

    @classmethod
    def load_by_id(cls, database_file: str, id: int):
        return cls._load(database_file, f"id = {id}")

    @classmethod
    def load_by_test_session(cls, database_file: str, test_session: "TestSession"):
        test_results = cls._load(database_file, f"testSessionID = {test_session.id}")
        test_results.test_session = test_session

        return test_results

    @classmethod
    def _load(cls, database_file: str, conditions: str = None):
        Entity.database_file = database_file

        # Retrieve the test results
        test_results: Dict[str, str] = {}
        for row in cls._select("TestResults", ["name", "value"], conditions):
            # Keep track of the current result
            name = row[0]
            value = row[1]

            # Store the results
            test_results[name] = value

        return TestResults(database_file, test_results)

    def __init__(self, database_file: str, test_results: Dict[str, str], test_session: "TestSession" = None):
        super().__init__(database_file)

        self.test_results = test_results
        self.test_session = test_session

    def table(self):
        return TablePlot(title="Results", table=[[name, value] for name, value in self.test_results.items()], columns=["Name", "Value"])
