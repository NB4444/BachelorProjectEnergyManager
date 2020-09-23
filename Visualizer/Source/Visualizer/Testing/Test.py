from functools import cached_property
from typing import List

from Visualizer import Plotting
from Visualizer.Persistence.Entity import Entity
from Visualizer.Testing.TestResults import TestResults


class Test(Entity):
    @classmethod
    def load_all(cls, database_file: str):
        Entity.database_file = database_file

        # Retrieve and process the test rows
        tests = list()
        for row in cls._select("Tests", ["id", "name"]):
            # Retrieve and process the test result rows
            id = row[0]
            name = row[1]

            tests.append(Test(
                database_file,
                id,
                name,
            ))

        return tests

    @classmethod
    def tests_table(cls, tests: List["Test"]):
        table = Plotting.plot_table(
            data=[[test.id, test.name] for test in tests],
            columns=["ID", "Name"]
        )
        table

        return table

    def __init__(self, database_file: str, id: int, name: str):
        super().__init__(database_file)

        self.id = id
        self.name = name

    @cached_property
    def test_results(self) -> TestResults:
        return TestResults(self.database_file, self)
