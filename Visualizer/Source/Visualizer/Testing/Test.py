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
            tests.append(Test(
                database_file,
                row[1],
                TestResults.load_by_test_id(database_file, row[0])
            ))

        return tests

    def __init__(self, database_file: str, name: str, test_results: TestResults):
        super().__init__(database_file)

        self.name = name
        self.test_results = test_results
