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
                TestResults.load_by_test_id(database_file, id)
            ))

        return tests

    def __init__(self, database_file: str, id: int, name: str, test_results: TestResults):
        super().__init__(database_file)

        self.id = id
        self.name = name
        self.test_results = test_results
