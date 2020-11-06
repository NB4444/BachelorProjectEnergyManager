from typing import List

from Visualizer.Monitoring.Persistence.ProfilerSession import ProfilerSession
from Visualizer.Persistence.Entity import Entity
from Visualizer.Plotting.TablePlot import TablePlot
from Visualizer.Testing.Persistence.TestResults import TestResults


class TestSession(Entity):
    @classmethod
    def load_all(cls, database_file: str):
        return cls._load(database_file)

    @classmethod
    def load_by_id(cls, database_file: str, id: int):
        return cls._load(database_file, f"id = {id}")[0]

    @classmethod
    def _load(cls, database_file: str, conditions: str = None):
        Entity.database_file = database_file

        test_sessions = []
        for row in cls._select("TestSession", ["id", "testName", "profilerSessionID"], conditions):
            id = row[0]
            test_name = row[1]
            profiling_session_id = row[2]

            test_session = TestSession(
                database_file,
                id,
                test_name,
                None,
                ProfilerSession.load_by_id(database_file, profiling_session_id)
            )
            test_session.test_results = TestResults.load_by_test_session(database_file, test_session)
            test_sessions.append(test_session)

        return test_sessions

    @classmethod
    def table(cls, test_sessions: List["TestSession"]):
        return TablePlot(title="Test Sessions", table=[[test_session.id, test_session.test_name, test_session.profiler_session.id] for test_session in test_sessions],
                         columns=["ID", "Test Name", "Profiler Session ID"])

    def __init__(self, database_file: str, id: int, test_name: str, test_results: "TestResults" = None, profiler_session: ProfilerSession = None):
        super().__init__(database_file)

        self.id = id
        self.test_name = test_name
        self.test_results = test_results
        self.profiler_session = profiler_session
