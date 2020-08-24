import sqlite3
from typing import List, Dict


class Entity(object):
    def __init__(self, database_file):
        self.database = sqlite3.connect(database_file)

    def execute(self, statement: str):
        self.database.cursor().execute(statement)

    def insert(self, table: str, row_column_values: List[Dict[str, str]]):
        # Collect columns
        columns: List[str] = list()
        for column_values in row_column_values:
            for column_value in column_values:
                columns.append(column_value)
        columns = sorted(set(columns))

        # Collect values
        row_values: List[List[str]] = list()
        for column_values in row_column_values:
            insert_values = list()

            for column in columns:
                insert_values.append(column_values[column])

            row_values.append(insert_values)

        self.execute(f"INSERT INTO {table}({','.join(columns)}) VALUES({'),('.join([','.join(row) for row in row_values])});")

    def insert(self, table: str, column_values: Dict[str, str]):
        self.insert(table, [column_values])

    def create_table(self, table: str, columns_with_attributes: Dict[str, str]):
        self.execute(f"CREATE TABLE {table}({','.join([column_with_attributes + ' ' + columns_with_attributes[column_with_attributes] for column_with_attributes in columns_with_attributes])});")

    def add_column(self, table: str, column: str, attributes: str):
        self.execute(f"ALTER TABLE {table} ADD {column} {attributes};")

    def save(self):
        self._