import sqlite3
from typing import List, Dict


class Entity(object):
    # database = None
    # database_file = ""

    def _execute_sql(self, statement: str):
        if self.database is None:
            self.database = sqlite3.connect(self.database_file)

        cursor = self.database.cursor()
        cursor.execute(statement)

        return cursor.fetchall()

    def _on_save(self):
        pass

    def __init__(self, database_file=""):
        # Entity.database_file = database_file
        self.database = None
        self.database_file = database_file

    def _add_column(self, table: str, column: str, attributes: str):
        return self._execute_sql(f"ALTER TABLE {table} ADD {column} {attributes};")

    def _create_table(self, table: str, columns_with_attributes: Dict[str, str]):
        return self._execute_sql(
            f"CREATE TABLE {table}({','.join([column_with_attributes + ' ' + columns_with_attributes[column_with_attributes] for column_with_attributes in columns_with_attributes])});")

    def _insert(self, table: str, row_column_values):
        if isinstance(row_column_values, Dict):
            row_column_values: List[Dict[str, str]] = [row_column_values]

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

        return self._execute_sql(
            f"INSERT INTO {table}({','.join(columns)}) VALUES({'),('.join([','.join(row) for row in row_values])});")

    def _select(self, table: str, columns: List[str], conditions: str = None, order: str = None):
        return self._execute_sql(
            f"SELECT {','.join(columns)} FROM {table}" + (f" WHERE {conditions}" if conditions is not None else "") + (
                f" ORDER BY {order}" if order is not None else "") + ";")

    def save(self):
        self._on_save()
