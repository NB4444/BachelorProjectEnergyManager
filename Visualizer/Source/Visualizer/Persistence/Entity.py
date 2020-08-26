import sqlite3
from typing import List, Dict


class Entity(object):
    database = None
    database_file = ""

    @classmethod
    def _execute_sql(cls, statement: str):
        if Entity.database is None:
            Entity.database = sqlite3.connect(cls.database_file)

        cursor = cls.database.cursor()
        cursor.execute(statement)

        return cursor.fetchall()

    def _on_save(self):
        pass

    def __init__(self, database_file):
        Entity.database_file = database_file

    @classmethod
    def _add_column(cls, table: str, column: str, attributes: str):
        return cls._execute_sql(f"ALTER TABLE {table} ADD {column} {attributes};")

    @classmethod
    def _create_table(cls, table: str, columns_with_attributes: Dict[str, str]):
        return cls._execute_sql(f"CREATE TABLE {table}({','.join([column_with_attributes + ' ' + columns_with_attributes[column_with_attributes] for column_with_attributes in columns_with_attributes])});")

    @classmethod
    def _insert(cls, table: str, row_column_values):
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

        return cls._execute_sql(f"INSERT INTO {table}({','.join(columns)}) VALUES({'),('.join([','.join(row) for row in row_values])});")

    @classmethod
    def _select(cls, table: str, columns: List[str], conditions: str = None, order: str = None):
        return cls._execute_sql(f"SELECT {','.join(columns)} FROM {table}" + (f" WHERE {conditions}" if conditions is not None else "") + (f" ORDER BY {order}" if order is not None else "") + ";")

    def save(self):
        self._on_save()
