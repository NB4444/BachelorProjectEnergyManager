from typing import Any, List

import pandas
from IPython.display import display

from Visualizer.Plotting.Plot import Plot


class TablePlot(Plot):
    def __init__(self, title: str, table: List[Any], columns: List[str], maximum_column_width: int = None, maximum_columns: int = None, minimum_rows: int = None, maximum_rows: int = 50,
                 interpolate: bool = False):
        super().__init__(title)

        self.table = table
        self.columns = columns
        self.maximum_column_width = maximum_column_width
        self.maximum_columns = maximum_columns
        self.minimum_rows = minimum_rows
        self.maximum_rows = maximum_rows
        self.interpolate = interpolate

    def on_plot(self):
        pandas.options.display.max_colwidth = self.maximum_column_width
        pandas.options.display.max_columns = self.maximum_columns
        pandas.options.display.min_rows = self.minimum_rows
        pandas.options.display.max_rows = self.maximum_rows

        display(self.pandas_table)

    @property
    def pandas_table(self):
        table = pandas.DataFrame(self.table, columns=self.columns).infer_objects()

        return table.interpolate(method="linear", limit_direction="both") if self.interpolate else table

    def merge(self, table_plot: "TablePlot"):
        # Add columns from the other table
        added_columns = 0
        for column in table_plot.columns:
            if column not in self.columns:
                self.columns.append(column)
                added_columns += 1

        # Add new empty values for any new columns
        new_table = []
        for row in self.table:
            new_table.append(row + added_columns * [float("NaN")])
        self.table = new_table

        # Add rows from the other table
        for row in table_plot.table:
            new_row = []

            for column in self.columns:
                new_row.append(row[table_plot.columns.index(column)] if column in table_plot.columns else float("NaN"))

            self.table.append(new_row)
