from typing import Dict, Any, OrderedDict

from Visualizer.Plotting.AxesPlot import AxesPlot


class ScatterPlot(AxesPlot):
    def __init__(self, title: str, plot_series: OrderedDict[str, Dict[Any, Any]], x_label: str = "x", y_label: str = "y", legend: bool = True, grid: bool = True):
        super().__init__(title=title, plot_series=plot_series, x_label=x_label, y_label=y_label, legend=legend, grid=grid, format_x_as_date=False)

    def on_plot_series(self, figure, axes):
        # Set the data
        for series_name, series_values in self.plot_series.items():
            if len(series_values) > 0:
                axes.scatter(series_values.keys(), series_values.values(), label=series_name)
