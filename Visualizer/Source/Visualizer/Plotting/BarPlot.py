from typing import Dict, Any, OrderedDict

from Visualizer.Plotting.AxesPlot import AxesPlot


class BarPlot(AxesPlot):
    def __init__(self, title: str, plot_series: OrderedDict[str, Dict[Any, Any]], x_label: str = "x",
                 y_label: str = "y", legend: bool = True, grid: bool = True, format_x_as_date=False,
                 random_colors: bool = False):
        super().__init__(title=title, plot_series=plot_series, x_label=x_label, y_label=y_label, legend=legend,
                         grid=grid, format_x_as_date=format_x_as_date, random_colors=random_colors)

    def on_plot_series(self, figure, axes):
        # Set the data
        for series_name, series_values in self.plot_series.items():
            if len(series_values) > 0:
                axes.bar(series_values.keys(), series_values.values(), label=series_name)

        self.on_plot_bars(figure, axes)

    def on_plot_bars(self, figure, axes):
        pass
