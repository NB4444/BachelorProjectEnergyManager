from typing import Dict, Any, OrderedDict

from matplotlib import pyplot

from Visualizer.Plotting.LinePlot import LinePlot
from Visualizer.Plotting.Plot import Plot


class TimeseriesPlot(LinePlot):
    def __init__(self, title: str, plot_series: OrderedDict[str, Dict[Any, Any]], y_label: str = "y", legend: bool = True, grid: bool = True,
                 show_final_values: bool = True, random_colors: bool = False):
        super().__init__(title=title, plot_series=plot_series, x_label="Timestamp", y_label=y_label, legend=legend, grid=grid, format_x_as_date=False, random_colors=random_colors)

        self.show_final_values = show_final_values

    def on_plot_lines(self, figure, axes):
        # Show the final values
        for series_name, series_values in self.plot_series.items():
            if self.show_final_values and len(series_values.values()) > 0:
                final_value = list(series_values.values())[-1]
                self.annotations.append(pyplot.annotate(f"{Plot.humanize_number(final_value)}", xy=(1, final_value), xytext=(8, 0), xycoords=("axes fraction", "data"), textcoords="offset points"))
