from typing import Dict, Any, OrderedDict

from matplotlib import pyplot

from Visualizer.Plotting.AxesPlot import AxesPlot
from Visualizer.Plotting.Plot import Plot


class TimeseriesPlot(AxesPlot):
    def __init__(self, title: str, plot_series: OrderedDict[str, Dict[Any, Any]], y_label: str = "y", legend: bool = True, grid: bool = True,
                 show_final_values: bool = True):
        super().__init__(title=title, plot_series=plot_series, x_label="Timestamp", y_label=y_label, legend=legend, grid=grid, format_x_as_date=False)

        self.show_final_values = show_final_values

    def on_plot_series(self, figure, axes):
        # Set the data
        for series_name, series_values in self.plot_series.items():
            if len(series_values) > 0:
                axes.plot(series_values.keys(), series_values.values(), label=series_name)

                if self.show_final_values and len(series_values.values()) > 0:
                    final_value = list(series_values.values())[-1]
                    self.annotations.append(pyplot.annotate(f"{Plot.humanize_number(final_value)}", xy=(1, final_value), xytext=(8, 0), xycoords=("axes fraction", "data"), textcoords="offset points"))
