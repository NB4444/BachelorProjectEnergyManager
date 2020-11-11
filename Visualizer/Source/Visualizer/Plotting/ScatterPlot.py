from typing import Dict, Any, OrderedDict, List

import mplcursors

from Visualizer.Plotting.AxesPlot import AxesPlot


class ScatterPlot(AxesPlot):
    def __init__(self, title: str, plot_series: OrderedDict[str, Dict[Any, Any]], x_label: str = "x", y_label: str = "y", z_label: str = "z", legend: bool = True, grid: bool = True,
                 random_colors: bool = False, colors: List[Any] = None):
        super().__init__(title=title, plot_series=plot_series, x_label=x_label, y_label=y_label, z_label=z_label, legend=legend, grid=grid, format_x_as_date=False, random_colors=random_colors)

        self.colors = colors

    def on_plot_series(self, figure, axes):

        # Set the data
        for series_name, series_values in self.plot_series.items():
            if len(series_values) > 0:
                xs = []
                ys = []
                zs = []

                if self.is_3d:
                    for x_value, y_z_values in series_values.items():
                        for y_value, z_value in y_z_values.items():
                            xs.append(x_value)
                            ys.append(y_value)
                            zs.append(z_value)
                else:
                    xs = list(series_values.keys())
                    ys = list(series_values.values())

                # if self.color_mapper is None:
                #     c = None
                # else:
                #     values = [self.color_mapper(series_name, x, y, 0) for x, y in zip(xs, ys)] if len(zs) == 0 else [self.color_mapper(series_name, x, y, z) for x, y, z in zip(xs, ys, zs)]
                #     min_value = min(values)
                #     max_value = max(values)
                #     range = max_value - min_value
                #     color_map = pyplot.get_cmap("gist_rainbow")
                #     c = [color_map((self.color_mapper(series_name, x, y, 0) - min_value) / (range if range > 0 else 1)) for x, y in zip(xs, ys)] if len(zs) == 0 else [
                #         color_map((self.color_mapper(series_name, x, y, z) - min_value) / (range if range > 0 else 1)) for x, y, z in zip(xs, ys, zs)]

                if self.is_3d:
                    points = axes.scatter(xs, ys, zs, c=self.colors, label=series_name)
                else:
                    points = axes.scatter(xs, ys, c=self.colors, label=series_name)

                mplcursors.cursor(points)
