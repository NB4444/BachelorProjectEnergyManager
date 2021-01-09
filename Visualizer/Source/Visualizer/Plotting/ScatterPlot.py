from typing import Any, OrderedDict, List

from Visualizer.Plotting.AxesPlot import AxesPlot


class ScatterPlot(AxesPlot):
    def __init__(self, title: str, plot_series: OrderedDict[str, OrderedDict[Any, Any]], x_label: str = "x",
                 y_label: str = "y", z_label: str = "z", legend: bool = True, grid: bool = True,
                 random_colors: bool = False, colors: List[Any] = None, labels: List[Any] = None):
        super().__init__(title=title, plot_series=plot_series, x_label=x_label, y_label=y_label, z_label=z_label,
                         legend=legend, grid=grid, format_x_as_date=False, random_colors=random_colors, labels=labels)

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

                label = f"{series_name} ({len(xs)})"
                if self.is_3d:
                    axes.scatter(xs, ys, zs, c=self.colors, label=label)
                else:
                    axes.scatter(xs, ys, c=self.colors, label=label)
