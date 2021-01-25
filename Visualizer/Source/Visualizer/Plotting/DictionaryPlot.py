from typing import Dict, Any

from matplotlib import pyplot

from Visualizer.Plotting.FigurePlot import FigurePlot


class DictionaryPlot(FigurePlot):
    def __init__(self, title: str, dictionary: Dict[Any, Any]):
        super().__init__(title=title)

        self.dictionary = dictionary

    def on_plot_figure(self, figure):
        axes = figure.add_subplot(1, 1, 1)

        # Set the labels
        axes.set(title=self.title)

        axes.axis("off")
        axes.invert_yaxis()
        text = ""
        for name, value in sorted(self.dictionary.items()):
            text += "{0:>100}  {1:<50}\n".format(str(name), str(value))
        axes.text(0.5, 0.5, text, verticalalignment="center", horizontalalignment="center", fontfamily="monospace")

        # Make everything fit
        pyplot.tight_layout()
