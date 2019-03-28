"""This file contains useful methods for plotting. """

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

class Interactive3dPlot:
    """A class for creating a Plotly 3D interactive plot.

    Examples:
        Supports automatic plot generation by calling:
        >>> data = df[['xCol', 'yCol', 'zCol']]
        >>> p = Interactive3dPlot(data=data, num_clusters = 2)
        >>> p.plot()

        Or plot generation by manually constructing trace layers with
        prefit data:
        >>> p = Interactive3dPlot()
        >>> p.create_trace_layer(df['xCol'], df['yCol'], df['zCol'],
                                                        'green', df.index)
        >>> p.create_trace_layer(df2['xCol'], df2['yCol'], df2['zCol'],
                                                        'blue', df2.index)
        >>> p.plot()
    """

    def __init__(self, data=None, axis_names=None, num_clusters=None):
        """This is the class constructor.

        Args:
        data (pd.DataFrame, optional) - Data with three columns representing
                                        x, y, and z columns, respectively.
        num_clusters (int, optional) - The number of clusters to fit to data.
        """
        self.__data = data
        self.__num_clusters = num_clusters
        self.__layers = []
        self.__colors = ['chocolate', 'mediumblue', 'darkmagenta', 'darkolivegreen',
             'darkred', 'cyan', 'dimgrey', 'salmon', 'gold', 'hotpink', 'ivory',
                 'mintcream', 'orchid', 'sienna', 'silver']

    def get_num_clusters(self):
        """Getter method for number of clusters."""
        return self.__num_clusters

    def get_layers(self):
        """Getter method for plot layers."""
        return self.__layers

    def get_colors(self):
        """Getter method for colors."""
        return self.__colors

    def get_data(self, trace=-1):
        """Getter method for data. Can either return entire set, or a
        specified trace.

        Args:
        trace (int, optional) - The trace to return.

        Returns:
        data (pd.Series or pd.DataFrame) - Requested dataset
        """
        if trace > -1:
            return self.__data[self.__data['cluster'] == trace]
        return self.__data

    def set_colors(self, colors):
        """Setter method for colors.

        Args:
        colors (list of Strings) - Colors to use in plots.
        """
        self.__colors = colors

    def __fit(self):
        """Fits data to sklearn 'KMeans' algorithm and appends column
        to data that identifies which of the clusters the data point
        belongs to.
        """
        kmeans = KMeans(n_clusters = self.get_num_clusters(),
            random_state=121).fit(self.__data)
        self.__data['cluster'] = kmeans.labels_

    def create_trace_layer(self, x, y, z, color, text):
        """Creates a single trace layer. Is either called automatically
        from the plot() method and will use default values
        or can be manually called by the user.

        Args:
        x (pd.Series) - Data to plot along x-axis.
        y (pd.Series) - Data to plot along y-axis.
        z (pd.Series) - Data to plot along z-axis.
        color (String) - Colors to use in plots.
        text (list of Strings) - Text to use for each data point.
        """
        self.__xlabel = x.name
        self.__ylabel = y.name
        self.__zlabel = z.name

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            text=text,
            marker=dict(
                size=5,
                color=color,
                colorscale='Viridis',
                opacity=0.7
            )
        )
        self.__layers.append(trace)

    def __stack_layers(self):
        """Private method called automatically by plot() method and
        creates a __num_clusters trace layers using default colors and text.
        """
        self.__colors = self.__colors[: self.get_num_clusters()]
        for i, color in enumerate(self.__colors):
            temp = self.__data[self.__data['cluster'] == i]
            x = temp[self.__data.columns[0]]
            y = temp[self.__data.columns[1]]
            z = temp[self.__data.columns[2]]
            self.create_trace_layer(x, y, z, color, self.__data.index)

    def __create_layout(self):
        """Creates layout needed for plotting."""
        layout = go.Layout(
            scene = dict(
            xaxis = dict(
                title=self.__xlabel),
            yaxis = dict(
                title=self.__ylabel),
            zaxis = dict(
                title=self.__zlabel)),
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )
        return layout

    def plot(self):
        """Plots interactive 3D plot. If called and __layers is NOT empty,
        this means that the user manually created layers. Therefore, it does
        not fit the data or automatically create layers using __stack_layers.
        This method automatically creates the layout and returns the plot.
        """
        if not self.__layers:
            self.__fit()
            self.__stack_layers()
            layers = self.__layers
        layout = self.__create_layout()
        fig = go.Figure(data=self.__layers, layout=layout)
        return py.iplot(fig, filename='3d-scatter-plot')
