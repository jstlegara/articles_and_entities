import numpy as np
import pandas as pd
import networkx as nx
from itertools import chain
from tqdm.notebook import tqdm


class ArticleEntityAnalysis:
    """
    A class for analyzing entities within articles over time.
    """
    
    def __init__(self, df,
                 date_column='date', article_column='id',
                 entity_column='unique_entities'):
        """
        Initializes the ArticleEntityAnalysis with the dataset and column
        specifications.

        Parameters:
        - df (pandas.DataFrame): The dataset containing articles and entities.
        - date_column (str): Column name for article dates.
        - article_column (str): Column name for article identifiers.
        - entity_column (str): Column name for entities within articles.
        """
        self.df = df
        self.date_column = date_column
        self.article_column = article_column
        self.entity_column = entity_column
        
        start_date = df[date_column].min()
        end_date = df[date_column].max()
        span = (end_date - start_date).days + 1
        
        self.start_date = start_date
        self.end_date = end_date
        self.span = span
        
        self.B = self._create_bipartite_network()


    def _create_bipartite_network(self, df=None):
        """
        Creates a bipartite network from the articles and entities.

        Parameters:
        - df (pandas.DataFrame, optional): The dataset to use. Defaults to
            self.df.

        Returns:
        - networkx.Graph: The bipartite graph.
        """
        df = self.df if df is None else df
        
        B = nx.Graph()
        B.add_nodes_from(df[self.article_column],
                         bipartite='article')
        unique_entities = set(chain.from_iterable(
            df[self.entity_column]
        ))  
        B.add_nodes_from(unique_entities, bipartite='entity')

        for _, row in df.iterrows():
            for entity in row[self.entity_column]:
                B.add_edge(row[self.article_column], entity)
        
        return B


    def convert_to_projection(self, kind='article', B=None):
        """
        Projects the bipartite network onto one set of nodes.

        Parameters:
        - kind (str): The type of projection ('article' or 'entity').
        - B (networkx.Graph, optional): The bipartite network. Defaults to
            self.B.

        Returns:
        - networkx.Graph: The projected graph.
        """
        B = self.B if B is None else B
        nodes = {n for n, d in B.nodes(data=True)
                 if d['bipartite'] == kind}

        return nx.bipartite.weighted_projected_graph(B, nodes)


    def _slice_analyze_average(self, window_size, function, kind):
        """
        Analyzes data over a rolling window and computes the average result.

        Parameters:
        - window_size (int): Size of the rolling window in days.
        - function (callable): The function to apply to each projected graph.
        - kind (str): The type of projection to perform.

        Returns:
        - float: The average result from the analysis function.
        """
        width = self.span - window_size + 1
        slide = window_size - 1

        start_slice = self.start_date
        end_slice = start_slice + pd.Timedelta(days=width)

        result = []
        while slide >= 0:
            cut = self.df[(self.df[self.date_column] >= start_slice) &
                          (self.df[self.date_column] < end_slice)]

            bipartite = self._create_bipartite_network(df=cut)
            projection = self.convert_to_projection(B=bipartite, kind=kind)
            result.append(function(projection))

            start_slice += pd.Timedelta(days=1)
            end_slice += pd.Timedelta(days=1)
            slide -= 1

        return np.mean(result)


    def _check_window_value(self, window):
        """
        Validates the window size or range for the analysis.

        Parameters:
        - window (int or tuple): The window size or (start, end) range.

        Raises:
        - ValueError: If window values are outside of valid ranges.
        - TypeError: If window is not an int or tuple of two ints.
        """
        if type(window) is int:
            if window < 1 or window > self.span:
                raise ValueError("Integer window value must be between 1 and "
                                 "'span'.")
        elif type(window) is tuple and len(window) == 2:
            if window[0] < 1 or window[1] > self.span or\
                (window[1] <= window[0]):
                raise ValueError("Tuple window range must start at least at "
                                 "1 and end within 'span', with end > start.")
        else:
            raise TypeError("Window must be an integer or a tuple of two "
                            "integers.")


    def rolling_window_analysis(self, function, window=1, kind='article'):
        """
        Conducts rolling window analysis using a specified function.

        Parameters:
        - function (callable): Function to apply to each window projection.
        - window (int or tuple): Size or range of window sizes to analyze.
        - kind (str): Type of projection for analysis ('article' or 'entity').

        Returns:
        - list or float: Results from analysis, list if window is tuple, else
            float.
        """
        self._check_window_value(window)

        if type(window) is int:

            result = self._slice_analyze_average(window_size=window,
                                                 function=function,
                                                 kind=kind)
        elif type(window) is tuple:

            result = [
                self._slice_analyze_average(window_size=i,
                                            function=function,
                                            kind=kind)
                for i in tqdm(np.arange(window[0], window[1] + 1))
            ]

        return result


##################################
##### Functions for Analysis #####
##################################

def average_weighted_clustering_coefficient(projection):
    """
    Compute the average weighted clustering coefficient of a graph.

    Parameters:
    - projection (nx.Graph): Graph with weights on edges as 'weight'.

    Returns:
    - float: The average weighted clustering across all nodes.

    Assumes the 'weight' attribute exists for each edge.
    """
    clustering_coefficient_weighted = nx.clustering(projection,
                                                    weight='weight')
    average_clustering_coefficient_weighted = sum(
        clustering_coefficient_weighted.values()
    ) / len(projection)

    return average_clustering_coefficient_weighted