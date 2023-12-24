import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import imageio
from datetime import timedelta
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
        if B.size() == 0:
            return None
        else:
            return nx.bipartite.weighted_projected_graph(B, nodes)


    def _slice_analyze_average(self, window_panel, function, kind, intersect=1, mean=True):
        """
        Analyzes data over a rolling window and computes the average result.
        The time series is divided into the number of 'window_panel'. e.g. if
        window_panel = 3, there are 3 equal windows for analysis.

        Parameters:
        - window_panel (int): Number of the rolling windows covering the
            dates.
        - function (callable): The function to apply to each projected graph.
        - kind (str): The type of projection to perform.
        - mean (bool): Whether to return a list of results or the averaged
            result.

        Returns:
        - result (list/float): List or average result from the analysis
            function depending on the parameter 'mean'.
        """
        width = self.span - window_panel + 1
        slide = window_panel - 1

        start_slice = self.start_date
        end_slice = start_slice + pd.Timedelta(days=width)

        result = []
        while slide >= 0:
            cut = self.df[(self.df[self.date_column] >= start_slice) &
                          (self.df[self.date_column] < end_slice)]

            bipartite = self._create_bipartite_network(df=cut)
            projection = self.convert_to_projection(B=bipartite, kind=kind)
            result.append(function(projection))

            start_slice += pd.Timedelta(days=intersect)
            end_slice += pd.Timedelta(days=intersect)
            slide -= 1

        return np.nanmean(result) if mean else result


    def _check_window_panel_value(self, window_panel):
        """
        Validates the window size or range for the analysis.

        Parameters:
        - window_panel (int or tuple): The number of window panels or
            (start, end) range of possible window panels to test.

        Raises:
        - ValueError: If window panel values are outside of valid ranges.
        - TypeError: If window panel is not an int or tuple of two ints.
        """
        if type(window_panel) is int:
            if window_panel < 1 or window_panel > self.span:
                raise ValueError("Integer window panel value must be between "
                                 "1 and 'span'.")
        elif type(window_panel) is tuple and len(window_panel) == 2:
            if window_panel[0] < 1 or window_panel[1] > self.span or\
                (window_panel[1] <= window_panel[0]):
                raise ValueError("Tuple window panel range must start at "
                                 "least at 1 and end within 'span', with end "
                                 "> start.")
        else:
            raise TypeError("Window panels must be an integer or a tuple of "
                            "two integers.")


    def aggregate_rolling_window_analysis(self, function, window_panel=1, intersect=1,
                                          kind='article', mean=True):
        """
        Conducts rolling window analysis using a specified function.

        Parameters:
        - function (callable): Function to apply to each window projection.
        - window_panels (int or tuple): Size or range of window panels to
            perform analysis and comparison.
        - kind (str): Type of projection for analysis ('article' or 'entity').

        Returns:
        - list or float: Results from analysis, list if window is tuple,
            else float.
        """
        self._check_window_panel_value(window_panel)

        if type(window_panel) is int:
            result = self._slice_analyze_average(window_panel=window_panel,
                                                 function=function,
                                                 kind=kind,
                                                 intersect=intersect,
                                                 mean=mean)
        elif type(window_panel) is tuple:
            result = [self._slice_analyze_average(window_panel=i,
                                                  function=function,
                                                  kind=kind,
                                                  intersect=intersect,
                                                  mean=mean)
                      for i in tqdm(np.arange(window_panel[0],
                                              window_panel[1] + 1))]

        return result


    def element_rolling_window_degree_analysis(self, p=5, window_size=7,
                                               kind='entity', plot=False):
        current_date = self.start_date
        
        results = {}
        
        while current_date <= self.end_date:
            cut = self.df[
                (self.df['date'] >= current_date) &
                (self.df['date'] <
                     current_date + pd.Timedelta(days=window_size))
            ]
            bipartite = self._create_bipartite_network(df=cut)
            projection = self.convert_to_projection(B=bipartite, kind=kind)
            
            if projection:
                weighted_degree_dict = dict(projection.degree(weight='weight'))
                for value, degree in weighted_degree_dict.items():
                    if value not in results:
                        results[value] = []
                    results[value].append((pd.Timestamp(current_date), degree))
            else:
                pass

            current_date += timedelta(1)
            
        all_dates = pd.date_range(self.start_date, self.end_date)
        for value in results:
            existing_dates = {date for date, _ in results[value]}
            missing_dates = set(all_dates) - existing_dates
            for missing_date in missing_dates:
                results[value].append((pd.Timestamp(missing_date), 0))
            results[value].sort()
        
        max_degree_elems = sorted(results,
                                  key=lambda x: max([degree for _, degree in
                                                     results[x]]),
                                  reverse=True)
        top_p_elems = max_degree_elems[:p]
        
        if plot:
            # Fetch default matplotlib colors
            default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_dict = {value: color for value, color in
                          zip(top_p_elems, default_colors)}

            filenames = []
            for index, focus_elem in enumerate(top_p_elems):
                plt.figure(figsize=(18, 8))

                line_handles = []
                line_labels = []

                # Plot all entities in background, including the focused
                # element
                for elem in top_p_elems:
                    values = results[elem]
                    dates, degrees = zip(*values)
                    if elem != focus_elem:
                        line, = plt.plot(dates, degrees, '-',
                                         color='lightgray', alpha=0.6)
                    else:
                        line, = plt.plot(dates, degrees, '-',
                                         color=color_dict[elem],
                                         linewidth=2.5)
                    line_handles.append(line)
                    line_labels.append(elem)

                # Replot the focused entity to ensure it's in the foreground
                values = results[focus_elem]
                dates, degrees = zip(*values)
                plt.plot(dates, degrees, '-', linewidth=2.5,
                         color=color_dict[focus_elem])

                plt.ylabel("Weighted Degree")
                plt.xlabel("Date")
                plt.legend(line_handles, line_labels, loc='upper left')
                plt.tight_layout()
                filename = f"{focus_elem}_plot.png"
                plt.savefig(filename, dpi=150)
                filenames.append(filename)
                plt.close()

            if flat_plot:
                plt.figure(figsize=(18, 8))

                for index, focus_elem in enumerate(top_p_elems):
                    values = results[focus_elem]
                    dates, degrees = zip(*values)
                    plt.plot(dates, degrees, '-', label=focus_elem,
                             color=color_dict[focus_elem], linewidth=2.5)
                plt.ylabel("Weighted Degree")
                plt.xlabel("Date")
                plt.legend(loc='upper left')
                plt.title(f"Weighted Degree Progression of Top {p} Elements")
                plt.tight_layout()
                plt.savefig(f"{kind}_weighted_degree_progression.png", dpi=150)
                plt.close()
            else:
                 # Create GIF
                with imageio.get_writer(f'{kind}_progression.gif', mode='I',
                                        duration=1) as writer:
                    for filename in filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)


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
    if projection:
        clustering_coefficient_weighted = nx.clustering(projection,
                                                        weight='weight')
        average_clustering_coefficient_weighted = sum(
            clustering_coefficient_weighted.values()
        ) / len(projection)

        return average_clustering_coefficient_weighted
    else:
        return np.nan
    

def average_centrality_metrics(projection):
    if projection:
        # Degree centrality
        degree_centrality = nx.degree_centrality(projection)
        avg_degree_centrality = sum(degree_centrality.values()) / len(projection)

        # Betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(projection)
        avg_betweenness_centrality = sum(betweenness_centrality.values()) / len(projection)

        # Closeness centrality
        closeness_centrality = nx.closeness_centrality(projection)
        avg_closeness_centrality = sum(closeness_centrality.values()) / len(projection)
        
        return [avg_degree_centrality, avg_betweenness_centrality, avg_closeness_centrality]
    else:
        return [np.nan, np.nan, np.nan, np.nan]