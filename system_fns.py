import networkx as nx

def check_window_value(window, span):
    """
    Validates the window parameter against the given 'span'.

    This function ensures that the 'window' size or range is within the
    acceptable limits defined by 'span'.
    The 'window' can be an integer representing the size or a tuple
    representing a range with start and end points.
    
    Parameters:
    - window (int or tuple): The window size as an integer or a range as a
        tuple with start (inclusive) and end (exclusive) indices.
    - span (int): The maximum allowable value for the window size or range end
        point.

    Raises:
    - ValueError: If 'window' is an integer less than 1 or greater than
        'span', or if it's a tuple where the start index is less than 1, the
        end index is greater than 'span', or the end index is less than or
        equal to the start index.
    - TypeError: If 'window' is neither an integer nor a tuple of two
        integers.

    Example:
    - check_window_value(5, 10) # Valid for window size within the span.
    - check_window_value((2, 5), 10) # Valid for start and end range within
        the span.
    """

    if type(window) is int:
        if window < 1 or window > span:
            raise ValueError("Integer window value must be between 1 and "
                             "'span'.")
    elif type(window) is tuple and len(window) == 2:
        if window[0] < 1 or window[1] > span or (window[1] <= window[0]):
            raise ValueError("Tuple window range must start at least at 1 "
                             "and end within 'span', with end > start.")
    else:
        raise TypeError("Window must be an integer or a tuple of two "
                        "integers.")


def convert_to_projection(df, kind='article'):
    """
    Constructs and returns a weighted projection of a bipartite graph from a
    dataframe.

    The bipartite graph is built from a dataframe 'df' that includes an 'id'
    for articles and 'unique_entities' mentioned in them. The projection is
    created for the node set specified by 'kind', which can be 'article' or
    'entity'.
    
    Parameters:
    - df (pandas.DataFrame): Dataframe with 'id' and 'unique_entities'
        columns.
    - kind (str): Node set to project onto ('article' or 'entity'). Default
        is 'article'.
    
    Returns:
    - networkx.Graph: The weighted projected graph.
    """

    B = nx.Graph()
    B.add_nodes_from(df['id'], bipartite='article')
    unique_entities = set(chain.from_iterable(df['unique_entities']))  
    B.add_nodes_from(unique_entities, bipartite='entity')

    for _, row in df.iterrows():
        for entity in row['unique_entities']:
            B.add_edge(row['id'], entity)

    nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == kind}

    return nx.bipartite.weighted_projected_graph(B, nodes)