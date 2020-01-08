import os
import random
from operator import itemgetter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from scipy import sparse

from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import preprocessing, feature_extraction

import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import link_prediction

PROJECT_DIR = os.path.abspath('')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
ML_10M_DIR = os.path.join(DATA_DIR, 'ml-10m')
ML_1M_DIR = os.path.join(DATA_DIR, 'ml-1m')
GRAPH_REPO_DIR = os.path.join(DATA_DIR, 'graph-repo')

def create_movielens_graph(movielens_dir, max_edges=None):
    rating_chunks = create_ratings_chunk_iterator(movielens_dir, max_edges)
    
    # Networkx doesn't have a custom bipartite graph class, so I need to use an undirected Graph
    graph = nx.Graph()
    
    for chunk in tqdm_notebook(rating_chunks, desc='Creating graph from rating chunks'):
        users = [f'u{user_id}' for user_id in chunk['UserID'].values]
        movies = [f'm{movie_id}' for movie_id in chunk['MovieID'].values]
        
        # Convention in Networkx is to identify sets of nodes with a node attribute named `bipartite`
        # Will be easier to retrieve separate sets of nodes in downstream tasks
        graph.add_nodes_from(users, bipartite='u')
        graph.add_nodes_from(movies, bipartite='m')
        
        ratings = chunk['Rating'].values
        edges = zip(users, movies, ratings)
        graph.add_weighted_edges_from(edges, weight='rating')
        
    del users, movies, edges
    return graph


def create_ratings_chunk_iterator(movielens_dir, max_rows=None, chunksize=1_000_000):
    filename = 'ratings.dat'
    filepath = os.path.join(movielens_dir, filename)
    
    it = pd.read_table(filepath, sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                       usecols=['UserID', 'MovieID', 'Rating'], nrows=max_rows, chunksize=chunksize,
                      engine='python')
    return it


def apply_features(graph, features_df, bipartite_set):
    nodes = [int(n[1:]) for n, d in graph.nodes(data=True) if d['bipartite']==bipartite_set]
    features = features_df.loc[nodes].to_dict('index')
    
    formatted_features = dict()
    for node, attribute_dict in features.items():

        formatted_features[f'{bipartite_set}{node}'] = dict()
        formatted_features[f'{bipartite_set}{node}']['label'] = 'movie' if bipartite_set=='m' else 'user'
        values = np.array(list(attribute_dict.values()))
        formatted_features[f'{bipartite_set}{node}']['feature'] = values

    nx.set_node_attributes(graph, formatted_features)

    return graph


def delete_bipartite_attribute(graph):
    for node in list(graph.nodes):
        del graph.node[node]['bipartite']

    return graph

def apply_movie_attributes(graph, movie_attributes):
    """Applies attributes from movies dataframe to the graph movie nodes"""
    nodes = [int(n[1:]) for n, d in graph.nodes(data=True) if d['bipartite']=='m']
    attributes = movie_attributes.loc[nodes].to_dict('index')
    formatted_attributes = {f'm{k}': v for k, v in attributes.items()}
    nx.set_node_attributes(graph, formatted_attributes)
    
    return graph


def apply_user_attributes(graph, user_attributes):
    """Applies attribues from user dataframe to user nodes in the graph"""
    nodes = [int(n[1:]) for n, d in graph.nodes(data=True) if d['bipartite']=='u']
    attributes = user_attributes.loc[nodes].to_dict('index')
    formatted_attributes = {f'u{k}': v for k, v in attributes.items()}
    nx.set_node_attributes(graph, formatted_attributes)
    
    return graph
    

def load_movies(movielens_dir):
    """
    Description: File contains 10681 movies
    """
    filename = 'movies.dat'
    filepath = os.path.join(movielens_dir, filename)
    index_col = 'MovieID'
    
    df = pd.read_table(filepath, sep='::', names=['MovieID', 'Title', 'Genres'], index_col=index_col, engine='python')
    
    # Transform Genres to a list of strings, since it will be easier to manipulate in downstream tasks
    df['Genres'] = df['Genres'].apply(lambda x : x.split('|'))
    
    return df


def load_movies_formatted(movielens_dir):
    """
    Description: File contains 10681 movies
    """
    filename = 'movies_formatted.dat'
    filepath = os.path.join(movielens_dir, filename)
    index_col = 'MovieID'
    
    df = pd.read_table(filepath, sep=',', 
        names=["MovieID", "Title", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy","Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"], 
        index_col=index_col, engine='python', 
        usecols=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    )
    
    return df


def load_users(movielens_dir):
    """
    Description: File contains 6040 users
    """
    filename = 'users.dat'
    filepath = os.path.join(movielens_dir, filename)
    
    df = pd.read_table(filepath, sep='::', index_col='UserID',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-Code'], engine='python')
    return df


def load_tags(movielens_dir):
    """
    Description: File contains 95580 tags
    """
    filename = 'tags.dat'
    filepath = os.path.join(movielens_dir, filename)
    
    df = pd.read_table(filepath, sep='::', names=['UserID', 'MovieID', 'Tag', 'Timestep'], 
                       usecols=['UserID', 'MovieID', 'Tag'])
    return df


def sample(dist,  num_samples, seed=42):
    random.seed(seed)
    return random.sample(dist, num_samples)


def create_user_features(users_df):
    # Names of user features than require transforming to numeric features:
    feature_names = ["Age", "Gender", "Occupation"]

    feature_encoding = feature_extraction.DictVectorizer(sparse=False, dtype=float, )
    user_features_transformed = feature_encoding.fit_transform(users_df[feature_names].to_dict('records'))

    # Assume that the age can be used as a continuous variable and rescale it
    age_index = 0
    user_features_transformed[:, age_index] = preprocessing.scale(user_features_transformed[:, age_index])

    # One-Hot Encode Occupations
    occupation_index = 3
    occupations = [[occupation] for occupation in user_features_transformed[:, occupation_index]]
    mlb = preprocessing.MultiLabelBinarizer()
    encodings = mlb.fit_transform(occupations)

    user_features_transformed = np.concatenate((
        np.delete(user_features_transformed, occupation_index, axis=1), encodings), axis=1)

    # Put features back in DataFrame
    user_features = pd.DataFrame(
        user_features_transformed, index=users_df.index, dtype="float64"
    )

    return user_features
