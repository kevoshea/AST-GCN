import pandas as pd
import numpy as np
import os

# Original connections
#carts => carts-db
#catalogue => catalogue-db
#front-end => user, orders, session-db, carts, catalogue
#orders => carts, orders-db, payment, shipping, user
#queue-master => rabbitmq
#shipping => rabbitmq
#user => user-db

num_nodes = 14

# indexing order
node_index = {
    'carts': 0, 'carts_db': 1,
    'catalogue': 2, 'catalogue_db': 3,
    'front_end': 4,
    'orders': 5, 'orders_db': 6,
    'payment': 7,
    'queue_master': 8,
    'rabbitmq': 9,
    'session_db': 10, 'shipping': 11,
    'user': 12, 'user_db': 13
}

def uni():

    adj_matrix_uni = np.zeros((num_nodes, num_nodes), dtype=float)

    # Define the connections
    connections = [
        ('front_end', ['user', 'orders', 'session_db', 'carts', 'catalogue']),
        ('orders', ['carts', 'orders_db', 'payment', 'shipping', 'user']),
        ('catalogue', ['catalogue_db']),
        ('carts', ['carts_db']),
        ('shipping', ['rabbitmq']),
        ('queue_master', ['rabbitmq']),
        ('user', ['user_db'])
    ]

    # Populate the adjacency matrix
    for source, targets in connections:
        for target in targets:
            adj_matrix_uni[node_index[source], node_index[target]] = 1

    return adj_matrix_uni


def uni_loops():

    adj_matrix_uni_loops = np.zeros((num_nodes, num_nodes), dtype=float)

    # Define the connections
    connections = [
        ('front_end', ['user', 'orders', 'session_db', 'carts', 'catalogue']),
        ('orders', ['carts', 'orders_db', 'payment', 'shipping', 'user']),
        ('catalogue', ['catalogue_db']),
        ('carts', ['carts_db']),
        ('shipping', ['rabbitmq']),
        ('queue_master', ['rabbitmq']),
        ('user', ['user_db'])
    ]

    # Populate the adjacency matrix
    for source, targets in connections:
        for target in targets:
            adj_matrix_uni_loops[node_index[source], node_index[target]] = 1
            adj_matrix_uni_loops[node_index[source], node_index[source]] = 1 # self-loops


    return adj_matrix_uni_loops



#Bidirectional connections (no self-loops):

#carts => carts-db, front-end, orders
#carts-db => carts
#catalogue => catalogue-db, front-end
#catalogue-db => catalogue
#front-end => carts, catalogue, orders, session-db, user
#orders => carts, orders-db, payment, shipping, user, front-end
#orders-db => orders
#payment => orders
#queue-master => rabbitmq
#rabbitmq => queue-master, shipping
#session-db => front-end
#shipping => rabbitmq, orders
#user => user-db, orders, front-end
#user-db => user

def bi():
    
    # Define the connections
    connections = [
        ('carts', ['carts_db', 'front_end', 'orders']),
        ('carts_db', ['carts']),
        ('catalogue', ['catalogue_db', 'front_end']),
        ('catalogue_db', ['catalogue']),
        ('front_end', ['carts', 'catalogue', 'orders', 'session_db', 'user']),
        ('orders', ['carts', 'orders_db', 'payment', 'shipping', 'user', 'front_end']),
        ('orders_db', ['orders']),
        ('payment', ['orders']),
        ('queue_master', ['rabbitmq']),
        ('rabbitmq', ['queue_master', 'shipping']),
        ('session_db', ['front_end']),
        ('shipping', ['rabbitmq', 'orders']),
        ('user', ['user_db', 'orders', 'front_end']),
        ('user_db', ['user'])
    ]

    adj_matrix_bi = np.zeros((num_nodes, num_nodes), dtype=float)

    for source, targets in connections:
        for ix,target in enumerate(targets):
            adj_matrix_bi[node_index[source], node_index[target]] = 1

    return adj_matrix_bi

def bi_loops():
    
    # Define the connections
    connections = [
        ('carts', ['carts_db', 'front_end', 'orders']),
        ('carts_db', ['carts']),
        ('catalogue', ['catalogue_db', 'front_end']),
        ('catalogue_db', ['catalogue']),
        ('front_end', ['carts', 'catalogue', 'orders', 'session_db', 'user']),
        ('orders', ['carts', 'orders_db', 'payment', 'shipping', 'user', 'front_end']),
        ('orders_db', ['orders']),
        ('payment', ['orders']),
        ('queue_master', ['rabbitmq']),
        ('rabbitmq', ['queue_master', 'shipping']),
        ('session_db', ['front_end']),
        ('shipping', ['rabbitmq', 'orders']),
        ('user', ['user_db', 'orders', 'front_end']),
        ('user_db', ['user'])
    ]

    adj_matrix_bi_loops = np.zeros((num_nodes, num_nodes), dtype=float)

    for source, targets in connections:
        for ix,target in enumerate(targets):
            adj_matrix_bi_loops[node_index[source], node_index[target]] = 1
            adj_matrix_bi_loops[node_index[source], node_index[source]] = 1 # self-loops

    return adj_matrix_bi_loops

def fully_con():
    
    a = np.ones([14,14])
    return a

def badly_con():
    a = fully_con()
    b = uni()
    c = a-b
    return c

def nodes_fc():
    return np.ones([5,5])