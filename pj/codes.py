##############################################################################
# our imported packages:
##############################################################################

import networkx as nx
import pandas as pd
from networkx.algorithms import community
from networkx.algorithms.core import core_number, k_core, k_shell
from infomap import Infomap
import random


##############################################################################
# read the *.graphml graph file:
##############################################################################

G = nx.read_graphml('graph.graphml')
print(G)


##############################################################################
# question 3.
##############################################################################

# convert directed graph to undirected one, just for this question:

G_undirected = G.to_undirected()
print(G_undirected)

# perform k-core algorithm on the graph:

G_k_core = k_core(G_undirected)
print(f'K-core results are: {G_k_core}')

G_k_shell = k_shell(G_undirected)
print(f'K-shell results are: {G_k_shell}')

# now, we can compute k-shell values:
k_shell_values = core_number(G_undirected)

# add the computed k-shell values for all of the nodes to the graph:

for node in G.nodes(data=True):
	node[1]['k-shell'] = k_shell_values[node[0]]

# proving that the k-shell values are added:

for _ in range(10):
	print(random.sample(G.nodes(data=True), 1))
	print('\n')

# saving the results as a dataframe for showing them in a nice way:

df = pd.DataFrame(columns=['id', 'label', 'age', 'k-shell'])

counter = 0

for item in G.nodes(data=True):
	df.loc[counter, 'id'] = item[0]
	df.loc[counter, 'label'] = item[1]['label']
	df.loc[counter, 'age'] = item[1]['age']
	df.loc[counter, 'k-shell'] = item[1]['k-shell']
	counter += 1

print(df.head())

print(max(df['k-shell']))

print(df.sort_values('k-shell', ascending=False).head(20))

# saving the modified graph as *.graphml to inspect in gephi:
nx.write_graphml(G, 'question3_resulting_graph.graphml')


##############################################################################
# question 5.
##############################################################################

# infomap:

# first, run the infomap algorithm on the graph:

im = Infomap(silent=True, num_trials=16)

im_graph = im.add_networkx_graph(G)

im.run()

print(f'Found {im.num_top_modules} modules with codelength {im.codelength:.8f} bits.')

# then, we assign the communities to each node:

communities = {}

for node in im.nodes:
	communities[im_graph[node.node_id]] = node.module_id

nx.set_node_attributes(G, communities, 'community')

# proving that the community values are added:

for _ in range(10):
	print(random.sample(G.nodes(data=True), 1))
	print('\n')

# saving the modified graph as *.graphml to inspect in gephi:
nx.write_graphml(G, 'question3_5_resulting_graph.graphml')

# # girvan_newman: (this approach gets stuck in the for loop, does not print communities)

# communities = community.girvan_newman(G)

# print(communities)

# for comm in communities:
# 	print(comm)

# # asyn_fluidc: (this approach gives me the error of wanting connected graph)

# communities = community.asyn_fluidc(G, k=100) # k == number of communities to be found.

# print(communities)

# for comm in communities:
# 	print(comm)

# asyn_lpa_communities:

communities = community.asyn_lpa_communities(G)

print(communities)

for comm in communities:
	print(comm)

# label_propagation_communities: (works with only undirected graphs)

communities = community.label_propagation_communities(G_undirected)

print(communities)

for comm in communities:
    print(comm)
