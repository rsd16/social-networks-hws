##############################################################################################################
##############################################################################################################

# SECTION 1. our imports:

import networkx as nx
import numpy as np
import math
import random
from networkit import *
from itertools import combinations, groupby
from networkx.utils import powerlaw_sequence
import matplotlib.pyplot as plt

##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################

# SECTION 2. our constants:

N = [10, 50, 100, 500, 1000, 5000, 10000]
Gammas = [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5] # my code doesn't accept less than or equal to 2.
K = 3
T = 1

##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################

# SECTION 3. the functions that we will use:

def create_random_graph_watts_strogattz(n, k=3):
	p = k / (n - 1)
	G = nx.watts_strogatz_graph(n=n, k=k, p=p)
	return G

def create_random_graph_gnp(n, k=3):
	p = k / (n - 1)
	G = nx.fast_gnp_random_graph(n=n, p=p)
	return G

def create_barabasi_albert_graph(n, k=3):
	G = nx.barabasi_albert_graph(n=n, m=k) # m == <k>
	return G

def average_shortest_path_length_random(G, n, model):
	if model == 'random_graph_watts_strogattz':
		return math.log(n)
	elif model == 'random_graph_gnp':
		return math.log(n)
	elif model == 'barabasi_albert_graph':
		return math.log(n) / math.log(math.log(n))

def average_shortest_path_length_gamma(G, n, gamma):
	if 2 < gamma < 3:
		return math.log(math.log(n)) / math.log(gamma - 1)
	elif gamma == 3:
		return math.log(n) / math.log(math.log(n))
	elif gamma > 3:
		return math.log(n)

def make_graph_connected(G):
	components = dict(enumerate(nx.connected_components(G)))
	components_combs = combinations(components.keys(), r=2)

	for _, node_edges in groupby(components_combs, key=lambda x: x[0]):
		node_edges = list(node_edges)
		random_comps = random.choice(node_edges)
		source = random.choice(list(components[random_comps[0]]))
		target = random.choice(list(components[random_comps[1]]))
		G.add_edge(source, target)

	return G

def connect_isolated_nodes(G):
	list_nodes = list(G.nodes() - list(nx.isolates(G)))

	for isolated_node in list(nx.isolates(G)):
		random_node = random.choice(list_nodes)

		G.add_edge(isolated_node, random_node)

		# add the target node, exclude it from next iteration
		list_nodes.remove(random_node)

	return G

def plot_diagrams(x_axis, dicts, plot_title, ylabel):
	for key in dicts.keys():
		plt.plot(x_axis, dicts[key], label=f'Gamma = {key}')

	plt.title(plot_title)

	plt.xlabel('#Nodes')
	plt.ylabel(ylabel)

	plt.legend(loc='best')

	plt.show()

##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################

# SECTION 4. first of all, let's create random graphs and calculate distance for them:

ws_results_eq = {n: [] for n in N}
ws_results_code = {n: [] for n in N}

gnp_results_eq = {n: [] for n in N}
gnp_results_code = {n: [] for n in N}

ba_results_eq = {n: [] for n in N}
ba_results_code = {n: [] for n in N}

for i in range(T):
	print('\n')
	print('#' * 80)

	print(f'{i = }')

	for n in N:
		print(f'{n = }')

		# Watts-Strogattz:

		G = create_random_graph_watts_strogattz(n, K)

		# to connect isolated nodes to other nodes, so, we wouldn't have any isolated nodes:
		G = connect_isolated_nodes(G)

		G = make_graph_connected(G)

		distance_ws_code = nx.average_shortest_path_length(G, weight=False)
		ws_results_code[n].append(distance_ws_code)

		distance_ws_eq = average_shortest_path_length_random(G, n, 'random_graph_watts_strogattz')
		ws_results_eq[n].append(distance_ws_eq)

		# Erdos-Renyi G(N, p) graph:

		G = create_random_graph_gnp(n, K)

		# to connect isolated nodes to other nodes, so, we wouldn't have any isolated nodes:

		# to connect isolated nodes to other nodes, so, we wouldn't have any isolated nodes:
		G = connect_isolated_nodes(G)

		G = make_graph_connected(G)

		distance_gnp_code = nx.average_shortest_path_length(G, weight=False)
		gnp_results_code[n].append(distance_gnp_code)

		distance_gnp_eq = average_shortest_path_length_random(G, n, 'random_graph_gnp')
		gnp_results_eq[n].append(distance_gnp_eq)

		# Barabassi-Albert graph:

		G = create_barabasi_albert_graph(n, K)

		# to connect isolated nodes to other nodes, so, we wouldn't have any isolated nodes:
		G = connect_isolated_nodes(G)

		G = make_graph_connected(G)

		distance_ba_code = nx.average_shortest_path_length(G, weight=False)
		ba_results_code[n].append(distance_ba_code)

		distance_ba_eq = average_shortest_path_length_random(G, n, 'barabasi_albert_graph')
		ba_results_eq[n].append(distance_ba_eq)

ws_results_code = {key: [np.mean(value), np.std(value)] for key, value in ws_results_code.items()}

print(f'{ws_results_eq = }')
print(f'{ws_results_code = }')

gnp_results_code = {key: [np.mean(value), np.std(value)] for key, value in gnp_results_code.items()}

print(f'{gnp_results_eq = }')
print(f'{gnp_results_code = }')

ba_results_code = {key: [np.mean(value), np.std(value)] for key, value in ba_results_code.items()}

print(f'{ba_results_eq = }')
print(f'{ba_results_code = }')

##############################################################################################################
##############################################################################################################

##############################################################################################################
##############################################################################################################

# SECTION 5. second, let's create some scale-free graphs and calculate distance for them:

##############################################################################################################

# SECTION 5.1. approach 1, using HyperbolicGenerator from Networkit class:

gamma_results_code = {gamma: {n: [] for n in N} for gamma in Gammas}
gamma_results_eq = {gamma: {n: [] for n in N} for gamma in Gammas}

for i in range(T):
	print('\n')
	print('#' * 80)

	print(f'{i = }')

	for gamma in Gammas:
		print(f'{gamma = }')
		print('\n')
		for n in N:
			# generate scale-free graph which uses power law distribution:
			G = generators.HyperbolicGenerator(n=n, k=3, gamma=gamma).generate()

			# convert it to the format of NetworkX package, so we can use useful methods from this package:
			G = nxadapter.nk2nx(G)

			# to connect isolated nodes to other nodes, so, we wouldn't have any isolated nodes:
			G = connect_isolated_nodes(G)

			# in order to be able to use NetworkX method of calculaing average distance, we need to connect components.
			# if we don't connect the components in the graph, then the method would'nt work at all.
			G = make_graph_connected(G)

			distance_gamma_code = nx.average_shortest_path_length(G, weight=False)
			gamma_results_code[gamma][n].append(distance_gamma_code)

			distance_gamma_eq = average_shortest_path_length_gamma(G, n, gamma)
			gamma_results_eq[gamma][n].append(distance_gamma_eq)

		print(f'{gamma_results_eq = }')
		print(f'{gamma_results_code = }')

gamma_results_code = {k1: {k2: [np.mean(v2), np.std(v2)] for k2, v2 in v1.items()} for k1, v1 in gamma_results_code.items()}

print(f'{gamma_results_eq = }')
print(f'{gamma_results_code = }')

##############################################################################################################

##############################################################################################################

# SECTION 5.2. approach 2, generating power-law sequence (incomplete):

for i in range(T):
	print('\n')
	print('#' * 80)

	print(f'{i = }')

	for gamma in Gammas:
		for n in N:
			while True:
				sequence = powerlaw_sequence(n, gamma)

				sequence = [round(num) for num in sequence]

				if sum(sequence) % 2 != 0:
					sequence[max(enumerate(iterable), key=lambda x: x[1])[0]] += 1

				if nx.is_graphical(sequence):
					break

			G = nx.configuration_model(sequence)

			# to connect isolated nodes to other nodes, so, we wouldn't have any isolated nodes:
			G = connect_isolated_nodes(G)

			# in order to be able to use NetworkX method of calculaing average distance, we need to connect components.
			# if we don't connect the components in the graph, then the method would'nt work at all.
			G = make_graph_connected(G)

			degrees = [tup[1] for tup in G.degree()]

			while True:
				average_degree = sum(degrees) / n

				if 2.9 < average_degree < 3.1:
					break

				if 2.9 <= average_degree:
					degrees[max(enumerate(iterable), key=lambda x: x[1])[0]] += 1
				elif average_degree <= 3.1:
					degrees[max(enumerate(iterable), key=lambda x: x[1])[0]] -= 1

			G = nx.configuration_model(degrees)

			distance_gamma_code = nx.average_shortest_path_length(G, weight=False)
			gamma_results_code[gamma][n].append(distance_gamma_code)

			distance_gamma_eq = average_shortest_path_length_gamma(G, n, gamma)
			gamma_results_eq[gamma][n].append(distance_gamma_eq)

gamma_results_code = {k1: {k2: [np.mean(v2), np.std(v2)] for k2, v2 in v1.items()} for k1, v1 in gamma_results_code.items()}

print(gamma_results_eq)
print(gamma_results_code)

##############################################################################################################
##############################################################################################################

##############################################################################################################

##############################################################################################################

# SECTION 5.3. approach 3, generating power-law sequence version 2 (incomplete):

for i in range(T):
	print('\n')
	print('#' * 80)

	print(f'{i = }')

	for gamma in Gammas:
		for n in N:
			while True:
				sequence = powerlaw_sequence(n, gamma)

				sequence = [round(num) for num in sequence]

				if sum(sequence) % 2 != 0:
					sequence[max(enumerate(iterable), key=lambda x: x[1])[0]] += 1

				if not nx.is_graphical(sequence):
					continue

				G = nx.expected_degree_graph(sequence)

				G = connect_isolated_nodes(G)

				G = make_graph_connected(G)

				degrees = [tup[1] for tup in G.degree()]
				average_degree = sum(degrees) / n

				if 2.9 < average_degree < 3.1:
					break

				try:
					if 2.9 <= average_degree:
						degrees[max(enumerate(iterable), key=lambda x: x[1])[0]] += 1
					elif average_degree <= 3.1:
						degrees[max(enumerate(iterable), key=lambda x: x[1])[0]] -= 1
				except NameError:
					continue

			G = nx.configuration_model(degrees)

			# in order to be able to use NetworkX method of calculaing average distance, we need to connect components.
			# if we don't connect the components in the graph, then the method would'nt work at all.
			G = make_graph_connected(G)

			degrees = [tup[1] for tup in G.degree()]

			average_degree = sum(degrees) / n
			print(average_degree)

			distance_gamma_code = nx.average_shortest_path_length(G, weight=False)
			gamma_results_code[gamma][n].append(distance_gamma_code)

			distance_gamma_eq = average_shortest_path_length_gamma(G, n, gamma)
			gamma_results_eq[gamma][n].append(distance_gamma_eq)

gamma_results_code = {k1: {k2: [np.mean(v2), np.std(v2)] for k2, v2 in v1.items()} for k1, v1 in gamma_results_code.items()}

print(gamma_results_eq)
print(gamma_results_code)

##############################################################################################################

##############################################################################################################

##############################################################################################################
##############################################################################################################

# SECTION 6. plot the results and hope for the best:

x_axis = N

values_code = {k1: [value[0] for value in v1.values()] for k1, v1 in gamma_results_code.items()}
print(f'{values_code =}')

print('\n')

values_eq = {k1: [value for sub_list in v1.values() for value in sub_list] for k1, v1 in gamma_results_eq.items()}
print(f'{values_eq =}')

differences = {}

for gamma in Gammas:
	differences[gamma] = [abs(x - y) for x, y in zip(values_code[gamma], values_eq[gamma])]

print(f'{differences = }')

plot_diagrams(x_axis, values_code,
	'Results for Average Distance Produced by Our Codes, for Graphs with Different Gammas',
	'Average Distance')

plot_diagrams(x_axis, differences,
	'Difference between the Results of Our Code and Equations, for Graphs with Different Gammas',
	'Difference')

##############################################################################################################
##############################################################################################################
