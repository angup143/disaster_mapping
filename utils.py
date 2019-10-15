
import networkx as nx
import numpy as np
from scipy.spatial import distance
from scipy import interpolate
import matplotlib.pyplot as plt
import cv2
import argparse
from rdp import rdp
from sklearn.neighbors import KDTree

def create_node_dict(distance_mat, threshold, gt_points, post_points):
    node_dict = {}
    matching_dist = distance_mat < threshold
    for idx, post_row in enumerate(matching_dist):
        gt_point = tuple(gt_points[idx])
        post_point = post_points[np.nonzero(post_row)]
        # print(gt_point, idx)
        node_dict[gt_point] = post_point
    return node_dict

def convert_graph_to_node_graph(graph):
    node_graph = nx.OrderedGraph()
    points_list = []
    # count = 0
    for node in graph.nodes():

        for u, v, d in graph.edges([node], data=True):
            def expand_edge(points_list):
                current_node = graph.node[u]
                prev_point = graph.node[u]['o']
                other_point = graph.node[v]['o']
                point = d['pts'][0]
                node_graph.add_node(tuple(prev_point),
                                    o=prev_point, pts=prev_point)
                points_list.append(prev_point)

                # if starting from left node of edge, start with next edge
                this_edge = False
                for current_point in graph.node[u]['pts']:
                    if distance(current_point, point) <= distance(other_point, point):
                        this_edge = True
                if not this_edge:
                    return

                for i, point in enumerate(d['pts']):
                    node_graph.add_node(tuple(point), o=point, pts=point)
                    points_list.append(point)

                    # prev_count = count - 1
                    l = np.linalg.norm(np.asarray(
                        prev_point)-np.asarray(point))

                    node_graph.add_edge(tuple(prev_point),
                                        tuple(point), weight=l, length=l, pts=[])
                    prev_point = point
                    # count = count + 1
                point = graph.node[v]['o']
                node_graph.add_node(tuple(point), o=point, pts=point)
                l = np.linalg.norm(np.asarray(prev_point)-np.asarray(point))

                node_graph.add_edge(tuple(prev_point),
                                    tuple(point), weight=l, length=l, pts=[])
                points_list.append(point)
            expand_edge(points_list)
    return node_graph, points_list

def remove_nodes(gt_node_graph, gt_points, diff_node_graph, diff_points, neighbours=1):
    gt_tree = KDTree(gt_points)
    dist, ind = gt_tree.query(diff_points, k=neighbours, return_distance=True)

    points_to_remove = [tuple(gt_points[i2]) for d, i in zip(
        dist, ind) for d2, i2 in zip(d, i) if d2 < 10]

    output_node_graph = nx.Graph.copy(gt_node_graph)
    output_node_graph.remove_nodes_from(points_to_remove)
    return output_node_graph

def convert_node_to_edge_graph(node_graph):
    # final_graph = nx.Graph()
    new_graph = node_graph.copy()
    for node in list(new_graph.nodes()):
        if new_graph.degree(node) == 2:
            edges = list(new_graph.edges(node))
            node_1 = edges[0][1]
            node_2 = edges[1][1]

            weight_1 = new_graph[node][node_1]['weight']
            weight_2 = new_graph[node][node_2]['weight']
            l_1 = new_graph[node][node_1]['length']
            l_2 = new_graph[node][node_2]['length']

            pts = new_graph[node][node_1]['pts']
            if len(pts) == 0:
                pts = [new_graph.nodes[node]['pts']]
            else:
                pts.extend([new_graph.nodes[node]['pts']])
            pts.extend(new_graph[node][node_2]['pts'])

            weight = weight_1 + weight_2
            length = l_1+l_2

            new_graph.add_node(node_1, o=node_1)
            new_graph.add_node(node_2, o=node_2)
            new_graph.add_edge(node_1, node_2, pts=pts,
                               weight=weight, length=length)
            new_graph.remove_node(node)

        corner_case = False
    for node in list(new_graph.nodes()):
        if new_graph.degree(node) == 0:
            new_graph.remove_node(node)
    return new_graph

class GraphSuper():
    def __init__(self, graph_path):
        self.graph = nx.read_gpickle(graph_path)
        self.all_nodes, self.all_points = self.get_all_points()
        self.node_keys = list(self.all_nodes)

    def get_all_points(self):
        pre_node, pre_nodes = self.graph.node, self.graph.nodes()
        pre_points = np.array([pre_node[i]['o'] for i in pre_nodes])
        return pre_node, pre_points

    def get_closest_point(self, point):
        graph = self.graph
        dist_arr = distance.cdist(point, self.all_points)
        start_index = np.argmin(dist_arr)
        closest_node = self.node_keys[start_index]
        # closest_node = self.all_nodes[start_index]
        return closest_node



# input networkx graph where edges are defined as sets of points
# output simplified nx graph where each edge is a line segment, hence previous edges now consist of multiple edges
def simplify_graph(graph, e, seg_len):
    node_graph = nx.Graph()
    count = 0
    for edge in graph.edges.data():
        # get all segments in edge:
        points_list = [point for point in edge[2]['pts']]
        new_edge = rdp(points_list, e)
        for i, segment in enumerate(new_edge):
            if i == 0:
                pre_segment = segment
                continue
            l = np.linalg.norm(np.asarray(segment)-np.asarray(pre_segment))
            if seg_len > l:
                all_x = np.append(pre_segment[0], segment[0])
                all_y = np.append(pre_segment[1], segment[1])
            else:
                num_parts = np.int(np.ceil(l/seg_len)+1)
                # print(num_parts)
                if pre_segment[0]== segment[0]: # interp wont work since slope is zero
                    xvals = np.full((num_parts), segment[0])
                    yvals = np.linspace(pre_segment[1], segment[1], num_parts)

                else:
                    interp_func = interpolate.interp1d( [pre_segment[0], segment[0]], [
                                    pre_segment[1], segment[1]])

                    xvals = np.linspace(pre_segment[0], segment[0], num_parts)
                    # yvals = np.interp(xvals, [pre_segment[0], segment[0]], [
                                    #   pre_segment[1], segment[1]])
                    yvals = interp_func(xvals)

                all_x = xvals
                all_y = yvals
            for j, (x, y) in enumerate(zip(all_x, all_y)):
                if j == 0:
                    pre_point = tuple((x, y))
                    continue
                point = tuple((x, y))
                el_l = np.linalg.norm(np.asarray(point)-np.asarray(pre_point))
                node_graph.add_node(pre_point, o=pre_point)
                node_graph.add_node(point, o=point)
                node_graph.add_edge(pre_point, point, weight=el_l, length=el_l)
                pre_point = point

            pre_segment = segment

    return node_graph


def generate_pairs(n=4000):
    np.random.seed(seed=1337)
    start_points = np.random.randint(5000, size=(n, 2))
    end_points = np.random.randint(5000, size=(n, 2))
    return start_points, end_points

def find_path(graph, start_point, end_point, metric, length_key):
    start_point = np.expand_dims(start_point, 0)
    end_point = np.expand_dims(end_point, 0)

    start_node = graph.get_closest_point(start_point)
    end_node = graph.get_closest_point(end_point)

    # get path
    try:
        path_nodes = nx.astar_path(
            graph.graph, start_node, end_node, weight='weight')
    except:
        metric.update_missed_path()
        return -1
    

    path_edges = []
    for i, node in enumerate(path_nodes):
        if i == 0:
            continue
        path_edges.append((path_nodes[i-1], path_nodes[i]))
        ps = graph.graph[path_nodes[i-1]][path_nodes[i]]['pts']

    path_weight = 0
    path_len = 0
    for i, (s, e) in enumerate(path_edges):
        metric.update_weight(np.array(graph.graph[s][e]['weight']))
        metric.update_len(np.array(graph.graph[s][e][length_key]))
        path_len += np.array(graph.graph[s][e][length_key])

    return (path_len)

