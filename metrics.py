import networkx as nx
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import cv2
from utils import generate_pairs, find_path, GraphSuper, create_node_dict, simplify_graph, convert_node_to_edge_graph


class Metric():
    def __init__(self, total_paths):
        self.missed_paths = 0
        self.total_weight = 0
        self.path_len = 0
        self.total_paths = total_paths
        self.correct = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.tl = 0
        self.ts = 0

    def update_weight(self, weight):
        self.total_weight += weight

    def update_len(self, path_len):
        self.path_len += path_len

    def update_fn(self, gt_len):
        if gt_len != -1:
            self.fn += 1
        else:
            self.tn += 1

    def update_correct(self, gt_len, current_len):
        if gt_len == -1:
            self.fp += 1
        else:
            lower_bound = 0.9 * gt_len
            upper_bound = 1.1 * gt_len

            if lower_bound < current_len and current_len < upper_bound:
                self.correct += 1
            elif current_len < lower_bound:
                self.ts += 1
                # print('gt length: {}, current len: {}'.format(gt_len,current_len))
            elif current_len > upper_bound:
                self.tl += 1
                # print('gt length: {}, current len: {}'.format(gt_len,current_len))

    def update_missed_path(self):
        self.missed_paths += 1

    def reduce_total_paths(self):
        self.total_paths -= 1

    def get_all(self):
        all_correct = (self.correct + self.tn) / self.total_paths
        too_long = self.tl / self.total_paths
        too_short = (self.fp + self.ts) / self.total_paths
        no_conn = self.fn / self.total_paths
        # print(all_correct, too_long, too_short, no_conn)
        return (all_correct, too_long, too_short, no_conn)

    def print_all(self):
        found_paths = self.total_paths - self.missed_paths
        av_len = (self.path_len) / found_paths
        av_cost = (self.total_weight) / found_paths
        norm_cost = (self.total_weight) / (self.path_len)

        print(
            'Missed paths: {}, Total Paths: {}, Av Len: {}, Ave Cost: {}, norm cost: {}'
            .format(self.missed_paths, self.total_paths, av_len, av_cost,
                    norm_cost))
        print(
            'True positive : {}, False Positive: {}, False Negative: {}, True Negative: {}, Too Short: {}, Too Long: {}'
            .format(self.correct, self.fp, self.fn, self.tn, self.ts, self.tl))


def get_connectivity_metrics(gt_graph_file,
                             osm_diff_graph_file,
                             post_graph_file,
                             verbose=False,
                             num_pairs=1000):
    s_points, e_points = generate_pairs(n=num_pairs)

    gt_graph = GraphSuper(gt_graph_file)
    post_graph = GraphSuper(post_graph_file)
    osm_diff_graph = GraphSuper(osm_diff_graph_file)

    osm_diff_metric = Metric(num_pairs)
    post_metric = Metric(num_pairs)
    gt_metric = Metric(num_pairs)

    for start_points, end_points in zip(s_points, e_points):

        gt_val = find_path(gt_graph,
                           start_points,
                           end_points,
                           gt_metric,
                           length_key='weight')
        if gt_val == -1:
            # osm_diff_metric.reduce_total_paths()
            # post_metric.reduce_total_paths()
            if verbose:
                print('couldnt find path in gt', start_points, end_points)

            # continue

        osm_val = find_path(osm_diff_graph,
                            start_points,
                            end_points,
                            osm_diff_metric,
                            length_key='weight')
        if osm_val == -1:
            if verbose:
                print('couldnt find path in osm', start_points, end_points)
            osm_diff_metric.update_fn(gt_val)
        else:
            osm_diff_metric.update_correct(gt_val, osm_val)

        post_val = find_path(post_graph,
                             start_points,
                             end_points,
                             post_metric,
                             length_key='weight')
        if post_val == -1:
            if verbose:
                print('couldnt find path in post', start_points, end_points)
            post_metric.update_fn(gt_val)
        else:
            post_metric.update_correct(gt_val, post_val)

    if verbose:
        print('\n osm diff')
        osm_diff_metric.print_all()
        print('\n post')
        post_metric.print_all()
        print('\n gt')
        gt_metric.print_all()

    return osm_diff_metric.get_all(), post_metric.get_all()


def get_segment_pr(gt_graph, output_orig_graph, threshold=25, e=2, seg_len=50):

    gt_graph = simplify_graph(gt_graph, e, seg_len)
    output_graph = simplify_graph(output_orig_graph, e, seg_len)

    # (n)
    gt_node, gt_nodes = gt_graph.node, gt_graph.nodes()
    gt_points = np.array([gt_node[i]['o'] for i in gt_nodes])

    # (m)
    post_node, post_nodes = output_graph.node, output_graph.nodes()
    post_points = np.array([post_node[i]['o'] for i in post_nodes])

    # (nxm)
    distance_mat = distance.cdist(gt_points, post_points)

    # (mx1)
    matching_idxs = np.argmin(distance_mat, axis=0)

    node_dict = create_node_dict(distance_mat, threshold, gt_points,
                                 post_points)

    missing_edge = []
    present_nodes = []
    present_edge = []
    total_len = 0
    missin_len = 0
    common_graph = nx.Graph()
    for edge in gt_graph.edges.data():
        # print(edge)

        def inner():
            nonlocal total_len
            nonlocal missin_len
            left_gt_node_idx = edge[0]
            left_gt_node = gt_node[left_gt_node_idx]['o']
            left_post_nodes = node_dict[tuple(left_gt_node)]
            if left_post_nodes.size == 0:
                missing_edge.append(edge)
                missin_len += edge[2]['weight']
                return
            right_gt_node_idx = edge[1]
            right_gt_node = gt_node[right_gt_node_idx]['o']
            right_post_nodes = node_dict[tuple(right_gt_node_idx)]
            if right_post_nodes.size == 0:
                missing_edge.append(edge)
                missin_len += edge[2]['weight']
                return
            for left_post_node in left_post_nodes:
                for right_post_node in right_post_nodes:
                    if output_graph.has_edge(tuple(left_post_node),
                                             tuple(right_post_node)):
                        l = edge[0]
                        common_graph.add_node(l,
                                              o=left_gt_node,
                                              pts=left_gt_node)
                        r = edge[1]
                        common_graph.add_node(r,
                                              o=right_gt_node,
                                              pts=right_gt_node)
                        common_graph.add_edge(l,
                                              r,
                                              pts=[],
                                              weight=edge[2]['weight'],
                                              length=edge[2]['length'])
                        total_len += edge[2]['weight']
                        present_edge.append(edge, )
                        present_nodes.append(left_gt_node)
                        present_nodes.append(right_gt_node)
                        return
            missing_edge.append(edge)
            missin_len += edge[2]['weight']

        inner()

    false_neg = len(missing_edge)
    true_pos = len(present_edge)
    return false_neg, true_pos
