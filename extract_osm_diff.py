

import cv2
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import networkx as nx
import argparse
import sknw

from skimage.morphology import skeletonize
sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'sknw'))
sys.path.append('/home/ananya/Documents/code/sknw')

from metrics import get_connectivity_metrics, get_segment_pr
from find_offset import read_image, dilate_image, erode_image, open_image
from utils import convert_graph_to_node_graph, remove_nodes, convert_node_to_edge_graph

def get_diff_graph(pre_mask, post_mask, output_prefix, intermediate=False):
    # dilate image
    dilated_pre_mask = dilate_image(pre_mask, kernel_size=6, num_iterations=7)
    dilated_post_mask = dilate_image(
        post_mask,  kernel_size=6, num_iterations=7)

    # subtract image
    sub_im = cv2.subtract(dilated_pre_mask, dilated_post_mask)

    # erode subtracted image to remove thin lines
    eroded_sub_im = erode_image(sub_im, kernel_size=5, num_iterations=6)

    # remove final noise from diff mask
    opened_diff_im = open_image(eroded_sub_im, kernel_size=3, num_iterations=2)
    opened_diff_mask = np.zeros(np.shape(opened_diff_im)).astype(np.uint8)
    opened_diff_mask[np.where(opened_diff_im > 0.5)] = 1

    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opened_diff_mask, 4)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    # min_size = 300
    min_size = 1500

    # your answer image
    clean_diff_mask = np.zeros((post_mask.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            clean_diff_mask[labels == i + 1] = 1

    diff_skeleton = skeletonize(clean_diff_mask).astype(np.uint8)
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        diff_skeleton, 4)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    clean_diff_skeleton = np.zeros((post_mask.shape))
    min_size = 50
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            clean_diff_skeleton[labels == i + 1] = 1

    if intermediate:
        dpi = 80
        w, h = post_mask.shape[::-1]
        figsize = w / float(dpi), h / float(dpi)
        fig1 = plt.figure(1, figsize=figsize)
        # plt.subplot(221),
        # ax = fig1.add_axes([0, 0, 1, 1])
        # ax.axis('off')
        # ax.imshow(pre_img, cmap='gray')
        plt.imshow(pre_mask, cmap='gray')
        plt.title('orig image'), plt.xticks([]), plt.yticks([])

        plt.figure(2, figsize=figsize)
        # plt.subplot(222),
        plt.imshow(post_mask, cmap='gray')
        plt.title('tmplate image'), plt.xticks([]), plt.yticks([])

        fig1 = plt.figure(3, figsize=figsize)
        # plt.subplot(221),
        # ax = fig1.add_axes([0, 0, 1, 1])
        # ax.axis('off')
        # ax.imshow(pre_img, cmap='gray')
        plt.imshow(dilated_pre_mask, cmap='gray')
        plt.title('dilated pr image'), plt.xticks([]), plt.yticks([])

        plt.figure(4, figsize=figsize)
        # plt.subplot(222),
        plt.imshow(dilated_post_mask, cmap='gray')
        plt.title('dilated post image'), plt.xticks([]), plt.yticks([])

        plt.figure(5, figsize=figsize)
        # plt.subplot(223),
        plt.imshow(sub_im, cmap='gray')
        plt.title('sub image'), plt.xticks([]), plt.yticks([])

        plt.figure(6, figsize=figsize)
        # plt.subplot(223),
        plt.imshow(eroded_sub_im, cmap='gray')
        plt.title('eroded sub image'), plt.xticks([]), plt.yticks([])

        plt.figure(7, figsize=figsize)
        # plt.subplot(224),
        plt.imshow(opened_diff_im, cmap='gray')
        plt.title('opened image'), plt.xticks([]), plt.yticks([])

        plt.figure(8, figsize=figsize)
        # plt.subplot(224),
        plt.imshow(clean_diff_mask, cmap='gray')
        plt.title('clean image'), plt.xticks([]), plt.yticks([])

        plt.figure(9, figsize=figsize)
        # plt.subplot(224),
        plt.imshow(diff_skeleton, cmap='gray')
        plt.title('diff skeleton'), plt.xticks([]), plt.yticks([])

        plt.figure(10, figsize=figsize)
        # plt.subplot(224),
        plt.imshow(clean_diff_skeleton, cmap='gray')
        plt.title('clean diff skeleton'), plt.xticks([]), plt.yticks([])

        plt.show()
        # cv2.imwrite(output_prefix+'_diff_mask.png', opened_diff_im)

    diff_graph = sknw.build_sknw(clean_diff_skeleton)
    return diff_graph


def main(pre_mask_files, post_mask_files, output_folder_prefix, osm_graph_file, img_file, gt_graph_file):
    # input params
    dpi = 400
    save_plot = True
    show_plot = False
    connectivity_metrics = False

    img = cv2.imread(img_file, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    osm_graph = nx.read_gpickle(osm_graph_file)
    gt_graph = nx.read_gpickle(gt_graph_file)
    osm_node_graph, osm_points = convert_graph_to_node_graph(osm_graph)

    for pre_mask_file, post_mask_file in zip(pre_mask_files, post_mask_files):
        model_log_prefix = pre_mask_file.split('/')[-3]
        model_name = pre_mask_file.split('/')[-4]
        print('Processing {}'.format(model_name))
        output_folder = os.path.join(
            output_folder_prefix, model_name, model_log_prefix)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_prefix = os.path.join(
            output_folder, model_name)
        post_graph_file = output_prefix + '_post_graph.gpickle'
        output_graph_file = output_prefix + '_osm_diff_graph.gpickle'
        output_image_file = output_prefix + '_osm_diff.png'
        output_post_image_file = output_prefix + '_post.png'
        print('pre_file: {} \n post_file: {}\n output graph file: {} \n output_image_file: {} \n'.format(
            pre_mask_file, post_mask_file, output_graph_file, output_image_file))

        pre_mask = np.squeeze(cv2.imread(pre_mask_file)[:, :, 0])
        post_mask = np.squeeze(cv2.imread(post_mask_file)[:, :, 0])

        post_binary_mask = np.zeros((post_mask.shape))
        post_binary_mask[post_mask > 0.5] = 1
        kernel = np.ones((5, 5), np.uint8)
        post_binary_mask = cv2.dilate(post_binary_mask, kernel, iterations=6)
        post_skeleton = skeletonize(post_binary_mask).astype(np.uint8)
        post_graph = sknw.build_sknw(post_skeleton)
        nx.write_gpickle(post_graph, post_graph_file)

        diff_graph = get_diff_graph(
            pre_mask, post_mask, output_prefix, intermediate=show_plot)
        diff_node_graph, diff_points = convert_graph_to_node_graph(diff_graph)

        output_node_graph = remove_nodes(
            osm_node_graph, osm_points, diff_node_graph, diff_points, neighbours=5)

        output_graph = convert_node_to_edge_graph(output_node_graph)
        nx.write_gpickle(output_graph, output_graph_file)

        if save_plot:
            plt_loc = 111
            dpi = 80
            c, w, h = img.shape[::-1]
            figsize = w / float(dpi), h / float(dpi)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            plt.axis('off')
            # plotting edge
            for (s, e) in output_graph.edges():
                ps = np.asarray(output_graph[s][e]['pts'])
                if len(ps) == 0:
                    continue
                plt.subplot(plt_loc), plt.plot(
                    ps[:, 1], ps[:, 0], 'b', linewidth=5)
            # ploting nodes
            # node, nodes = output_graph.node, output_graph.nodes()
            # ps = np.array([node[i]['o'] for i in nodes])
            # plt.subplot(plt_loc), plt.plot(
                # ps[:, 1], ps[:, 0], '.', color='black', alpha=1)
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.savefig(output_image_file, bbox_inches='tight')
            # plt.imshow(img)

            # saving post image
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            plt.axis('off')
            # plotting edge
            for (s, e) in post_graph.edges():
                ps = np.asarray(post_graph[s][e]['pts'])
                if len(ps) == 0:
                    continue
                plt.subplot(plt_loc), plt.plot(
                    ps[:, 1], ps[:, 0], 'b', linewidth=5)
            # ploting nodes
            # node, nodes = post_graph.node, post_graph.nodes()
            # ps = np.array([node[i]['o'] for i in nodes])
            # plt.subplot(plt_loc), plt.plot(
                # ps[:, 1], ps[:, 0], '.', color='black', alpha=1)
            ax.set_ylim(ax.get_ylim()[::-1])
            # plt.imshow(img)
            plt.savefig(output_post_image_file, bbox_inches='tight')

            # plt.show()
            # print('done')

        if connectivity_metrics:
            print('MODEL: {}'.format(model_name))
            osm_diff_connectivity_results, post_connectivity_results = get_connectivity_metrics(
                gt_graph_file, osm_diff_graph_file=output_graph_file, post_graph_file=post_graph_file, num_pairs=10000)
            print('Connectivity correct, tl, ts, noc, OSM DIFF: {}, POST: {}'.format(
                osm_diff_connectivity_results, post_connectivity_results))

            false_neg, true_pos1 = get_segment_pr(gt_graph, output_graph)
            false_pos, true_pos2 = get_segment_pr(output_graph, gt_graph)
            true_pos = (true_pos1 + true_pos2)/2
            pr = true_pos/(true_pos+false_pos)
            re = true_pos/(true_pos+false_neg)
            print('OSM DIFF PR, TP: {}, FP: {}, FN: {}, Precision: {}, Recall: {}'.format(
                true_pos, false_pos, false_pos, pr, re))

            false_neg, true_pos1 = get_segment_pr(gt_graph, post_graph)
            false_pos, true_pos2 = get_segment_pr(post_graph, gt_graph)
            true_pos = (true_pos1 + true_pos2)/2
            pr = true_pos/(true_pos+false_pos)
            re = true_pos/(true_pos+false_neg)
            print('POST PR, TP: {}, FP: {}, FN: {}, Precision: {}, Recall: {}'.format(
                true_pos, false_pos, false_pos, pr, re))


if __name__ == '__main__':

    # create segmentation mask using generate_masks.py
    pre_mask_files = ['/home/ananya/Documents/rds-share/data/digitalglobe/indonesia/opendata.digitalglobe.com/palu-tsunami/pre-event/2018-04-07/103001007B2D7C00/clipped_disaster/output/roads/LinkNet34_pretrained/20190906-112929/103001007B2D7C00/3230213_disaster_comb.png']
    post_mask_files = ['/home/ananya/Documents/rds-share/data/digitalglobe/indonesia/opendata.digitalglobe.com/palu-tsunami/post-event/2018-10-02/1040010042376D00/clipped_disaster/output/roads/LinkNet34_pretrained/20190906-112929/1040010042376D00/3230213_disaster_comb.png']

    # set output folder
    output_folder_prefix = '/home/ananya/Documents/rds-share/data/digitalglobe/indonesia/opendata.digitalglobe.com/palu-tsunami/test_im/roads/'

    # location of osm graph file as gpickle
    osm_graph_file = '/home/ananya/Documents/rds-share/data/digitalglobe/indonesia/opendata.digitalglobe.com/palu-tsunami/test_im/clipped_disaster/mask_3230213-disaster_clip_graph.gpickle'

    # post disaster img file for visualisation
    img_file = '/home/ananya/Documents/rds-share/data/digitalglobe/indonesia/opendata.digitalglobe.com/palu-tsunami/post-event/2018-10-02/1040010042376D00/clipped_disaster/3230213_disaster_comb.tif'

    # gt graph for metrics
    gt_graph_file = '/home/ananya/Documents/rds-share/data/digitalglobe/indonesia/opendata.digitalglobe.com/palu-tsunami/test_im/clipped_disaster/osm_gt/mask_3230213-post_gt_graph.gpickle'

    main(pre_mask_files, post_mask_files, output_folder_prefix,
         osm_graph_file, img_file, gt_graph_file)
