import os
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

from utils import ascii_pcd, open3d_manager, plot_manager, dict_operation, pcd_util, file_manager
from utils import http_manager
from semantic.semantic_dataset import SemanticDataset


class SemanticManager(object):
    def __init__(self, logger, args) -> None:
        
        self.logger = logger
        self.args = args
        self.semanticDataset = SemanticDataset(logger, args)

    def get_vis_classes(self):
        return [self.semanticDataset.custom_class_map[value] for value in self.args.vis_class]

    def convert2kitti(self):
        file_list = os.listdir(self.semanticDataset.raw_label_path)
        self.logger(f'start to convert files at {self.semanticDataset.raw_label_path}')
        
        for f in file_list:
            file_path = self.semanticDataset.raw_label_path / f
            pcd = ascii_pcd.read_ascii_pcd(file_path)

            xyz = np.array(pcd[:, :3], dtype=np.float32)
            label = np.array(pcd[:, 3], dtype=np.uint32).reshape(-1, 1)
            
            xyz.tofile((str(file_path))[:-3] + 'bin') 
            label.tofile((str(file_path))[:-3] + 'label')

            self.logger(f'convert {f}')

    def normalize_pcd_format_from_editor(self):
        editor_url = "http://10.0.119.87:13002"
        file_list_path = "/api/listing"
        raw_pcd_path = "/api/pcdfile"

        file_list_url = http_manager.join_url_list(editor_url, [file_list_path])
        file_list = http_manager.get_http_json_handler(file_list_url)
        if (file_list is None):
            self.logger.error(f'get file list filed, url:{file_list_url}')
            return
        
        select_folder = set(["/tai_zhong/kuangka_pcd"])
        for file_obj in file_list:
            if file_obj['folder'] not in select_folder:
                continue

            if self.semanticDataset.is_in_raw_pcd_folder(file_obj['file']):
                continue
            
            file_url = http_manager.join_url_list(editor_url, [raw_pcd_path, file_obj['folder'], file_obj['file']])
            # pcd = http_manager.get_http_content_robust_handler(file_url, self.logger)
            pcd = http_manager.wget_handler(file_url)
            if pcd is None:
                continue

            save_file_path = self.semanticDataset.raw_pcd_path / file_obj['file']
            with open(save_file_path, 'wb') as f:
                f.write(pcd)

    def visualize_commont_pcd(self):
        self.logger.info(f'start process {self.semanticDataset.raw_pcd_path}')

        for file_path in file_manager.read_file_list(self.semanticDataset.raw_pcd_path):
            pcd1 = pcd_util.load_lidar_data(file_path)
            pcd2 = pcd_util.load_lidar_bin_data(file_path)

            open3d_manager.visualize_commont_point_cloud(pcd1)
            open3d_manager.visualize_commont_point_cloud(pcd2)

    def process_truth_label(self):
        self.logger.info(f'start process {self.semanticDataset.kitti_label_path}')

        file_list = os.listdir(self.semanticDataset.kitti_label_path)
        for f in file_list:
            file_path = self.semanticDataset.kitti_label_path / f
            if file_path.suffix != '.pt':
                continue

            self.logger.info(f'start process {f[:-3]}')

            truth_label = self.semanticDataset.get_truth_label()[f[:-3]]

            self.visualize_truth_labeled_pcd(truth_label, file_path)

    def process_predict_label(self):
        self.logger.info(f'start process {self.semanticDataset.kitti_label_path}')

        file_list = os.listdir(self.semanticDataset.kitti_label_path)

        for f in file_list:
            file_path = self.semanticDataset.kitti_label_path / f
            if file_path.suffix != '.pt':
                continue

            related_file_path = file_path.parent / (f[:-3] + '.bin')
            if (related_file_path.exists() is False):
                continue
            
            self.logger.info(f'start process {f[:-3]}')

            output = torch.load(file_path).cpu()

            predict_label = output.max(1)[1].numpy()
            predict_label = predict_label + 1
            truth_label = self.semanticDataset.get_truth_label()[f[:-3]]
            assert len(predict_label) == len(truth_label)

            self.count_predict_value_for_one_label(predict_label, truth_label)
            self.visualize_labeled_pcd(predict_label, truth_label, file_path)

    def process_predict_label_wo_truth(self):
        self.logger.info(f'start process {self.semanticDataset.kitti_label_path}')

        file_list = os.listdir(self.semanticDataset.kitti_label_path)

        for f in file_list:
            file_path = self.semanticDataset.kitti_label_path / f
            if file_path.suffix != '.pt':
                continue
            
            self.logger.info(f'start process {f[:-3]}')

            output = torch.load(file_path).cpu()

            predict_label = output.max(1)[1].numpy()
            predict_label = predict_label + 1

            self.visualize_labeled_pcd_wo_truth(predict_label, file_path)

    def count_predict_value_for_one_label(self, predict_label, truth_label):
        truth_model_predict = defaultdict(lambda: defaultdict(int))
        truth_kitti_label = defaultdict(lambda: defaultdict(int))
        for key, value in zip(truth_label, predict_label):
            truth_model_predict[key][value] += 1
            truth_kitti_label[key][self.semanticDataset.get_rev_learning_map()[value]] += 1

        truth_model_predict = {k: dict(v) for k, v in truth_model_predict.items()}
        truth_kitti_label = {k: dict(v) for k, v in truth_kitti_label.items()}
        self.logger.info(f'model predict value for one truth label: {truth_model_predict}')
        self.logger.info(f'predict value with kitti format for one truth label: {truth_kitti_label}')
        # self.plot_predict_value(truth_kitti_label)

    def plot_predict_value(self, labels):
        truth_label_mapping = {v: k for k, v in self.semanticDataset.custom_class_map.items()}
        plot_manager.subplot_dict(dict_operation.replace_dict_keys(labels, truth_label_mapping, self.semanticDataset.str_labels))

    def visualize_labeled_pcd(self, predict_label, truth_label, label_file_path):
        if self.args.vis_predict is False:
            return
        
        vis_label = predict_label
        vis_classes = set(self.get_vis_classes())
        ploted_classes = set()
        for i in range(len(truth_label)):
            if truth_label[i] not in vis_classes:
                vis_label[i] = 0
            else:
                if (vis_label[i] in ploted_classes):
                    continue
                # self.semanticDataset.vis_class_color(self.semanticDataset.get_rev_learning_map()[vis_label[i]])
                ploted_classes.add(vis_label[i])
        
        rev_label = self.semanticDataset.convert_rev_learning_map(vis_label)
        if len(self.args.vis_predict_label) != 0:
            for i in range(len(rev_label)):
                if self.semanticDataset.str_labels[rev_label[i]] not in self.args.vis_predict_label:
                    rev_label[i] = 0

        colors = []
        for key in rev_label:
            colors.append(self.semanticDataset.color_map[key])
        colors = np.array(colors)

        pcd = np.fromfile((str(label_file_path))[:-2] + 'bin', dtype=np.float32).reshape(-1, 3)

        open3d_manager.visualize_labeled_pcd(pcd, colors)

    def visualize_labeled_pcd_wo_truth(self, predict_label, label_file_path):
        if self.args.vis_predict is False:
            return
        
        vis_label = predict_label
        
        rev_label = self.semanticDataset.convert_rev_learning_map(vis_label)
        if len(self.args.vis_predict_label) != 0:
            for i in range(len(rev_label)):
                if self.semanticDataset.str_labels[rev_label[i]] not in self.args.vis_predict_label:
                    rev_label[i] = 0

        colors = []
        for key in rev_label:
            colors.append(self.semanticDataset.color_map[key])
        colors = np.array(colors)

        pcd = np.fromfile((str(label_file_path))[:-2] + 'bin', dtype=np.float32).reshape(-1, 3)

        open3d_manager.visualize_labeled_pcd(pcd, colors)

    def visualize_truth_labeled_pcd(self, truth_label, label_file_path):
        if self.args.vis_truth is False:
            return

        convert_label = self.semanticDataset.convert_custom_class_to_kitti(truth_label)
        self.visualize_labeled_pcd_wo_truth(convert_label, label_file_path)