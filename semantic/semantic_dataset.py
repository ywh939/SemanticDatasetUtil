from pathlib import Path
import yaml, os
from collections import defaultdict
import numpy as np

from utils import dict_operation, vis_rgb


class SemanticDataset(object):
    def __init__(self, logger, args) -> None:
        
        self.logger = logger

        self.raw_pcd_path = Path(args.raw_pcd_path)
        self.raw_label_path = Path(args.raw_label_path)
        self.kitti_label_path = Path(args.kitti_label_path)
        self.label_config_path = "config\\semantic_label.yaml"

        with open(self.label_config_path, 'rb') as stream:
            self.semkittiyaml = yaml.safe_load(stream)
        self.learning_map = self.semkittiyaml['learning_map']
        self.rev_learning_map = defaultdict()
        self.color_map = self.semkittiyaml['color_map']
        self.custom_class_map = self.semkittiyaml['custom_class_map']
        self.custom_class_learning_map = self.semkittiyaml['custom_class_learning_map']
        self.str_labels = self.semkittiyaml['labels']
        self.truthLabel = defaultdict()

        self.check_raw_pcd_list = None

    def get_rev_learning_map(self):
        if len(self.rev_learning_map) == 0:
            self.rev_learning_map = dict_operation.create_new_dict_based_on_values(self.learning_map)

        return self.rev_learning_map
    
    def convert_rev_learning_map(self, lrmap):
        return np.vectorize(self.get_rev_learning_map().__getitem__)(lrmap)
    
    def convert_custom_class_to_kitti(self, labeled_pcd):
        return np.vectorize(self.custom_class_learning_map.__getitem__)(labeled_pcd)

    def get_truth_label(self):
        if len(self.truthLabel) == 0:
            file_list = os.listdir(self.kitti_label_path)
            for f in file_list:
                file_path = self.kitti_label_path / f
                if file_path.suffix != '.label':
                    continue
                
                self.truthLabel[f[:-6]] = np.fromfile(file_path, dtype=np.uint32)

        return dict(self.truthLabel)
    
    def vis_class_color(self, vis_class):
        vis_rgb.vis_rgb_color(self.color_map[vis_class])

    def is_in_raw_pcd_folder(self, file_name):
        if self.check_raw_pcd_list is None:
            self.check_raw_pcd_list = set(os.listdir(self.raw_pcd_path))

        return file_name in self.check_raw_pcd_list
        
