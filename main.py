from pathlib import Path
import os
import argparse
import datetime

from utils import logger_manager
from semantic.semantic_manager import SemanticManager


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--save_log', action='store_true', default=False, help='whether to save log file')
    parser.add_argument('--raw_pcd_path', type=str, default="", help='the path of raw pcd file')
    parser.add_argument('--raw_label_path', type=str, default="", help='the path of raw label file')
    parser.add_argument('--kitti_label_path', type=str, default="", help='the path of kitti label file')
    parser.add_argument('--vis_predict', action='store_true', default=False, help='whether to visualize predict results')
    parser.add_argument('--vis_truth', action='store_true', default=False, help='whether to visualize truth results')
    parser.add_argument('--vis_test', action='store_true', default=False, help='whether to visualize test results')
    parser.add_argument('--vis_class', nargs='*', help='visualize which class', required=True)
    parser.add_argument('--vis_predict_label', nargs='*', default=[], help='visualize which predicted label')

    args = parser.parse_args()
    
    return args
    
def create_logger(args):
    log_file = None
    if (args.save_log):
        log_dir = Path(os.getcwd()) / 'log'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / ('log_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    return logger_manager.create_logger(log_file)


def main():
    args = parse_config()
    logger = create_logger(args)

    semanticManager = SemanticManager(logger, args)
    # semanticManager.normalize_pcd_format_from_editor()
    semanticManager.process_truth_label()
    semanticManager.process_predict_label()
    semanticManager.process_predict_label_wo_truth()

if __name__=='__main__':
    main()