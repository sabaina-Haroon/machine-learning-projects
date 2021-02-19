# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)


UCF101 = {
    'num_classes': 25,         #1 additional for background detection
    #'lr_steps': (80000, 100000, 120000),
    # 'lr_steps': (500, 700, 900),
    # 'lr_steps': (100, 900, 1000, 2000),
    # 'lr_steps': (100, 4200,4500),
    # 'max_iter': 4800,
    'lr_steps': (100, 1800,1975),
    'max_iter': 12000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'UCF101',
}


