import matplotlib.pyplot as plt
import numpy as np


def vis_rgb_color(color):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = color
    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
