import matplotlib.pyplot as plt


def subplot_dict(a):
    # 准备数据
    keys = list(a.keys())
    subkeys = set(k for b in a.values() for k in b.keys())

    data = {subkey: [a[key].get(subkey, 0) for key in keys] for subkey in subkeys}

    # 创建堆积柱状图
    fig, ax = plt.subplots()

    bottom = [0] * len(keys)

    for subkey in subkeys:
        ax.bar(keys, data[subkey], bottom=bottom, label=subkey)
        bottom = [i+j for i,j in zip(bottom, data[subkey])]

    ax.set_ylabel('Counts')
    ax.set_title('Stacked Bar Chart')
    ax.legend()

    plt.show()