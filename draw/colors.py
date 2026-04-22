# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import matplotlib.pyplot as plt

colors = {
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkturquoise': '#00CED1',
    'deeppink': '#FF1493',
    'dodgerblue': '#1E90FF',
    'black': '#000000',
    'crimson': '#DC143C',
    'darkgreen': '#006400',
    'darkkhaki': '#BDB76B',
    'gold': '#FFD700',
    'indianred': '#CD5C5C',
    'lawngreen': '#7CFC00',
    'mediumblue': '#0000CD',
    'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371',
    'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'midnightblue': '#191970',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'palegoldenrod': '#EEE8AA',
    'paleturquoise': '#AFEEEE',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#FAA460',
    'seagreen': '#2E8B57',
    'seashell': '#FFF5EE',
    'sienna': '#A0522D',
    'silver': '#C0C0C0',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'snow': '#FFFAFA',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'wheat': '#F5DEB3',
    'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32'
}


def getColors(models=None, length=None, show=False):
    vals = []
    for i, (key, val) in enumerate(colors.items()):
        if i >= length:
            break
        if models is not None:
            print(models[i], val)
        if show:
            plt.scatter(i + 10, i + 10, c=val, marker='o')
        vals.append(val)

    if show:
        plt.show()
        plt.close()
    return vals
