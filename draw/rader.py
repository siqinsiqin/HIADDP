# -*-coding:utf-8 -*-
"""
# Time       ：2023/6/14 9:44
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import pygal

radar_chart = pygal.Radar()

radar_chart.title = 'Subtlety'  # 图的标题
radar_chart.x_labels = ["Extremely Subtle", "Moderately Subtle", "Fairly Subtle", "Moderately Obvious",
                        "Obvious"]  # 雷达图中的坐标的标签
radar_chart.add('UNet', [68.81, 78.21, 82.95, 83.22, 86.99])  # 添加数据
radar_chart.add('UNet++', [71.08, 79.94, 83.13, 83.27, 86.84])  # 添加数据
radar_chart.add('DBANet', [75.06, 80.25, 84.46, 85, 87.81])  # 添加数据
radar_chart.render()
radar_chart.render_to_file('./wenjian.svg')  # 保存文件
