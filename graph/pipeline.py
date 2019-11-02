
import pygraphviz as pgv

G = pgv.AGraph()

G.node_attr['shape'] = 'circle'
G.edge_attr['color'] = 'red'

G.add_nodes_from(['f', 'g', 'h'])
G.add_node('a', color='red', label="input")
G.add_edge('b', 'c', color='blue', label="*")

n = G.get_node('f')
n.attr['shape'] = 'box'
# 填充，字体，颜色，大小

e = G.get_edge('b', 'c')
e.attr['color'] = 'green'
# 粗细， 实现，虚线, 颜色
# doT language:
# dot hello.dot -T png -o hello.png

# https://www.cnblogs.com/liang1100/p/7641984.html
layouts = ['dot', 'neato', 'twopi', 'circo', 'fdp', 'sfdp']
# dot: 明确的方向性
# neato: 缺乏方向性
# twopi: 放射性布局
# circo: 环形布局
# fdp: 缺乏方向性
# sfdp: 渲染大型的图，缺少方向性

import os
if not os.path.exists("images/"):
    os.mkdir("images/")

for lay in layouts:
    G.graph_attr['label'] = "new"
    G.layout(lay)
    G.draw("images/test_{}.png".format(lay))
    break