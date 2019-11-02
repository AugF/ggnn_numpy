
import pygraphviz as pgv

G = pgv.AGraph()

# 1. add node
input_node_list = ["H(t-1)", "Av", "Wa", "Wz", "Wr", "Wh", "Uz", "Ur", "Uh", "End"]
temp_node_list = [("At(t-1)", "*"), ("Av(t-1)", "-*"),  # -*表示左乘
                  ("Za(t-1)", "*"),("Ra(t-1)", "*"), ("Ha(t-1)", "*"),
                  ("Zh(t-1)", "*"),("Rh(t-1)", "*"),
                  ("Zt(t)", "+"), ("Z(t)", "sigmoid"),("1-Z(t)", "-"),
                  ("Rt(t)", "+"), ("R(t)", "sigmoid"),
                  ("Ht(t)", "X"), ("Hh(t)", "*"), ("Hz(t)", "tanh"),
                  ("H1(t)", "X"), ("H2(t)", "X"), ("H(t)", "+")]

edges_list = [("Wa", "At(t-1)"), ("H(t-1)", "At(t-1)"),
              ("At(t-1)", "Av(t-1)"), ("Av", "Av(t-1)"),
              ("Av(t-1)", "Za(t-1)"), ("Wz", "Za(t-1)"),
              ("Av(t-1)", "Ra(t-1)"), ("Wr", "Ra(t-1)"),
              ("Av(t-1)", "Ha(t-1)"), ("Wh", "Ha(t-1)"),
              ("H(t-1)", "Zh(t-1)"), ("Uz", "Zh(t-1)"),
              ("H(t-1)", "Rh(t-1)"), ("Ur", "Rh(t-1)"),
              ("Za(t-1)", "Zt(t)"), ("Zh(t-1)", "Zt(t)"),
              ("Ra(t-1)", "Rt(t)"), ("Rh(t-1)", "Rt(t)"),
              ("Zt(t)", "Z(t)"), ("Z(t)", "1-Z(t)"), ("Rt(t)", "R(t)"),
              ("H(t-1)", "Ht(t)"), ("R(t)", "Ht(t)"),
              ("Uh", "Hh(t)"), ("Ht(t)", "Hh(t)"), ("Hh(t)", "Hz(t)"),
              ("Hz(t)", "H2(t)"), ("1-Z(t)", "H2(t)"),
              ("H(t-1)", "H1(t)"), ("Z(t)", "H1(t)"),
              ("H1(t)", "H(t)"), ("H2(t)", "H(t)"),
              ("H(t)", "End")]

# set attribute
G.graph_attr["label"] = "GRU hidden layer"

# add input node
for n_i in input_node_list:
    G.add_node(n_i, color="red")

# add temp node
for (n_t, label) in temp_node_list:
    G.add_node(n_t, label=label)

# add edge
for e in edges_list:
    G.add_edge(e[0], e[1], label=G.get_node(e[0])) # 这里直接获取name

# set layout
layouts = ["dot"]

import os
print(os.getcwd())
# save pic
for lay in layouts:
    G.layout(lay)
    G.draw("images/GRU_{}.png".format(lay))