import chainer.computational_graph as c
from chainercv.datasets import voc_bbox_label_names
import numpy as np
from chainer.variable import Variable
import pydotplus
from multi_task.multi_task_300 import Multi_task_300

x_data = np.zeros([1, 3, 300, 300], dtype=np.float32)

x = Variable(x_data)

model = Multi_task_300(n_fg_class=len(voc_bbox_label_names), pretrained_model='imagenet',detection=True,segmentation=True,attention=True)

detection,mask = model(x)
a,b=detection
#loc,conf=detection
g = c.build_computational_graph([a,b,mask],remove_variable=True)

dot_format = g._to_dot()


graph=pydotplus.graph_from_dot_data(dot_format)

graph.write_pdf('visualization.pdf')

with open('visualization.gv', 'w') as o:
    o.write(g.dump())
