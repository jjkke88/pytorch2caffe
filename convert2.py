import torch
from torch.autograd import Variable
import torchvision

import os
from pytorch2caffe import pytorch2caffe

import openpose

m = openpose.get_model(44, 19, 2).cuda()
m.eval()
print(m)

input_var = Variable(torch.rand(1, 3, 368, 368)).cuda()
output_var = m(input_var)

output_dir = './'
# plot graph to png
# plot_graph(output_var, os.path.join(output_dir, 'inception_v3.dot'))

pytorch2caffe(input_var, output_var, 
              os.path.join(output_dir, 'pytorch2caffe.prototxt'),
              os.path.join(output_dir, 'pytorch2caffe.caffemodel'))
