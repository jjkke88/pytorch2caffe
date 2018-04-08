import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable

import torch
import torch.nn as nn

def get_model(out_paf, out_heatmap, out_mask):
    print "out_paf: ", out_paf
    print "out_heatmap: ", out_heatmap
    print "out_mask: ", out_mask
             
    class pose_model(nn.Module):
        def __init__(self, transform_input=False):
            super(pose_model, self).__init__()
      
            def conv_bn(inp, oup, kernel, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, oup, kernel, stride, (kernel-1)/2, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )
            
            def conv_dw(inp, oup, kernel, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, inp, kernel, stride, (kernel-1)/2, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )
            
            def conv_dw_last(inp, oup, kernel, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, inp, kernel, stride, (kernel-1)/2, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                )
            
            def block(output_size):
                return nn.Sequential(
                    conv_dw(128, 128, 3, 1),
                    conv_dw(128, 128, 3, 1),
                    conv_dw(128, 128, 3, 1),
                    conv_dw(128, 512, 1, 1),
                    conv_dw_last(512, output_size, 1, 1),
                )

            def stage(input_size, output_size):
                return nn.Sequential(
                    conv_dw(input_size, 128, 7, 1),
                    conv_dw(128, 128, 7, 1),
                    conv_dw(128, 128, 7, 1),
                    conv_dw(128, 128, 7, 1),
                    conv_dw(128, 128, 7, 1),
                    conv_dw(128, 128, 1, 1),
                    conv_dw_last(128, output_size, 1, 1),
                )
            
            # backbone: mobilenet v1
            self.conv_1 = conv_bn(  3,  32, 3, 2)
            self.conv_2 = conv_dw( 32,  64, 3, 1)
            self.conv_3 = conv_dw( 64, 128, 3, 2)
            self.conv_4 = conv_dw(128, 128, 3, 1)
            self.conv_5 = conv_dw(128, 256, 3, 2)
            self.conv_6 = conv_dw(256, 256, 3, 1)
            self.conv_7 = conv_dw(256, 512, 3, 2)
            self.conv_8 = conv_dw(512, 512, 3, 1)
            self.conv_9 = conv_dw(512, 512, 3, 1)
            self.conv_10 = conv_dw(512, 512, 3, 1)
            self.conv_11 = conv_dw(512, 512, 3, 1)

            self.conv_4_pool = conv_dw(128, 128, 3, 2)
            self.conv_11_up = nn.Upsample(scale_factor=2)
            
            self.smooth1 = conv_dw(896, 512, 3, 1)
            self.smooth2 = conv_dw(512, 256, 3, 1)
            self.smooth3 = conv_dw(256, 128, 3, 1)
            
            input_size = 128 + out_paf + out_heatmap + out_mask
            
            self.model1_1 = block(out_paf)       
            self.model2_1 = stage(input_size, out_paf)
            self.model3_1 = stage(input_size, out_paf)
            self.model4_1 = stage(input_size, out_paf)
            self.model5_1 = stage(input_size, out_paf)
            self.model6_1 = stage(input_size, out_paf)
            
            self.model1_2 = block(out_heatmap)       
            self.model2_2 = stage(input_size, out_heatmap)
            self.model3_2 = stage(input_size, out_heatmap)
            self.model4_2 = stage(input_size, out_heatmap)
            self.model5_2 = stage(input_size, out_heatmap)
            self.model6_2 = stage(input_size, out_heatmap)
            
            self.model1_3 = block(out_mask)       
            self.model2_3 = stage(input_size, out_mask)
            self.model3_3 = stage(input_size, out_mask)
            self.model4_3 = stage(input_size, out_mask)
            self.model5_3 = stage(input_size, out_mask)
            self.model6_3 = stage(input_size, out_mask)
            
            # init the layer
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.xavier_uniform(m.weight.data)
                    # nn.init.constant(m.bias.data, val=0)
                    m.weight.data.normal_(0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()  
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            pass
            
        def forward(self, x): 
            x1 = self.conv_1(x)
            x2 = self.conv_2(x1)
            x3 = self.conv_3(x2)
            x4 = self.conv_4(x3)
            x5 = self.conv_5(x4)
            x6 = self.conv_6(x5)
            x7 = self.conv_7(x6)
            x8 = self.conv_8(x7)
            x9 = self.conv_9(x8)
            x10 = self.conv_10(x9)
            x11 = self.conv_11(x10)
            
            conv_pool = self.conv_4_pool(x4)
            conv_up = self.conv_11_up(x11)
            out1 = torch.cat([conv_pool, x6, conv_up], 1)
            out1 = self.smooth3(self.smooth2(self.smooth1(out1)))
            
            
            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out1_3 = self.model1_3(out1)
            
            out2  = torch.cat([out1_1,out1_2,out1_3,out1], 1)
            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out2_3 = self.model2_3(out2)
            
            out3   = torch.cat([out2_1,out2_2,out2_3,out1], 1)
            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out3_3 = self.model3_3(out3)
            
            out4   = torch.cat([out3_1,out3_2,out3_3,out1], 1)
            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            out4_3 = self.model4_3(out4)
            
            out5   = torch.cat([out4_1,out4_2,out4_3,out1], 1)  
            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            out5_3 = self.model5_3(out5)
            
            out6   = torch.cat([out5_1,out5_2,out5_3,out1], 1)
            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)
            out6_3 = self.model6_3(out6)
            
            return out6_1,out6_2,out6_3

    model = pose_model()     
    return model