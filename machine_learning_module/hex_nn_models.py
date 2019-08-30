from collections import OrderedDict
import torch.nn as nn

# 2 Layers model
class DLCM_hex_8_2(nn.Module):
    """
    Network for hex 8 elements
    Input - 8x3
    2 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_hex_8_2, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(8*3, 50)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(50, 20)),
        ]))

    def forward(self, input):
        output = self.fc(input)
        return output

# 3 Layers model
class DLCM_hex_8_3(nn.Module):
    """
    Network for hex 8 elements
    Input - 8x3
    3 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_hex_8_3, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(8*3, 50)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(50, 50)),
            ('relu2', nn.ReLU()),
            ('f3', nn.Linear(50, 20)),
        ]))

    def forward(self, input):
        output = self.fc(input)
        return output


class DLCM_hex_8_4(nn.Module):
    """
    Network for hex 8 elements
    Input - 8x3
    4 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_hex_8_4, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(8*3, 50)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(50, 50)),
            ('relu2', nn.ReLU()),
            ('f3', nn.Linear(50, 40)),
            ('relu3', nn.ReLU()),
            ('f4', nn.Linear(40, 20)),
        ]))

    def forward(self, input):
        output = self.fc(input)
        return output

#5 layers model
class DLCM_hex_8_5(nn.Module):
    """
    Network for hex 8 elements
    Input - 8x3
    5 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_hex_8_5, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(8*3, 50)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(50, 50)),
            ('relu2', nn.ReLU()),
            ('f3', nn.Linear(50, 40)),
            ('relu3', nn.ReLU()),
            ('f4', nn.Linear(40, 30)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(30, 20))
        ]))

    def forward(self, input):
        output = self.fc(input)
        return output

#6 layers model
class DLCM_hex_8_6(nn.Module):
    """
    Network for hex 8 elements
    Input - 8x3
    6 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_hex_8_6, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(8*3, 50)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(50, 50)),
            ('relu2', nn.ReLU()),
            ('f3', nn.Linear(50, 40)),
            ('relu3', nn.ReLU()),
            ('f4', nn.Linear(40, 30)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(30, 25)),
            ('relu5', nn.ReLU()),
            ('f6', nn.Linear(25, 20)),

        ]))

    def forward(self, input):
        output = self.fc(input)
        return output

#7 layers model
class DLCM_hex_8_7(nn.Module):
    """
    Network for hex 8 elements
    Input - 8x3
    7 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_hex_8_7, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(8*3, 50)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(50, 50)),
            ('relu2', nn.ReLU()),
            ('f3', nn.Linear(50, 45)),
            ('relu3', nn.ReLU()),
            ('f4', nn.Linear(45, 40)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(40, 35)),
            ('relu5', nn.ReLU()),
            ('f6', nn.Linear(35, 30)),
            ('relu6', nn.ReLU()),
            ('f6', nn.Linear(35, 20)),
        ]))

    def forward(self, input):
        output = self.fc(input)
        return output
