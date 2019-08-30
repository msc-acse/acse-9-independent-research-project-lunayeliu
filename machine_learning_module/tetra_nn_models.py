from collections import OrderedDict

# 2 layers model
class DLCM_tetra_10_2(nn.Module):
    """
    Network for tetra 10 elements
    Input - 10x3
    2 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_tetra_10_2, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(10*3, 50)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(50, 20)),

        ]))

    def forward(self, input):
        output = self.fc(input)
        return output


# 3 layers model
class DLCM_tetra_10_3(nn.Module):
    """
    Network for tetra 10 elements
    Input - 10x3
    3 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_tetra_10_3, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(10*3, 50)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(50, 50)),
            ('relu2', nn.ReLU()),
            ('f3', nn.Linear(50, 20)),

        ]))

    def forward(self, input):
        output = self.fc(input)
        return output

# 4 layers model
class DLCM_tetra_10_4(nn.Module):
    """
    Network for tetra 10 elements
    Input - 10x3
    4 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_tetra_10_4, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(10*3, 50)),
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

# 5 layers model
class DLCM_tetra_10_5(nn.Module):
    """
    Network for tetra 10 elements
    Input - 10x3
    5 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_tetra_10_5, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(10*3, 50)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(50, 50)),
            ('relu2', nn.ReLU()),
            ('f3', nn.Linear(50, 40)),
            ('relu3', nn.ReLU()),
            ('f4', nn.Linear(40, 40)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(40, 20)),

        ]))

    def forward(self, input):
        output = self.fc(input)
        return output

# 6 layers model
class DLCM_tetra_10_6(nn.Module):
    """
    Network for tetra 10 elements
    Input - 10x3
    6 layers
    Output - 20
    """
    def __init__(self):
        super(DLCM_tetra_10_6, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(10*3, 50)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(50, 50)),
            ('relu2', nn.ReLU()),
            ('f3', nn.Linear(50, 40)),
            ('relu3', nn.ReLU()),
            ('f4', nn.Linear(40, 40)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(40, 30)),
            ('relu5', nn.ReLU()),
            ('f6', nn.Linear(30, 20))
        ]))

    def forward(self, input):
        output = self.fc(input)
        return output
