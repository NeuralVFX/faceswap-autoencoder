import torch
import torch.nn as nn
import matplotlib

matplotlib.use('agg')


############################################################################
# Helper Utilities
############################################################################


def weights_init_normal(m):
    # Set initial state of weights

    classname = m.__class__.__name__
    if hasattr(m, 'no_init'):
        print(f'Skipping Init on Pre-trained:{classname}')
    else:
        if 'ConvTrans' == classname:
            pass
        elif 'Linear' in classname:
            #TODO - CHECK IF ORTHAGONAL IS BETTER
            nn.init.kaiming_normal(m.weight.data)
        elif 'Conv2d' in classname or 'ConvTrans' in classname:
            nn.init.kaiming_normal(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def weights_init_icnr(m):
    # Apply ICNR to our PreShuffConv modules

    classname = m.__class__.__name__
    if 'PreShuffConv' in classname:
        print(m.__class__.__name__)
        m.init_icnr()
        print('new icnr init')


def mft(tensor):
    # Return mean float tensor #
    return torch.mean(torch.FloatTensor(tensor))
