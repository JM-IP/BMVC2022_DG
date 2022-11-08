from models import resnet
from models import resnet_binary
nets_map = {
    'resnet18': resnet.resnet18,
    'resnet18_binary': resnet_binary.birealnet18,
    # TODO:
    # 'mobilenet_binary': mobilenet_bianry.mobilenet
}


def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn
