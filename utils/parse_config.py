

def parse_model_config(path):
    """
    Parse the YOLOv3 layer configuration file and returns module definition
    :param path:
    :return:
    """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith("#")]
    line = [x.rstrip().lstrip() for x in lines]          # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):     # this marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()   # define block names
            if module_defs[-1]['type'] == "convolutional":
                module_defs[-1]['batch_normalize'] = 0
            else:
                key,value = line.split("=")
                module_defs[-1][key.strip()] = value.strip()

    return module_defs


def parse_data_config(path):
    """
    Parse the data configuration file
    :param path:
    :return:
    """
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()

    return options

