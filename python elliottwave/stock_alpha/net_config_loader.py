import os
cfg_dir='cfg'

def load_net_config(net_name):
    net_file_path = os.path.join(cfg_dir, net_name+'.cfg')
    if os.path.exists(net_file_path):
        with open(net_file_path, 'r') as f:
            net_cfg = []
            for line in f.readlines():
                k, v = line.split(' ')
                net_cfg.append((k, int(v)))
        return net_cfg


if __name__ == '__main__':
    print(load_net_config('classify_net'))