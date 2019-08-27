import os
import argparse
CLIENT_NUM = 6


def get_port4client(path):
    with open(path) as f:
        for c in f.readlines():
            port = int(c.split()[1])
            print(port)
            os.system(f'iperf3 -s -p {port}&')
    # os.system(f'iperf3 -s -p 1223')


parser = argparse.ArgumentParser()
parser.add_argument('-p',default='./server_info.txt')
args = parser.parse_args()
get_port4client(args.p)
