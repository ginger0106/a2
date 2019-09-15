import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-b',default=247569,type=float)
parser.add_argument('-p',default='./server_info.txt')
parser.add_argument('-d',default='ens5')

args = parser.parse_args()

def control_bw(bw,path,device):
    del_cmd = f'tcdel {device} --all'
    p = subprocess.Popen(del_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    addr_lst = read_addr(path)
    bw = bw*8
    for addr in addr_lst:
        set_cmd = f'tcset {device} --rate {bw}Kbps --network {addr}'
        p = subprocess.Popen(set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
    print('Done!')

def read_addr(path):
    addr_lst = []
    with open(path) as f:
        for server in f.readlines():
            # print(num)
            server_addr = str(server.split()[0])
            addr_lst.append(server_addr)
    return addr_lst

control_bw(args.b,args.p,args.d)



