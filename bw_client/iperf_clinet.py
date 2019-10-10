import iperf3
import asyncio
from collections import deque
import json
import argparse
import json
import os
import time
from shlex import quote

# server_list = [('127.0.0.1',1234),('127.0.0.1',1234)

class IperfClient():
    def __init__(self,path,interval_time=1,time_slot=5*60):
        self.bw = 0
        self.bw_deque = deque(maxlen=int(time_slot/interval_time))
        self.bw_deque_dict = {}
        # self.
        self.interval_time = interval_time
        self.time_slot = time_slot
        self.avg_bw =0
        self.avg_bw_dict = {}
        self.avg_bw_val = {}
        self.server_len =0
        self.time = 0
        self.count =0
        self.path = path
        if os.path.exists('/home/ubuntu/a2/bw_client/bw_time.txt'):
            os.remove('/home/ubuntu/a2/bw_client/bw_time.txt')


    async def run_iperf_all_servers(self):
        with open(self.path) as f:
            task = []
            num = 0
            for server in f.readlines():
                # print(num)
                server_addr = str(server.split()[0])
                server_port = int(server.split()[1])
                bw_deque = deque(maxlen=10)
                self.bw_deque_dict[server_addr]= bw_deque
                task.append(self.client(server_addr,server_port,self.bw_deque_dict[server_addr],num))
                num +=1
            self.server_len = num
        await asyncio.gather(*task)

    async def client(self,addr,port,bw_deque,num):
        while True:
            await asyncio.sleep(self.interval_time)
            await self.iperf3(addr,port)
            bw_deque.append(self.bw)
            self.pop_bw(addr,bw_deque)
            self.count+=1
            self.time =self.interval_time*self.count
            # print(num,self.avg_bw_dict)
            self.avg_bw_dict[num] = f'{addr},{port},{self.avg_bw_val[addr]}'
            await self.write_log()
            # print(self.avg_bw, len(self.bw_deque),self.bw)

    async def iperf3(self,addr,port):
        cmd = f'iperf3 -c {addr} -p {port} -t {3} --json '
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        # print(json.loads(stdout.decode()))
        try:
            self.bw =json.loads(stdout.decode())['end']['sum_received']['bits_per_second']/(8*1e3) ##KBps
        except KeyError:
            await asyncio.sleep(10)

    def pop_bw(self,addr,deque):
        sum_bw = 0
        for i in deque:
            item = {addr:{'bw':i,'time':self.get_time()}}
            self.write_bw_time(item,addr)
            sum_bw += i
        self.avg_bw_val[addr] = sum_bw/len(deque)

    async def write_log(self):
        await asyncio.sleep(self.interval_time)
        # print(self.avg_bw_dict)
        with open('/home/ubuntu/a2/bw_client/bw.txt','w+') as f:
            for server_num in range(self.server_len):
                f.write(self.avg_bw_dict[server_num]+'\n')

    def write_bw_time(self,item,addr):
        # await asyncio.sleep(self.time_slot)

        with open(f'/home/ubuntu/a2/bw_client/bw_time.txt','a+') as f:
            json.dump(item,f)
            # for server_num in range(self.server_len):
            #     # print(self.avg_bw_dict)
            #     f.write(item+'\n')

    def get_time(self):
        # t = time.localtime(time.time())
        # return "%s:%s:%s" % (t.tm_hour, t.tm_min, t.tm_sec)
        return time.time()


parser = argparse.ArgumentParser()
parser.add_argument('-t',default=60*10,type=int)
parser.add_argument('-i',default=1,type=int)
parser.add_argument('-p',default='./server_info.txt')
args = parser.parse_args()


bw_cls = IperfClient(args.p,args.i, args.t)
asyncio.run(bw_cls.run_iperf_all_servers())

