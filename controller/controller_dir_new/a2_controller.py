import asyncio
import logging
from .allocator import allocator
from .dict_bytes import dict_bytes
from .server_cls import server_cls

# logging.basicConfig(level= logging.DEBUG)
logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='controller.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

DNS_path_test = 'controller_dir_new/config_real_exp/DNS_list.txt'
server_info_path_test = 'controller_dir_new/config/server_info.txt'

cpu_info_path_aws = 'controller_dir_new/config_real_exp/cpu_info.txt'
gpu_info_path_aws = 'controller_dir_new/config_real_exp/gpu_info.txt'
DNS_path_aws = 'controller_dir_new/config_real_exp/DNS_list.txt'

DNS_list_path_test = 'controller_dir_new/config/DNS_list.txt'
DNS_list_path_aws = 'controller_dir_new/config_real_exp/DNS_list.txt'

gpu_info_path_aws_test = 'controller_dir_new/config_test_aws/cpu_info.txt'
DNS_list_path_aws_test ='controller_dir_new/config_test_aws/DNS_list.txt'

class a2_controller():
    def __init__(self,version_stg,device_type,time_slot,con_addr,con_port,CLIENT_NUM=1,debug = 'test'):
        log = f'----CONTROLLER----: {version_stg,device_type,time_slot,con_addr,con_port,CLIENT_NUM,debug }'
        self.save_log (log)
        self.dict_tool = dict_bytes()
        self.con_addr =con_addr
        self.con_port = con_port
        self.device_type =device_type
        self.history_list = []
        self.server_setup_done = []
        self.debug = debug
        self.allocator = allocator(version_stg,device_type,time_slot,self.get_server_path(),self.get_dns_path(),CLIENT_NUM,debug)
        self.server_dict = server_cls(self.get_server_path()).get_server_info() # {'addr':s[0],'BW':s[1],'R':s[2],'port':s[3]}
        self.CLIENT_NUM =CLIENT_NUM
        self.run = asyncio.run(self.main())

    async def read_msg_from_client(self,reader,writer):
        # while True:
        msg = await self.dict_tool.read_bytes2dict (reader, writer)
        log = 'INFO:[SETP1][SOCKET] GET msg from client!'
        self.save_log(log)
        msg['writer'] = writer
        await self.client_msg_que.put(msg)
        # log = 'INFO:[SETP1][SOCKET] PUT msg to Queue!'
        # self.save_log(log)

    async def send_to_client(self):
        while True:
            send_list = []
            for j in range(len(self.server_dict.keys())):
                print(self.server_dict.keys())
               # log = f'INFO:[SETP4][SOCKET] GET msg {[j]} from Servers!'
               # self.save_log (log)
                await self.server_msg_que.get()
                log = f'INFO:[SETP4][SOCKET] GET msg {[j]} from Server!'
                self.save_log (log)
            log = 'INFO:[SETP4][SOCKET] GET msg from Servers!'
            self.save_log (log)
            schedule_for_all_client_dict = await self.schedule_for_all_client_list_que.get()
            for client in schedule_for_all_client_dict.values():
                schedule_dict = client['result']
                schedule_dict['type'] = 'client_message'
                writer = client['writer']
                send_list.append(self.dict_tool.send_dict2bytes (schedule_dict, writer))
 #           print(schedule_for_all_client_dict) 
            await asyncio.gather(*send_list)
            log = 'INFO:[SETP5][SOCKET] SEND [schedule] to Clients!'
            self.save_log (log)

        # await self.send_to_client()

    async def send_to_server(self):
        while True:
            server_list = []
            for i in range (self.CLIENT_NUM):
                msg = await self.client_msg_que.get ()
                # print(f'from client {i}',msg)
                self.history_list.append (msg)
                log = f'INFO:[SETP2][SOCKET] GET msg {[i]} from Client!'
                self.save_log (log)
            # log = 'INFO:[SETP2][SOCKET] GET msg from Queue!'
            # self.save_log (log)
#            print(self.history_list)
 #           print('\n')
          #  print(self.history_list)
            schedule_for_all_client_dict, allocation2serv_for_all_server = self.allocator.controller_engine (
                self.history_list)
            await self.schedule_for_all_client_list_que.put(schedule_for_all_client_dict)
            log = 'INFO:[SETP2][A2] Allcation done!'
            self.save_log (log)
           # print(allocation2serv_for_all_server)
            for val in self.server_dict.values():
                server_list.append (self.handle_server (val['addr'],val['port'],allocation2serv_for_all_server))
          #  self.history_list = []
            await asyncio.gather (*server_list)
            log = 'INFO:[SETP3][SOCKET] SEND [Allocation] to Servers!'
            self.history_list = []
            self.save_log (log)

    async def handle_server(self,addr,port,allocation2serv_for_all_server):

       # for addr,allocation in allocation2serv_for_all_server.items():
            reader, writer = await asyncio.open_connection (
                addr, port)
  #          print(allocation2serv_for_all_server )
            allocation2serv = allocation2serv_for_all_server[addr]
        #    print(allocation2serv,addr)
        # allocation2serv = {'mobile_dcp_0':{'port':[8501,8502],'frac':0.1,'batch':2,'timeout':10,'threads':16,'device':'gpu'},
        #                    'mobile_dcp_1': {'port': [8503, 8504], 'frac': 0.1, 'batch': 2, 'timeout': 10, 'threads': 16,
        #                                     'device': 'gpu'}
        #                    }
            allocation2serv['type'] = 'allocation'
            print(allocation2serv,addr)
            await self.dict_tool.send_dict2bytes (allocation2serv, writer)
            msg = await self.dict_tool.read_bytes2dict (reader, writer)
          #  log = 'get the msg for server'
#            self.save_log(log)
            await self.server_msg_que.put (msg)

    async def server(self):
        server = await asyncio.start_server(
            self.read_msg_from_client, self.con_addr, self.con_port,limit=2**64)
        addr = server.sockets[0].getsockname()
        print(f'Serving on {addr}')
        async with server:
            await server.serve_forever()

    # async def check_msg4client(self):
    #     while True:
    #         self.history_list.append (await self.client_msg_que.get ())
    #         print(33333333,len(self.history_list))
    #         if len(self.history_list)== self.CLIENT_NUM:
    #             break

    async def main(self):
        self.client_msg_que = asyncio.Queue()
        self.server_msg_que = asyncio.Queue()
        self.schedule_for_all_client_list_que = asyncio.Queue()

        tasks_lst = [self.server(),self.send_to_server(),self.send_to_client()]

        await asyncio.gather(*tasks_lst)

    def get_server_path(self):
        if self.debug =='test':
            return server_info_path_test
        elif self.debug =='aws':
            return gpu_info_path_aws
        # elif self.debug =='aws' and self.device_type =='gpu':
        #     return gpu_info_path_aws
        # elif self.debug =='aws' and self.device_type =='cpu':
        #     return cpu_info_path_aws
        elif self.debug == 'aws_test':
            return gpu_info_path_aws_test

    def get_dns_path(self):
        if self.debug =='test':
            return DNS_list_path_test
        elif self.debug =='aws':
            return DNS_list_path_aws
        elif self.debug =='aws_test':
            return DNS_list_path_aws_test

    def save_log(self,log):
        print(log)
        logging.info(log)


