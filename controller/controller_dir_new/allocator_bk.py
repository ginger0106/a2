from .model_vm_cls import model_vm
from .server_cls import server_cls
import numpy as np
from gurobipy import *
import pickle

DELTA_T = 2
LOCATION_NUM = 3
CLIENT_NUM = 1
DATA_VERSION = 12
#
# IMG_SIZE = np.load('controller_dir/IMG_SIZE.npy')[:12]
# MBV1ACC = np.load('controller_dir/mobile_v1_acc_new.npy')
# MBV1FLP = np.array([1.1375, 1.1375 - 1.1375 * 0.4, 1.1375 - 1.1375 * 0.6, 1.1375 - 1.1375 * 0.8])
#
# RES18ACC = np.load('controller_dir/res18_acc_new.npy')
# RES18FLP = np.array([3.65, 3.65 - 3.65 * 0.4, 3.65 - 3.65 * 0.6, 3.65 - 3.65 * 0.8])




IMG_SIZE = np.load('controller_dir_new/measurements/IMG_SIZE.npy')
MBV1ACC = np.load('controller_dir_new/measurements/mobile_v1_acc_new.npy')
MBV1FLP = np.array([1.1375, 1.1375 - 1.1375 * 0.4, 1.1375 - 1.1375 * 0.6, 1.1375 - 1.1375 * 0.8])

RES18ACC = np.load('controller_dir_new/measurements/res18_acc_new.npy')
RES18FLP = np.array([3.65, 3.65 - 3.65 * 0.4, 3.65 - 3.65 * 0.6, 3.65 - 3.65 * 0.8])


class allocator():
    def __init__(self,version_stg,device_type,time_slot,info_path,dns_path,CLIENT_NUM=1,debug = 'test'):
        self.version_stg = version_stg
        self.device_type = device_type
        self.debug = debug
        self.model_vm_cls =model_vm(device_type,dns_path)
        self.server_dict = server_cls(info_path).get_server_info ()
        self.acc_dict = {'res18':RES18ACC,'mobile':MBV1ACC}
        self.history_list = []
        self.time_slot =time_slot
        self.CLIENT_NUM =CLIENT_NUM
        self.K = self.model_vm_cls.vm_version_dict.keys() #mobile,res18
        self.S = [val['addr'] for val in self.server_dict.values()]#server_id [0,1,3]
        self.I,self.J = self.adaptive_version()
#        print(self.S) 
    def adaptive_version(self):
        I,J =0,0
        if self.version_stg == 'a2':
            I = self.model_vm_cls.vm_version_dict['mobile']
            J = 12
        elif self.version_stg == 'adapative_m':
            I = self.model_vm_cls.vm_version_dict['mobile']
            J = 1
        elif self.version_stg == 'adapative_d':
            I = 1
            J = 12
        else:
            I = self.model_vm_cls.vm_version_dict['mobile']
            J = 12

        return I,J

    def get_available_set(self,client_dict):
        avai_data_ver_set,avai_model_ver_set,avai_server_set =[],[],[]
        for model_ver in range(self.model_vm_cls.vm_version_dict['mobile']):
            for data_ver in range(self.J):
                for h in self.model_vm_cls.h_ki[client_dict['model_name'], model_ver]:
                    for server_addr in [val['addr'] for val in self.server_dict.values()]:
                        est_latency_for_request = len(client_dict['requests'])*IMG_SIZE[data_ver]/client_dict['bw'][server_addr]\
                                                   +self.model_vm_cls.timeout_ki[client_dict['model_name'],model_ver]/1000\
                                                   +self.model_vm_cls.t_kih[client_dict['model_name'],model_ver,h]
                        est_acc_for_request = self.acc_dict[client_dict['model_name']][model_ver][data_ver]
                        if client_dict['acc_limit']<=est_acc_for_request and \
                            client_dict['latency_limit']>=est_latency_for_request :
                            avai_data_ver_set.append(data_ver)
                            avai_model_ver_set.append(model_ver)
                            avai_server_set.append(server_addr)
        return list(set(avai_data_ver_set)),list(set(avai_model_ver_set)),list(set(avai_server_set))

    def get_Qt(self):
        Qt = {}
        for model_name in self.model_vm_cls.vm_version_dict.keys():
            num = 0
            Qkt = {}
            for client_dict in self.history_list:
               # if client_dict == {}:
                #    Qkt[num]={'Y':0,'acc_limit':1,
                 #               'latency_limit':0,
                  #              'avai_data_ver_set':avai_data_ver_set,
                   #             'avai_model_ver_set':avai_model_ver_set,
                    #            'avai_server_set':avai_server_set,
                     #           'writer':client_dict['writer']}
                               
                if client_dict['model_name'] == model_name:
                    avai_data_ver_set, avai_model_ver_set, avai_server_set = self.get_available_set(client_dict)
                    Qkt[num] = {'Y':len(client_dict['requests']),'acc_limit':client_dict['acc_limit'],
                                'latency_limit':client_dict['latency_limit'],
                                'avai_data_ver_set':avai_data_ver_set,
                                'avai_model_ver_set':avai_model_ver_set,
                                'avai_server_set':avai_server_set,
                                'writer':client_dict['writer']
                                }
                    num += 1
            Qt[model_name] = Qkt
            print(Qt)
        return Qt

    def controller_engine(self,history_list):
        self.history_list= history_list
        # print('got msg from all clients')
        allocation_for_all_server_dict, schedule_for_all_client_dict,cost = self.gurobi(self.K,self.I,self.S,self.get_Qt())
        self.process_and_save_data(cost)
        return schedule_for_all_client_dict,allocation_for_all_server_dict


    def process_and_save_data(self,cost):
        avg_latency,avg_bw,avg_acc = [],[],[]
        num=0
        for client_dict in self.history_list:
            client_dict.pop('writer')
            for _,request in client_dict['requests'].items():
                num+=1
                avg_latency.append(request['real_latency'])

                avg_bw.append(request['data_ver']*IMG_SIZE[request['data_ver']])
                avg_acc.append(self.acc_dict[client_dict['model_name']][request['model_ver']][request['data_ver']])

        avg_latency = np.average (avg_latency)
        avg_bw = np.average (avg_bw)
        avg_acc = np.average (avg_acc)
        re = {'avg_latency':avg_latency,'avg_bw':avg_bw,'avg_acc':avg_acc,'history':self.history_list,'cost':cost}
        output = open(f'controller_dir_new/results/result_{self.version_stg}_{self.device_type}_{self.time_slot}.pkl','wb')
        pickle.dump(re,output)


    def gurobi(self,K,I,S,Qt):
        log = '----------------------------------GUROBI----------------------------------'
        print(log)
        m = Model ('PLACEMENT')
        x = self.var_x(m,K,I,S)
        y = self.var_y(m,K,I,S,Qt) 
        print(Qt)
        continuous_x,continuous_y,_ = self.optimation_solver(m,K,I,S,Qt,x,y,0)
        if self.version_stg =='heu':
            integer_x = self.heuristic_rounding(K,I,S,continuous_x)
        else:
            integer_x = self.RA_rounding(continuous_x,continuous_y,K,I,S,Qt)
        _,new_y,cost= self.optimation_solver(m,K,I,S,Qt,integer_x,self.var_y(m,K,I,S,Qt),1)
#        print( integer_x,new_y)
        allocation_for_all_server_dict = self.process_gurobi_result_x(K,I,S,integer_x)
        schedule_for_all_client_dict = self.process_gurobi_result_y(K,I,S,Qt,integer_x,new_y)
        return allocation_for_all_server_dict, schedule_for_all_client_dict,cost

    def optimation_solver(self,m,K,I,S,Qt,x,y,flg):
        # m = Model ('PLACEMENT')
        x_skih =x
        y_skijqh = y

        m.setObjective (
            quicksum (quicksum (quicksum (quicksum(x_skih[s, k, i,h] * self.model_vm_cls.c_ki[k,i]
                                        for h in self.model_vm_cls.h_ki[k, i])
                                for s in S)
                    for k in K)
            for i in range (I))
            , GRB.MINIMIZE)

        for s in S:
            for k in K:
                for i in range(I):
                    for h in self.model_vm_cls.h_ki[k, i]:
                        m.addConstr (quicksum (quicksum ( Qt[k][q]['Y'] * y_skijqh[s, k, i, j, q,h]
                                                         for j in range(self.J))
                                               for q in range (len (Qt[k]))) <=self.time_slot*x_skih[s, k, i,h]*self.model_vm_cls.v_kih[k,i,h])


        for s in S:
            for k in K:
                for i in range(I):
                    for h in self.model_vm_cls.h_ki[k, i]:
                        m.addConstr (quicksum (quicksum ( Qt[k][q]['Y']*IMG_SIZE[j] * y_skijqh[s, k, i, j, q,h]
                                                         for j in range(self.J))
                                               for q in range (len (Qt[k]))) <=x_skih[s, k, i,h]*self.server_dict[s]['BW'])

        for k in K:
            for q in range(len (Qt[k])):
                m.addConstr (quicksum (quicksum (quicksum (quicksum( y_skijqh[s, k, i, j, q,h]
                                                               for h in self.model_vm_cls.h_ki[k, i])
                                                           for i in range(I))
                                                 for j in range(self.J))
                                       for s in S)
                ==1)

        m.optimize ()
        log = '----------------------------------GUROBI----------------------------------'
        print(log)
        # print(o,1111111111)
        c = self.get_cost(K,I,S,x,y)
        if flg==0:
            # print(x)
            y = m.getAttr ('x', y_skijqh)
            x = m.getAttr ('x',x_skih)
            return x,y,c
        else:
            return 0,m.getAttr ('x', y_skijqh),c

    def var_x(self,m,K,I,S):
        x_skih={}
        for k in K:
            for s in S:
                for i in range(I):
                    for h in  self.model_vm_cls.h_ki[k,i]:
                        x_skih[s, k, i,h] = m.addVar (lb=0, vtype=GRB.CONTINUOUS,name="x_%s_%s_%s" % (s, k, i))
        return x_skih

    def var_y(self,m,K,I,S,Qt):
        y_skijqh = {}
        for k in K:
            for s in S:
                for i in range(I):
                    for h in  self.model_vm_cls.h_ki[k,i]:
                        for q in range (len (Qt[k])):
                            for j in range(self.J):
                                y_skijqh[s, k, i, j, q,h] = m.addVar (lb=0,ub=0, vtype=GRB.CONTINUOUS,name="y_%s_%s_%s_%s_%s" % (s,k,i,j,q))
                                m.addVar (lb=0,ub=0, vtype=GRB.CONTINUOUS,name="y_%s_%s_%s_%s_%s" % (s,k,i,j,q))
        for k in K:
            for q in range (len (Qt[k])):
                for s in Qt[k][q]['avai_server_set']:
                    for i in Qt[k][q]['avai_model_ver_set']:
                        for j in Qt[k][q]['avai_data_ver_set']:
                            for h in self.model_vm_cls.h_ki[k, i]:
                                y_skijqh[s, k, i, j, q,h] = m.addVar (lb=0,ub=1, vtype=GRB.CONTINUOUS,name="y_%s_%s_%s_%s_%s" % (s,k,i,j,q))
        return y_skijqh

    def get_cost(self,K,I,S,x,y):
        re = 0
        for s in S:
            for k in K:
                for i in range (I):
                    for h in self.model_vm_cls.h_ki[k, i]:
                        re+= x[s, k, i, h] * self.model_vm_cls.c_ki[k, i]
        return re

    def heuristic_rounding(self,K,I,S,x):
        for s in S:
            for k in K:
                for i in range(I):
                    for h in self.model_vm_cls.h_ki[k, i]:
                        x[s, k, i,h] = int(np.ceil(x[s, k, i,h]))
        return x

    def process_gurobi_result_x(self,K,I,S,x):
        allocation_for_all_server_dict = {}
       # port = 8500
        print(x)
        for s in S:
            item = {}
            port = 8500
            sum_for_server=0
            for k in K:
                for i in range(I):
                    for h in  self.model_vm_cls.h_ki[k,i]:
                        if float(x[s,k,i,h]) >0.0:
                            sum_for_server+=x[s,k,i,h]
                            if sum_for_server > 8: 
                                print('!!!!!!!!!!!!!!!!!! more than 8')
                                exit()
                            item[k+'_dcp_'+str(i)] = {'port':[port+port_num for port_num in range(x[s,k,i,h])],'frac':self.model_vm_cls.frac_ki[k,i],
                                       'batch':h, 'timeout': self.model_vm_cls.timeout_ki[k,i],
                                       'threads': 64, 'device': self.model_vm_cls.hw_ki[k,i]
                                       }
                            port += x[s,k,i,h]
                      #  else:
                           # print(s,k,i,h)  
                           
            allocation_for_all_server_dict[s]= item
                       # port +=1
                        #print(s,allocation_for_all_server_dict)
        return allocation_for_all_server_dict

    def process_gurobi_result_y(self,K,I,S,Qt,x,y):
        schedule_for_all_client_dict = {}
        url_list =[]
        for s in S:
            port = 8500
            for k in K:
               
                for i in range (I):
                    for h in self.model_vm_cls.h_ki[k, i]:
                        url_list = []
                        if (x[s,k,i,h])!= 0:
                            url_list = [self.url_generator (s,port+x,k,i)for x in range(x[s,k,i,h]) ]
                            port += x[s,k,i,h]
                       # else:
                        #    url_list = []
                        for q in range (len (Qt[k])):
                            # print(q,5674657465)
                            # print(s,k,i,h)
                            for j in range (self.J):
                                # print(s, k, i, j, q,h,y[s, k, i, j, q,h])
                                if y[s, k, i, j, q,h] >0:
                                    # print (k, i, j, q,h,y[s, k, i, j, q, h])

                                    result = {f'{k}_dcp_{i}':{'url':url_list,'model_ver':i,'data_ver':j,'batch':h,'prob':y[s, k, i, j, q,h]}}
                                    item = {'writer':Qt[k][q]['writer'],'result':result}
                                    schedule_for_all_client_dict[k,q]=item
        return schedule_for_all_client_dict

    def url_generator(self,server_addr,port,model_name,model_ver):
        worker_addr = self.model_vm_cls.dns_to_worker_addr(server_addr)
        url = f'http://{worker_addr}:{port}/v1/models/{model_name}_dcp_{model_ver}:classify'
        return url

    def set_construct(self,K,S,I,Qt,y):
        z_skijqh={}
        set_all_lst = [] #[{q:[1,2,3]}]
        set_all_copy = []
        set = {} # [s1:[q1,q2]]


        for k in K:
            for s in S:
                for i in range (I):
                    for h in self.model_vm_cls.h_ki[k, i]:
                        for q in range (len (Qt[k])):
                            for j in range (self.J):
                                z_skijqh[s, k, i, j, q, h] = min(Qt[k][q]['Y']*y[s, k, i, j, q, h]/(self.time_slot*self.model_vm_cls.v_kih[k,i,h]),
                                                                 Qt[k][q]['Y'] * y[s, k, i, j, q, h] /self.server_dict[s]['BW'])

        break_all = False
        set_q_dict = {}
        for k in K:
            for q in range (len (Qt[k])):
                set_q_dict = {q:Qt[k][q]['avai_server_set'].copy()}
                for s in S:
                    for i in range (I):
                        for h in self.model_vm_cls.h_ki[k, i]:
                            for j in range (self.J):
                                # print(z_skijqh[s, k, i, j, q, h])
                                if s in set_q_dict[q] and z_skijqh[s, k, i, j, q, h] !=0:
                                     set_all_lst.append(set_q_dict)

        set_all_copy = set_all_lst.copy()
        while len(set_all_copy)!=0:
            set_len_max = 0
            s_max = ''
            q_dict = {}
            for s in S:
                q_lst = []
            #    for set_q in set_all_lst:
             #       for q in set_q.keys():
              #          if s in set_q[q]:
              #  q_lst = []
                for set_q in set_all_lst:
                    for q in set_q.keys():
               #         if s in set_q[q]:
                        if s in set_q[q]:
                            q_lst.append(q)
                            set_len = len(set_q.values())
                            if set_len> set_len_max:
                                set_len_max = set_len
                                s_max = s
                q_dict[s] = q_lst
            for set_q in set_all_lst:
                for q in set_q.keys ():
                    if s_max in set_q[q]:
                # if s_max in set_q.values():
                        set_all_copy.remove(set_q)

            set[s_max] = q_dict[s_max]

        return set,set_all_lst, z_skijqh

    def RA_rounding(self,x,y,K,I,S,Qt):
        xx_skih={}
        x_fractional = {}
        for k in K:
            for s in S:
                for i in range(I):
                    for h in  self.model_vm_cls.h_ki[k,i]:
                        if not float(x[s, k, i,h]).is_integer():
                            x_fractional[s, k, i,h] =  x[s, k, i,h]

        set, set_all_lst, z_skijqh = self.set_construct(K,S,I,Qt,y)

        for k in K:
            for q in range (len (Qt[k])):
                s_hat = ''
                set_q_out = {}
                for s, q_s_lst in set.items():
                    for item in set_all_lst:
                        for qq in item.keys():
                            if qq ==q:
                                if  s in item[qq]:
                                    s_hat = s
                                    set_q_out = item

                # print (q, set_q_out)
                set_q_out[q].remove (s_hat)
                for s in set_q_out[q]:
                    for k in K:
                        for i in range (I):
                            for h in self.model_vm_cls.h_ki[k, i]:
                                for q in range (len (Qt[k])):
                                    for j in range (self.J):
                                        z_hat = z_skijqh[s_hat, k, i, j, q, h]
                                        z = z_skijqh[s, k, i, j, q, h]
                                        if z_hat - np.floor(z)<=np.ceil(z) - z:
                                            z_hat = z_hat+ z- np.floor(z)
                                            z = np.floor(z)
                                        else:
                                            z_hat = z_hat-(np.ceil(z)-z)
                                            z = np.ceil(z)
                                        z_skijqh[s_hat, k, i, j, q, h] =z_hat
                                        z_skijqh[s, k, i, j, q, h] = z


        for s in S:
            for k in K:
                for i in range (I):
                    for h in self.model_vm_cls.h_ki[k, i]:
                        sum_z =0
                        for q in range (len (Qt[k])):
                            for j in range (self.J):
                                sum_z += z_skijqh[s, k, i, j, q, h]
                        if (s, k, i,h) in x_fractional.keys():
                            x[s, k, i,h] = int(np.ceil(sum_z))
                        else:
                            x[s, k, i,h] = int(np.ceil(x[s, k, i,h]))

        return x


                    
                    
            
            
        
        
        
        






