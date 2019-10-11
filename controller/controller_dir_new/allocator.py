from .model_vm_cls import model_vm
from .server_cls import server_cls
import numpy as np
from gurobipy import *
import pickle
import time
import timeit
from math import isnan


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
        self.x = {}
        self.y = {}
        self.inter = 0
        self.complete_time = -1
        self.inter_start_time = timeit.default_timer()
        self.inter_end_time = timeit.default_timer()

        # self.H = self.adaptive_H(device_type)
    
    def adaptive_version(self):
        I,J =0,0
        if self.version_stg == 'a2':
            I = self.model_vm_cls.vm_version_dict['mobile']
            J = 12
        elif self.version_stg == 'adapative_m':
            # print()
            I = self.model_vm_cls.vm_version_dict['mobile']
            J = 1
        elif self.version_stg == 'adapative_d':
            I = 1
            J = 12
        elif self.version_stg == 'heu':
            I = self.model_vm_cls.vm_version_dict['mobile']
            J = 12
        else:
            I = 1
            J = 1
        # print(222222222222,I,J)
        return I,J

    def adaptive_H(self,device_type,k,i):
        # H = []
        if device_type == 'cpu':
            H = [1]
        else:
            H = self.model_vm_cls.h_ki[k,i]
        return H

    def get_available_set(self,client_dict):
        avai_data_ver_set,avai_model_ver_set,avai_server_set,avai_h_set =[],[],[],[]
        for model_ver in range(self.model_vm_cls.vm_version_dict['mobile']):
            for data_ver in range(self.J):
                for h in self.adaptive_H(self.device_type,client_dict['model_name'],model_ver):
                        # self.model_vm_cls.h_ki[client_dict['model_name'], model_ver]:
                    for server_addr in [val['addr'] for val in self.server_dict.values()]:
                        # print(222,server_addr,[val['addr'] for val in self.server_dict.values()])
                        # len(client_dict['requests']) *
                        est_latency_for_request = IMG_SIZE[data_ver]/client_dict['bw'][server_addr]\
                                                   +self.model_vm_cls.timeout_ki[client_dict['model_name'],model_ver]/1000\
                                                   +self.model_vm_cls.t_kih[client_dict['model_name'],model_ver,h]*1.5 #
                        # print(len(client_dict['requests'])*IMG_SIZE[data_ver])
                        est_acc_for_request = self.acc_dict[client_dict['model_name']][model_ver][data_ver]
                        if client_dict['acc_limit']<=est_acc_for_request and \
                            client_dict['latency_limit']>=est_latency_for_request :
                            avai_data_ver_set.append(data_ver)
                            avai_model_ver_set.append(model_ver)
                            avai_server_set.append(server_addr)
                            avai_h_set.append (h)


        return list(set(avai_data_ver_set)),list(set(avai_model_ver_set)),list(set(avai_server_set)),list(set(avai_h_set))

    def get_Qt(self):
        Qt = {}
        for model_name in self.model_vm_cls.vm_version_dict.keys():
            num = 0
            Qkt = {}
            for client_dict in self.history_list:
                if client_dict['model_name'] == model_name:
                    avai_data_ver_set, avai_model_ver_set, avai_server_set,avai_h_set = self.get_available_set(client_dict)
                    Qkt[num] = {'Y':len(client_dict['requests']),'acc_limit':client_dict['acc_limit'],
                                'latency_limit':client_dict['latency_limit'],
                                'avai_data_ver_set':avai_data_ver_set,
                                'avai_model_ver_set':avai_model_ver_set,
                                'avai_server_set':avai_server_set,
                                'avai_h_set':avai_h_set,
                                'writer':client_dict['writer']
                                }
                    num += 1
            Qt[model_name] = Qkt
        # print(Qt)
        return Qt

    def controller_engine(self,history_list):
        self.history_list= history_list
        # print('got msg from all clients')
        allocation_for_all_server_dict, schedule_for_all_client_dict,cost = self.gurobi(self.K,self.I,self.S,self.get_Qt())
        # print(schedule_for_all_client_dict)
        return schedule_for_all_client_dict,allocation_for_all_server_dict


    def process_and_save_data(self,cost,x,y,cost_opt,cost_heu,overhead):
        avg_latency,avg_bw,avg_acc = [],[],[]
        # complete_time = 0

        num=0
        if self.inter==0:
            self.inter_start_time = timeit.default_timer()
            self.inter+=1

        elif self.inter ==1:

            self.inter_end_time = timeit.default_timer()
            self.complete_time = self.inter_end_time - self.inter_start_time

            for client_dict in self.history_list:
                client_dict.pop('writer')
                for _,request in client_dict['requests'].items():
                    num+=1
                    avg_latency.append(request['real_latency'])

                    avg_bw.append(IMG_SIZE[request['data_ver']])
                    avg_acc.append(self.acc_dict[client_dict['model_name']][request['model_ver']][request['data_ver']])

            avg_latency = np.average (avg_latency)
            avg_bw = np.average (avg_bw)
            avg_acc = np.average (avg_acc)
            xx = {str(key):val for key,val in x.items()}
            yy = {str(key):val for key,val in y.items()}

            re = {'avg_latency':avg_latency,'avg_bw':avg_bw,'avg_acc':avg_acc,'history':self.history_list,'cost':cost,
                  'cost_opt':cost_opt,'cost_heu':cost_heu,'x':xx,'y':yy,'overhead':overhead,'complete':self.complete_time}
            output = open(f'controller_dir_new/results/result_{self.version_stg}_{self.device_type}_{self.time_slot}_{self.inter}.pkl','wb')
            pickle.dump(re,output)
            # self.inter+=1
            exit()


        # else:
            # self.inter ==2:
            # exit()



    def gurobi(self,K,I,S,Qt):
        # last_a2 = 0
        # last_heu = 0
        # last_opt = 0
        # last_x = {}
        # last_y = {}

        log = '----------------------------------GUROBI----------------------------------'
        print(log)
        start_time = timeit.default_timer()
        m = Model ('PLACEMENT')
        x = self.var_x(m,K,I,S)
        y = self.var_y(m,K,I,S,Qt)
        continuous_x,continuous_y,cost_opt = self.optimation_solver(m,K,I,S,Qt,x,y,0)
        if self.version_stg =='heu':
            integer_x = self.heuristic_rounding(K,I,S,continuous_x)
        else:
            integer_x = self.RA_rounding(continuous_x,continuous_y,K,I,S,Qt)

        try:
            _,new_y,cost= self.optimation_solver(m,K,I,S,Qt,integer_x,self.var_y(m,K,I,S,Qt),1)
            endtime = timeit.default_timer()
            cost_heu = self.heu_cost(K,I,S,continuous_x,Qt,m)
            if self.inter ==0:
                self.last_a2 = cost
                self.last_heu = cost_heu
                self.last_opt = cost_opt
                self.last_x = integer_x
                self.last_y = new_y
                self.overhead = endtime-start_time

            self.process_and_save_data(self.last_a2,self.last_x,self.last_y,self.last_opt,self.last_heu,self.overhead)
        except:
            print('Something wrong in gurobi')
            self.process_and_save_data(self.last_a2,self.last_x,self.last_y,self.last_opt,self.last_heu,self.overhead)



        allocation_for_all_server_dict = self.process_gurobi_result_x(K,I,S,integer_x)
        # print(8888888888888,allocation_for_all_server_dict)

        schedule_for_all_client_dict = self.process_gurobi_result_y(K,I,S,Qt,integer_x,new_y)
        # self.x, self.y = integer_x,new_y
        return allocation_for_all_server_dict, schedule_for_all_client_dict,cost

    def heu_cost(self,K,I,S,continuous_x,Qt,m):
        integer_x = self.heuristic_rounding(K, I, S, continuous_x)
        _,new_y,cost_heu= self.optimation_solver(m,K,I,S,Qt,integer_x,self.var_y(m,K,I,S,Qt),1)
        return cost_heu





    def optimation_solver(self,m,K,I,S,Qt,x,y,flg):
        # m = Model ('PLACEMENT')
        # print(m,K,I,S,Qt)
        x_skih =x
        y_skijqh = y

        m.setObjective (
            quicksum (quicksum (quicksum (quicksum(x_skih[s, k, i,h] * self.model_vm_cls.c_ki[k,i]
                                        for h in self.adaptive_H(self.device_type,k,i))
                                for s in S)
                    for k in K)
            for i in range (I))
            , GRB.MINIMIZE)

        for s in S:
            for k in K:
                for i in range(I):
                    for h in self.adaptive_H(self.device_type,k,i):
                        m.addConstr (quicksum (quicksum ( Qt[k][q]['Y'] * y_skijqh[s, k, i, j, q,h]
                                                         for j in range(self.J))
                                               for q in range (len (Qt[k]))) <=h*self.time_slot*x_skih[s, k, i,h]*self.model_vm_cls.v_kih[k,i,h])


        for s in S:
            for k in K:
                for i in  range(I):
                    for h in self.adaptive_H(self.device_type,k,i):
                        m.addConstr (quicksum (quicksum ( Qt[k][q]['Y']*IMG_SIZE[j] * y_skijqh[s, k, i, j, q,h]
                                                         for j in range(self.J))
                                               for q in range (len (Qt[k]))) <=x_skih[s, k, i,h]*self.server_dict[s]['BW'])
        for k in K:
            for q in range(len (Qt[k])):
                m.addConstr (quicksum (quicksum (quicksum (quicksum( y_skijqh[s, k, i, j, q,h]
                                                               for h in self.adaptive_H(self.device_type,k,i))
                                                           for i in range(I))
                                                 for j in range(self.J))
                                       for s in S)== 1)

        m.optimize ()
        log = '----------------------------------GUROBI----------------------------------'
        print(log)
        # print(o,1111111111)
        if flg==0:
            # print(x)
            y = m.getAttr ('x', y_skijqh)
            x = m.getAttr ('x',x_skih)
            c = self.get_cost (K, I, S, x, y)
            return x,y,c
        else:
            c = self.get_cost (K, I, S, x, y)
            return 0,m.getAttr ('x', y_skijqh),c

    def var_x(self,m,K,I,S):
        x_skih={}
        for k in K:
            for s in S:
                for i in range(I):
                    for h in self.adaptive_H(self.device_type,k,i):
                        x_skih[s, k, i,h] = m.addVar (lb=0, vtype=GRB.CONTINUOUS,name="x_%s_%s_%s" % (s, k, i))
        return x_skih

    def var_y(self,m,K,I,S,Qt):
        # no_avai_S,no_avai_I,no_avai_J,no_avai_H, = {},{},{},{}
        # for k in K:
        #     for q in range (len (Qt[k])):
        #         no_avai_S[k,q] =list(set(S).difference(set(Qt[k][q]['avai_server_set'])))
        #         no_avai_I[k,q] =list(set(range(I)).difference(set(Qt[k][q]['avai_model_ver_set'])))
        #         no_avai_J[k,q] =list(set(range(self.J)).difference(set(Qt[k][q]['avai_data_ver_set'])))
        # for k in K:
        #     for q in range (len (Qt[k])):
        #         for i in no_avai_I[k,q]:
        #             no_avai_H[k,q,i] =list(set(self.model_vm_cls.h_ki[k, i]).difference(set(Qt[k][q]['avai_h_set'])))

        y_skijqh = {}
        # for k in K:
        #     for q in range (len (Qt[k])):
        #         for s in no_avai_S[k,q]:
        #             for i in no_avai_I[k,q]:
        #                 for h in no_avai_H[k,q,i]:
        #                         for j in no_avai_J[k,q]:
        #                             # print(34352345)
        #                             y_skijqh[s, k, i, j, q,h] = m.addVar (lb=0,ub=0, vtype=GRB.CONTINUOUS,name="y_%s_%s_%s_%s_%s" % (s,k,i,j,q))
        #                             m.addVar (lb=0,ub=0, vtype=GRB.CONTINUOUS,name="y_%s_%s_%s_%s_%s" % (s,k,i,j,q))

        for k in K:
            for q in range (len (Qt[k])):
                for s in S:
                    for i in range(I):
                        for h in self.adaptive_H(self.device_type,k,i):
                                for j in range(self.J):
                                    # print(34352345)
                                    y_skijqh[s, k, i, j, q,h] = m.addVar (lb=0,ub=0, vtype=GRB.CONTINUOUS,name="y_%s_%s_%s_%s_%s" % (s,k,i,j,q))
                                    m.addVar (lb=0,ub=0, vtype=GRB.CONTINUOUS,name="y_%s_%s_%s_%s_%s" % (s,k,i,j,q))
        for k in K:
            for q in range (len (Qt[k])):
                for s in Qt[k][q]['avai_server_set']:

                    for i in Qt[k][q]['avai_model_ver_set']:

                        for j in Qt[k][q]['avai_data_ver_set']:
                            for h in Qt[k][q]['avai_h_set']:
                                y_skijqh[s, k, i, j, q,h] = m.addVar (lb=0,ub=1, vtype=GRB.CONTINUOUS,name="y_%s_%s_%s_%s_%s" % (s,k,i,j,q))
                    # print(num1,num2)
        return y_skijqh

    def get_cost(self,K,I,S,x,y):
        re = 0
        for s in S:
            for k in K:
                for i in range (I):
                    for h in self.adaptive_H(self.device_type,k,i):
                        re+= x[s, k, i, h] * self.model_vm_cls.c_ki[k, i]
        return re

    def heuristic_rounding(self,K,I,S,x):
        for s in S:
            for k in K:
                for i in range(I):
                    for h in self.adaptive_H(self.device_type,k,i):
                        x[s, k, i,h] = int(np.ceil(x[s, k, i,h]))
        return x

    def process_gurobi_result_x(self,K,I,S,x):
        allocation_for_all_server_dict = {}
        for s in S:
            item = {}
            port = 8500

            sum_for_server = 0
            for k in K:
                for i in range(I):
                    for h in self.adaptive_H(self.device_type,k,i):
                        if float(x[s,k,i,h])!= 0:
                            sum_for_server+=x[s,k,i,h]
                            if sum_for_server>8:
                                print('!!!!!!!!!!!!!more than 8')
                                # exit()
                            print(port,int(x[s,k,i,h]), (x[s,k,i,h]))

                            item[str(h)+k+'_dcp_'+str(i)] = {'port':[port+x for x in range(int(x[s,k,i,h]))],'frac':self.model_vm_cls.frac_ki[k,i],
                                       'batch':h, 'timeout': self.model_vm_cls.timeout_ki[k,i],
                                       'threads': 8, 'device': self.model_vm_cls.hw_ki[k,i]
                                       }
                            port += int (x[s, k, i, h])
            allocation_for_all_server_dict[s]= item
            #print(s,item)
        return allocation_for_all_server_dict

    def process_gurobi_result_y(self,K,I,S,Qt,x,y):
        schedule_for_all_client_dict = {}
        item ={}
        result ={}
        url_list =[]
        re = {}
        ree = {}
        kk = ''
        qq= -1
        resultt= {}
        ss = ''
        for k in K:
            for q in range (len (Qt[k])):
                re[k,q] = {}

        url_dict = {}
        for s in S:
            port = 8500
            for k in K:
                for i in range (I):
                    for h in self.adaptive_H (self.device_type, k, i):
                        url_list = []
                        if float (x[s, k, i, h]) != 0:
                            url_list = [self.url_generator (s, port + x, k, i) for x in range (int (x[s, k, i, h]))]
                            url_dict[s, k, i, h] = url_list
                            port += x[s, k, i, h]
        for k in K:
            for q in range (len (Qt[k])):
                result = {}
                for s in S:
                    for i in range (I):
                        for h in self.adaptive_H (self.device_type, k, i):
                            for j in range (self.J):
                                if float (y[s, k, i, j, q, h]) > 0.0:
                                    result[f'{k}{i}{j}{q}{h}{s}'] = {'url': url_dict[s, k, i, h], 'model_ver': i, 'data_ver': j,
                                                                      'batch': h, 'prob': y[s, k, i, j, q, h]}
                                    # print(k,q,result)
                                    item[k, q] = {'writer': Qt[k][q]['writer'], 'result': result}
                                    schedule_for_all_client_dict[k, q] = item[k, q]

        # for k in K:
        #     for q in range (len (Qt[k])):
        #         sum = 0
        #         for s in S:
        #             for i in range (I):
        #                 for h in self.adaptive_H (self.device_type, k, i):
        #                     for j in range (self.J):
        #                         if float (y[s, k, i, j, q, h]) > 0.0:
        #                             sum += schedule_for_all_client_dict[k, q]['result'][s, k, i, j, q, h]['prob']
        #         print('2222222222222222',sum)

        #
        #
        # for s in S:
        #     for k in K:
        #         for i in range (I):
        #             for h in self.adaptive_H(self.device_type,k,i):
        #                 if float(x[s,k,i,h])!= 0:
        #                 for q in range (len (Qt[k])):
        #                     for j in range (self.J):
        #                         if float(y[s, k, i, j, q,h]) >0:
        #                             # if y[s, k, i, j, q,h] ==1.0:
        #                             #     result = {f'{k}_dcp_{i}':{'url':url_list,'model_ver':i,'data_ver':j,'batch':h,'prob':y[s, k, i, j, q,h]}}
        #                             #     item[k,q]={'writer':Qt[k][q]['writer'],'result':result}
        #                             #     schedule_for_all_client_dict[k, q] = item[k, q]
        #                             # else:
        #                             resultt[k,q,f'{k}_dcp_{i}'] = {'url': url_list, 'model_ver': i, 'data_ver': j,
        #                                                       'batch': h, 'prob': y[s, k, i, j, q, h]}
        #
        #                             re[k,q].update({f'{k}_dcp_{i}':resultt[k,q,f'{k}_dcp_{i}']})
        #                             # print(re)
        #                             # for key,items in resultt.items():
        #                             #
        #                             #     if (k,q) in [key[:2]]:
        #                             #         name = key[2]
        #                             #         # ree = {re[name]:items}
        #                             #         re[name] = items
        #                             #         ree[k,q] = re
        #                             #      # = {list (resultt.keys ())[0][:2]: resultt[k, q, f'{k}_dcp_{i}'] for jj in
        #                             #      #      list (resultt.keys ())[0][:2]}
        #
        #                             # re[k,q][f'{k}_dcp_{i}'] = {'url': url_list, 'model_ver': i, 'data_ver': j,
        #
        #                             # re = {list(resultt.keys())[0][:2]:resultt[k,q,f'{k}_dcp_{i}'] for jj in list(resultt.keys())[0][:2]}
        #                             # re[k,q][f'{k}_dcp_{i}'] = {'url': url_list, 'model_ver': i, 'data_ver': j,
        #                             #                           'batch': h, 'prob': y[s, k
        #                             item[k,q]={'writer':Qt[k][q]['writer'],'result':re[k,q]}
        #
        #                             # result[f'{k}_dcp_{i}'] = {'url':url_list,'model_ver':i,'data_ver':j,'batch':h,'prob':y[s, k, i, j, q,h]}
        #                             # re[k,q] = result
        #                             # item[k,q]={'writer':Qt[k][q]['writer'],'result':re[k,q]}
        #                             schedule_for_all_client_dict[k,q]=item[k,q]
        #
        #
        # for s in S:
        #     port = 8500
        #     for k in K:
        #         for i in range (I):
        #             for h in self.adaptive_H(self.device_type,k,i):
        #                 url_list = []
        #                 if float(x[s,k,i,h])!= 0:
        #                     url_list = [self.url_generator (s,port+x,k,i)for x in range(int(x[s,k,i,h])) ]
        #                     port += x[s,k,i,h]
        #                 for q in range (len (Qt[k])):
        #                     for j in range (self.J):
        #                         if float(y[s, k, i, j, q,h]) >0:
        #                             # if y[s, k, i, j, q,h] ==1.0:
        #                             #     result = {f'{k}_dcp_{i}':{'url':url_list,'model_ver':i,'data_ver':j,'batch':h,'prob':y[s, k, i, j, q,h]}}
        #                             #     item[k,q]={'writer':Qt[k][q]['writer'],'result':result}
        #                             #     schedule_for_all_client_dict[k, q] = item[k, q]
        #                             # else:
        #                             resultt[k,q,f'{k}_dcp_{i}'] = {'url': url_list, 'model_ver': i, 'data_ver': j,
        #                                                       'batch': h, 'prob': y[s, k, i, j, q, h]}
        #
        #                             re[k,q].update({f'{k}_dcp_{i}':resultt[k,q,f'{k}_dcp_{i}']})
        #                             # print(re)
        #                             # for key,items in resultt.items():
        #                             #
        #                             #     if (k,q) in [key[:2]]:
        #                             #         name = key[2]
        #                             #         # ree = {re[name]:items}
        #                             #         re[name] = items
        #                             #         ree[k,q] = re
        #                             #      # = {list (resultt.keys ())[0][:2]: resultt[k, q, f'{k}_dcp_{i}'] for jj in
        #                             #      #      list (resultt.keys ())[0][:2]}
        #
        #                             # re[k,q][f'{k}_dcp_{i}'] = {'url': url_list, 'model_ver': i, 'data_ver': j,
        #
        #                             # re = {list(resultt.keys())[0][:2]:resultt[k,q,f'{k}_dcp_{i}'] for jj in list(resultt.keys())[0][:2]}
        #                             # re[k,q][f'{k}_dcp_{i}'] = {'url': url_list, 'model_ver': i, 'data_ver': j,
        #                             #                           'batch': h, 'prob': y[s, k
        #                             item[k,q]={'writer':Qt[k][q]['writer'],'result':re[k,q]}
        #
        #                             # result[f'{k}_dcp_{i}'] = {'url':url_list,'model_ver':i,'data_ver':j,'batch':h,'prob':y[s, k, i, j, q,h]}
        #                             # re[k,q] = result
        #                             # item[k,q]={'writer':Qt[k][q]['writer'],'result':re[k,q]}
        #                             schedule_for_all_client_dict[k,q]=item[k,q]
        # print(schedule_for_all_client_dict)
        # print(schedule_for_all_client_dict)
        # exit()
        return schedule_for_all_client_dict

    def url_generator(self,server_addr,port,model_name,model_ver):
        worker_addr = self.model_vm_cls.dns_to_worker_addr(server_addr)
        url = f'http://{worker_addr}:{port}/v1/models/{model_name}_dcp_{model_ver}:classify'
        return url

    def set_construct(self, S, Qt, z_skijqh, k, i, j, h,Q):
        # z_skijqh={}
        set_all_lst = [] #[{q:[1,2,3]}]
        overlap_set = [] # [s1:[q1,q2]]
        #for q in range (len (Qt[k])):
        for q in Q:  # for q in range (len (Qt[k])):
            set_q_dict = Qt[k][q]['avai_server_set'].copy()
            for s in S:
                # s = item[0]
                # print(set_q_dict,s,k)
                print(z_skijqh)
                if (s in set_q_dict) and (z_skijqh[s, k, i, j, q, h] >0):
                    set_all_lst.append(set_q_dict)
                    break
        set_all_copy = set_all_lst.copy()

        while len(set_all_copy)!=0:
            s_max = ''
            find_s_lst_len_max = 0
            for s in S:
                find_s_lst = []
                for set_q in set_all_lst:
                    if s in set_q:
                        find_s_lst.append(set_q)
                if len(find_s_lst)>=find_s_lst_len_max:
                    find_s_lst_len_max = len(find_s_lst)
                    s_max = s
            for set_q in set_all_lst:
                if s_max in set_q:
                    set_all_copy.remove (set_q)
            overlap_set.append(s_max)
        return list(set(overlap_set)),set_all_lst

    def compute_z(self,k,i,S,Qt,y,x,Q,j,h):
        z_skijqh = {}
        sum =0
        a,b =0,0
        flag = ''
        #for k in K:
      #  for i in range (I):
      #      for h in self.adaptive_H(self.device_type,k,i):
        for s in S:
            LSH_b, RSH_b, LSH_c, RSH_c = self.constraint_is_tight (Qt, k, s, i, h, x, y,Q,j)
            for q in Q: #for q in range (len (Qt[k])):
           #     for j in range (self.J):
                    # a = float(Qt[k][q]['Y']*y[s, k, i, j, q, h]) / (self.time_slot*self.model_vm_cls.v_kih[k,i,h]*h)
                    # b = float(Qt[k][q]['Y'] * y[s, k, i, j, q, h] * IMG_SIZE[j]) /self.server_dict[s]['BW']
                    # print(111,a,b,LSH_b, RSH_b, LSH_c, RSH_c)
                    # z_skijqh[s, k, i, j, q, h] = max(a,b)
                    #
                if self.almost_equal(LSH_b,RSH_b):
                    #     flag = 'vps'
                    #     print('vps',LSH_b, RSH_b, LSH_b -RSH_b)
                    z_skijqh[s, k, i, j, q, h] = float(Qt[k][q]['Y']*y[s, k, i, j, q, h]) / (self.time_slot*self.model_vm_cls.v_kih[k,i,h]*h)
                        # print(z_skijqh[s, k, i, j, q, h],Qt[k][q]['Y'],y[s, k, i, j, q, h],self.time_slot*self.model_vm_cls.v_kih[k,i,h]*h,i,j,h,s,x[s,k,i,h] )
                elif self.almost_equal(LSH_c,RSH_c):
                    #     flag = 'bw'
                    #     print('bw')
                    z_skijqh[s, k, i, j, q, h] = float(Qt[k][q]['Y'] * y[s, k, i, j, q, h] * IMG_SIZE[j]) /self.server_dict[s]['BW']
                    # else:
                    #     print(LSH_b, RSH_b, LSH_c, RSH_c)
                    #     print('sssssssss')

        return z_skijqh,flag

    def constraint_is_tight(self,Qt,k,s,i,h,x,y,Q,j):
        LSH_b, RSH_b, LSH_c, RSH_c = 0.0,0.0,0.0,0.0
        #print(y)
        for q in Q:  # for q in range (len (Qt[k])):
            #for j in range (self.J):
            #print(q, j)

            LSH_b += float(Qt[k][q]['Y']*y[s, k, i, j, q, h])
            LSH_c += float(Qt[k][q]['Y'] * y[s, k, i, j, q, h] * IMG_SIZE[j])
        RSH_b = float(self.time_slot*self.model_vm_cls.v_kih[k,i,h]*x[s,k,i,h]*h)
        RSH_c = float(self.server_dict[s]['BW']*x[s,k,i,h])
        return LSH_b,RSH_b,LSH_c,RSH_c

    def RA_rounding(self,x,y,K,I,S,Qt):
        x = dict(x)
        x_cp = x.copy()
        for k in K:
            for s in S:
                for i in range(I):
                    for h in self.adaptive_H(self.device_type,k,i):
                        if x_cp[s, k, i, h]!=0:
                            print(s, k, i, h,x[s, k, i, h])

        #z_skijqh,flag = self.compute_z(K,I,S,Qt,y,x)
        #results = {}
        for k in K:
            for j in range (self.J):
                for i in range(I):
                    for h in self.adaptive_H(self.device_type,k,i):
                        x_fractional, y_fractional,Q,S_lst = self.compute_omiga(Qt,k,S,i,j,h,x,y)
                        # if y_fractional != {}:
                        z_skijqh, flag = self.compute_z(k, i, S_lst, Qt, y_fractional, x_fractional,Q,j,h)
                        if z_skijqh!={}:
                            overlap_set, set_all_lst = self.set_construct(S_lst, Qt, z_skijqh, k, i, j, h,Q)
                            if len(overlap_set) !=0:
                                z_skijqh = self.change_z(Qt,k,i, j,h,z_skijqh,overlap_set,set_all_lst,x,Q)
                                self.resemble_x(S_lst,Qt,k, i,j, h,z_skijqh,x,Q)
                        # else:
                        #     print ('66666',h,i,j,k)

        # if flag == 'bw':
        # print(results)
        return x
    def compute_omiga(self,Qt,k,S,i,j,h,x,y):
        x_fractional = {}
        y_fractional ={}
        q_lst =[]
        s_lst = []
        for q in range (len (Qt[k])):
            for s in S:
                if not float(x[s, k, i, h]).is_integer ():
                    y_fractional[s, k, i, j, q, h] = y[s, k, i, j, q, h]
                    x_fractional[s, k, i, h] = x[s, k, i, h]
                    q_lst.append(q)
                    s_lst.append(s)
                else:
                    y_fractional[s, k, i, j, q, h] = 0
                    x_fractional[s, k, i, h] = 0




        return x_fractional,y_fractional,q_lst,s_lst

    def change_z(self,Qt,k,i, j,h,z_skijqh,overlap_set,set_all_lst,x,Q):
        sum_s = 0
        sum_s_hat = 0
        #x_cp = x.copy()
        for q in Q:  #for q in range (len (Qt[k])):
            theta_q = Qt[k][q]['avai_server_set'].copy ()
            if theta_q in set_all_lst:
                s_hat = self.find_s_hat(theta_q,overlap_set)
                if len(theta_q)<= 1:
                    continue
                else:
                    theta_q.remove (s_hat)
                    # print(s_hat,55555)

                    for s in theta_q:
                        # z = z_skijqh[s, k, i, j, q, h]
                        # z_hat = z_skijqh[s_hat, k, i, j, q, h]

                        # if k == 'mobile':
                            # print(s,z_hat,z,i,j,h)
                        # print(sum_s,sum_s_hat)
                        # if float(sum_s).is_integer() and sum_s>0:
                        #     print(4999)
                        #     continue
                        # print(self.compute_z_in_s(s,Qt,k, i,h,z_skijqh,j),self.compute_z_in_s(s_hat,Qt,k, i,h,z_skijqh,j))
                        # print(self.compute_z_in_s(s,Qt,k, i,h,z_skijqh,j),np.ceil(x[s, k, i, h])-0.1 ,self.compute_z_in_s(s_hat,Qt,k, i,h,z_skijqh,j), np.ceil(x[s_hat, k, i, h])-0.01)
                        # upper = max(x[s_hat, k, i, h],x[s, k, i, h])

                            # break
                        # print (99,self.compute_z_in_s(s_hat,Qt,k, i,h,z_skijqh,j),self.compute_z_in_s(s,Qt,k, i,h,z_skijqh,j),np.ceil(x[s, k, i, h]), np.floor (x[s, k, i, h]), k)
#################
                        # if self.compute_z_in_s (s, Qt, k, i, h, z_skijqh, j) > np.floor(x_cp[s, k, i, h]):
                        #     print (99, self.compute_z_in_s (s, Qt, k, i, h, z_skijqh, j), np.floor (x_cp[s, k, i, h]), s, k, i,h, x_cp[s, k, i, h])

                        z_skijqh[s, k, i, j, q, h], z_skijqh[s_hat, k, i, j, q, h] = self.change_z_core(z_skijqh[s, k, i, j, q, h],z_skijqh[s_hat, k, i, j, q, h])
##############
                           # break
                            # return z_skijqh
                        # else:
                        #     print (88, self.compute_z_in_s (s, Qt, k, i, h, z_skijqh, j), (x[s, k, i, h]))


                                # print(3333,k,s,z_hat,z,i,j,h)
                        # else:
                        # print(z,z_hat)

            # else:
            #     print(99999999)

        # print(z_skijqh)
        return z_skijqh

    def change_z_core(self,z,z_hat):
        if z_hat - np.floor (z) <= np.ceil (z) - z:  # to s_hat
            # print(111,q,k,s,z_hat,z,i,j,h)

            z_hat = z_hat + z - np.floor (z)
            z = np.floor (z)
            # z_skijqh[s_hat, k, i, j, q, h] = z_hat
            # z_skijqh[s, k, i, j, q, h] = z
            # sum_s += z
            # sum_s_hat += z_hat

            # break
            # print(4444,k,s,z_hat,z,i,j,h)
        else:

            # print(2222,q,k,s,z_hat,z,i,j,h)
            z_hat = z_hat - (np.ceil (z) - z)
            z = np.ceil (z)
        return z,z_hat
            # z_skijqh[s_hat, k, i, j, q, h] = z_hat
            # z_skijqh[s, k, i, j, q, h] = z
            # sum_s += z
            # sum_s_hat += z_hat


    def find_s_hat(self,theta_q,overlap_set):
        for s in overlap_set:
            if s in theta_q:
                return s

    def resemble_x(self,S,Qt,k, i,j, h,z_skijqh,x,Q):

        for s in S:
            sum_z = self.compute_z_in_s (s, Qt, k, i, h, z_skijqh, j,Q)
            # print(222,s, k, i, h,x[s, k, i, h],sum_z)
            x[s, k, i, h] = int (np.ceil (sum_z))
            # if (s, k, i, h) in x_fractional.keys ():
            #     x[s, k, i, h] = int (np.ceil (sum_z))
            #     print('issss',s, k, i, h,x[s, k, i, h],sum_z)
            #
            # else:
            #     x[s, k, i, h] = int (np.ceil (x[s, k, i, h]))
            #     # print('nottt', s, k, i, h,x[s, k, i, h],sum_z)

        #return x

    def compute_z_in_s(self,s,Qt,k, i,h,z_skijqh,j,Q):
        sum_z = 0
        for q in Q:  #for q in range (len (Qt[k])):
            for n in range(self.J):
                # print(z_skijqh[s, k, i, n, q, h])
                sum_z += z_skijqh[s, k, i, n, q, h]
        # print(s,k,i,h,sum_z)

        return sum_z

    def resemble_y(self,z,flag,K,I,S,Qt):
        y ={}
        for k in K:
            for i in range (I):
                for h in self.adaptive_H(self.device_type,k,i):
                    for s in S:
                        for q in range (len (Qt[k])):
                            for j in range (self.J):
                                if flag == 'bw':
                                    y[s, k, i, j, q, h] = z[s, k, i, j, q, h]* self.server_dict[s]['BW']/(Qt[k][q]['Y'] * IMG_SIZE[j])
                                else:
                                    y[s, k, i, j, q, h] = z[s, k, i, j, q, h]*self.time_slot * self.model_vm_cls.v_kih[k, i, h] * h / (Qt[k][q]['Y'])
        return y

    def almost_equal(self,x,N):
        # print(x,N)
        if x - N >-0.01 and x - N <=0.1:
            return True
        else:
            return False
                    
                    
            
            
        
        
        
        











