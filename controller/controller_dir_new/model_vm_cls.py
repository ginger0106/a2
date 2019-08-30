import numpy as np
# model_info_path = 'controller_dir_new/config/model_info.txt'
# DNS_list_path = 'controller_dir_new/config/DNS_list.txt'

IMG_SIZE = np.load('controller_dir_new/measurements/IMG_SIZE.npy')
MBV1ACC = np.load('controller_dir_new/measurements/mobile_v1_acc_new.npy')
MBV1FLP = np.array([1.1375, 1.1375 - 1.1375 * 0.4, 1.1375 - 1.1375 * 0.6, 1.1375 - 1.1375 * 0.8])

RES18ACC = np.load('controller_dir_new/measurements/res18_acc_new.npy')
RES18FLP = np.array([3.65, 3.65 - 3.65 * 0.4, 3.65 - 3.65 * 0.6, 3.65 - 3.65 * 0.8])



MODEL_NUM =2
MODEL_VERSION =4
DEVICE_NUM =2
BATCH_VERSION =4
np.random.seed(2)
# QPS = np.round(np.random.lognormal(5,4,MODEL_NUM*MODEL_VERSION*DEVICE_NUM)).reshape(MODEL_NUM,MODEL_VERSION,DEVICE_NUM)
QPS = np.load('controller_dir_new/measurements/QPS.npy')
LATENCY = np.load('controller_dir_new/measurements/Average_latency.npy')


class model_vm():
    def __init__(self,device_type,DNS_path):
        # self.model_vm_dict = {}
        self.f_ki = {}
        self.c_ki = {}
        self.v_kih = {}
        self.h_ki = {}
        self.r_ki = {}
        self.hw_ki = {}
        self.frac_ki = {}
        self.timeout_ki = {}
        self.dns_dict ={}
        self.t_kih = {} #latency
        self.device_type = device_type
        self.vm_model_type_dict = {'res18':0, 'mobile':1}
        self.vm_flp_dict = {'res18':RES18FLP,'mobile':MBV1FLP}
        self.vm_device = {'cpu':0, 'gpu':1}
        self.vm_version_dict = {'res18':MODEL_VERSION,'mobile':MODEL_VERSION}
        self.model_ver_dict = self.vm_version_dict
        self.vm_batch_dict = {'res18':[2**x for x in range(BATCH_VERSION)],'mobile':[2**x for x in range(BATCH_VERSION)]}
        self.vm_time_out = {'cpu':5, 'gpu':15}
        self.DNS_list_path = DNS_path
        self.compute_ki = self.get_ki()

    def read_DNS_info(self):
        with open(self.DNS_list_path) as f:
            num = 0
            for vm in f.readlines():
                #print(vm)
                vm = vm.split(' ')
                # print(vm)
                self.dns_dict[num] = {'server_addr':vm[0],'worker_addr':vm[1],'hw':vm[2]}
                num+=1
        return self.dns_dict


    def get_ki(self):
        for model_name,model_ver_num in self.vm_version_dict.items():
            for i_id in range (model_ver_num):
                    self.f_ki[model_name,i_id] = self.vm_flp_dict[model_name][i_id]
                    self.c_ki[model_name,i_id] = self.f_ki[model_name,i_id]
                    self.h_ki[model_name,i_id] = self.vm_batch_dict[model_name]
                    self.hw_ki[model_name,i_id] = self.device_type
                    self.frac_ki[model_name,i_id] = 1.000
                    self.timeout_ki[model_name,i_id] = self.vm_time_out[self.device_type]
                    for h in self.vm_batch_dict[model_name]:
                        self.v_kih[model_name,i_id,h] = QPS[self.vm_model_type_dict[model_name]][i_id][self.vm_device[self.device_type]][int(np.sqrt(h))]
                        self.t_kih[model_name,i_id,h] = LATENCY[self.vm_model_type_dict[model_name]][i_id][self.vm_device[self.device_type]][int(np.sqrt(h))]

    def dns_to_worker_addr(self,server_addr):
        for num,dns in self.read_DNS_info().items():
            if self.device_type == dns['hw'] and server_addr == dns['server_addr']:
                return dns['worker_addr']
            else:
                return server_addr
