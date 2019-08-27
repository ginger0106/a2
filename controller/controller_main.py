from controller_dir_new.a2_controller import a2_controller
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a",default='0.0.0.0',type=str,help='controller_ip') #server ip addr
parser.add_argument("-p",default=8888,type=int,help='controller_port,8888')
parser.add_argument("-x",default='a2',type=str,help='placement policy:adapative_m/adapative_d/a2/heu') #placement policy adapative_m,adapative_d
parser.add_argument("-t",default=60,type=int,help='time slot')
parser.add_argument("-d",default='cpu',type=str,help='device')
parser.add_argument("-n",default=2,type=int,help='client number')
parser.add_argument("-debug",default='test',type=str,help='debug mode: test/aws/aws_test')


#version_stg,device_type,time_slot,con_addr,con_port,client_num
args = parser.parse_args()
a2_controller(args.x,args.d,args.t,args.a,args.p,args.n,args.debug)