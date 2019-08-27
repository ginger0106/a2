
def read_bw():
    bw_dict = {}
    with open('./bw.txt') as f:
        for item in f.readlines():
            print(item)
            item = item.split(',')
            bw_dict[item[0]]=float(item[2])
    return bw_dict

read_bw()
