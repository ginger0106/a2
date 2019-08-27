
class server_cls():
    def __init__(self,info_path):
        self.server_dict={}
        self.server_info_path = info_path

    def get_server_info(self):
        with open (self.server_info_path) as f:
            for s in f.readlines ():
                s = s.split ()
                self.server_dict[s[0]] = {'addr':s[0],'BW':float(s[1]),'R':s[2],'port':int(s[3])}
        return self.server_dict
        # print(self.server_dict)


# server_cls().get_server_info()