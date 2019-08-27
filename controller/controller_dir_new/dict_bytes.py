import json
import logging

class dict_bytes():
    def __init__(self):
        pass
    async def send_dict2bytes(self,input_dict,writer):
        input_bytes = json.dumps (input_dict)
        writer.write (input_bytes.encode ()+'\n'.encode())
        await  writer.drain ()
        logging.info(f'Send: send data {input_dict} ')

    async def read_bytes2dict(self,reader,writer):
        addr = writer.get_extra_info ('peername')
        data_bytes = await reader.readuntil(separator=b'\n')
        data_dict = json.loads (data_bytes.decode ())
        logging.info (f'Receive: receive data {data_dict} from {addr} ')
        return data_dict
