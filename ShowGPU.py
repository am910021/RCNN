#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':
    gpus = get_available_gpus()
    print()
    print("============Find available gpus==============")
    print('\r\n'.join(gpus))