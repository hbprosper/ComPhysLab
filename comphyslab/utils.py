# ----------------------------------------------------------------------------
# Based on some code from Machine Learning in Physics Course at Florida State 
# University.
# Harrison B. Prosper
# Created: Tues Jan 27 2026
# ----------------------------------------------------------------------------
import os, sys, re
import numpy as np
import time
from datetime import datetime
try:
    import scipy.stats as st
except:
    raise ImportError('''
    Please install scipy:

        conda install scipy
    ''')

import yaml
# ----------------------------------------------------------------------------
def elapsed_time(now, start):
    etime = now() - start    
    t = etime
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    seconds = t - 60 * minutes
    etime_str = "%2.2d:%2.2d:%2.2d" % (hours, minutes, seconds)
    return etime_str, etime, (hours, minutes, seconds)
# ---------------------------------------------------------------------------
class Missing:
    pass
MISSING = Missing()

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M")
    
class Config:
    '''
        Manage application configuration

          name:      name stub for all files, including the yaml file
    '''
    def __init__(self, name, dirname=MISSING, verbose=0):
        '''
        name  : string   Stub for all files, including the yaml file, or 
                         the name of a yaml file. A yaml file is identified 
                         by the extension .yaml
                
                            1. if name is a name stub, create a new yaml object.
                            2. if name is a yaml filename, create the yaml object
                               from the file.
                         
        dirname : string If given use this as the name of the folder: 
                         config/<dirname>.
        '''

        self.dirname = dirname
        if self.dirname is MISSING:
            self.cfgdir = "config"
        else:
            self.cfgdir = f"config/{self.dirname}"

        # make config directory
        os.makedirs("config", exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True) 
            
        # check if a yaml file has been specified
        if name.endswith('.yaml') or name.endswith('.yml'):
            self.cfg_filename = name # cache filename
            self.load(name)
        else:
            # this not a yaml file specification, assume it is a name stub
            # and build a Python dictionary that specifies the structure of
            # 
            self.cfg = {}
            cfg = self.cfg

            cfg['name'] = name
            cfg['time'] = time.ctime()
            
            # create a default name for yaml configuration file
            # this name will be used if a filename is not
            # specified in the save method
            self.cfg_filename = f'{self.cfgdir}/{name}_config.yaml'
    
        if verbose:
            print(self.__str__())
        
    def load(self, filename):
        # make sure file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f'{filename}')
        
        # read yaml file and cache as Python dictionary
        with open(filename, mode="r") as file:
            self.cfg = yaml.safe_load(file)

    def save(self, filename=None):
        # if no filename specified use default filename
        if filename == None:
            filename = self.cfg_filename

        # require .yaml extension
        if not (filename.endswith('.yaml') or filename.endswith('.yml')):
            raise NameError('the output file must have extension .yaml')
            
        # save to yaml file
        open(filename, 'w').write(self.__str__())
        
    def __call__(self, key, value=None):
        '''
        Return the value of the specified key.

        Notes
        -----
        1. If the key is in the dictionary and value is specified then 
        update the value of the key and return the value, otherwise 
        return the existing value of the key.

        2. If the key is not in the dictionary add it to the dictionary with
        the specified value and return the value. If no value is given raise 
        a KeyError exception.
        '''
        # this method can be used to fill out the rest
        # of the Python dictionary
        keys = key.split('/')
        
        # if key exists and value !=None update the value
        # else return its value
        cfg = self.cfg
        
        for ii, lkey in enumerate(keys):
            depth = ii + 1
            
            if lkey in cfg:
                # key is in dictionary
                
                val = cfg[lkey]
                if depth < len(keys):
                    # recursion
                    cfg = val
                else:
                    if type(value) == type(None):
                        # key exists and no value has been specified
                        # so return existing value
                        value = val
                    else:
                        # key exists and a value has been specified
                        # so update key and return new value
                        cfg[key] = value # update value
                    break
            else:
                # key is not in dictionary object, so add it
                
                if value == None:
                    # no value specified, so we can't add this key
                    raise KeyError(f'key "{lkey}" not found')
                    
                elif depth < len(keys):
                    cfg[lkey] = {}
                    cfg = cfg[lkey]
                else:
                    try:
                        cfg[lkey] = value
                    except:
                        pkey = keys[ii-1]
                        print(
                            f'''
    Warning: key '{key}' not created because '{pkey}' is 
    of type {str(type(pkey))}
                        ''')
        return value

    def __str__(self):
        # return a pretty printed string of the yaml object (help from ChatGPT)
        return str(yaml.dump(
            self.cfg,                 
            sort_keys=False,           # keep key order
            default_flow_style=False,  # use block style 
            indent=1,                  # indentation level
            allow_unicode=True))