# ---------------------------------------------------------------------------
# Based on some code from Machine Learning in Physics Course at 
# Florida State University.
# Harrison B. Prosper
# Created: Tues Jan 27 2026
# ---------------------------------------------------------------------------
import os, sys, re
import numpy as np
import time
import socket
import math

try:
    import h5py
except:
    raise ImportError('''
    Please install h5py:

        conda install h5py
    ''')

from types import SimpleNamespace as NS
from datetime import datetime

try:
    import scipy.stats as st
except:
    raise ImportError('''
    Please install scipy:

        conda install scipy
    ''')

try:
    import yaml
except:
    raise ImportError('''
    Please install yaml:

        conda install yaml
    ''')
# ---------------------------------------------------------------------------
def round_sig(x, sig=5):     # ChatGPT 5.2
    if x == 0:
        return 0.0
    return float(round(x, sig - int(math.floor(math.log10(abs(x)))) - 1))
# -------------------------------------------------------------------------- 
def round_sig_np(x, sig=5):  # ChatGPT 5.2
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    nz = x != 0
    mags = np.floor(np.log10(np.abs(x[nz]))).astype(int)
    decimals = sig - mags - 1

    out[nz] = float(np.round(x[nz], decimals))
    return out
# ---------------------------------------------------------------------------
def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M")
 
def elapsed_time(now, start):
    etime = now() - start    
    t = etime
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    seconds = t - 60 * minutes
    etime_str = "%2.2d:%2.2d:%2.2d" % (hours, minutes, seconds)
    return etime_str, etime, (hours, minutes, seconds)
# ---------------------------------------------------------------------
# Simple class to store state and widgets
class Bag(NS):
    def __init__(self, verbose=False):
        super().__init__()
        self.gfx = NS() # widgets stored in this "bag"
        self.verbose = verbose

    def set(self, name, value):
        setattr(self, name, value)

    def get(self, name):
        try:
            return getattr(self, name)
        except:
            return None
        
    def clear(self):
        # free all graphics objects
        for key, obj in self.gfx.__dict__.items():
            try:
                obj.delete()
                if self.verbose:
                    print(f'\tdeleted: {key}')
            except:
                pass
        # clear all references to graphics objects.
        self.gfx.__dict__.clear()        
# ---------------------------------------------------------------------------
class CircularBuffer: # Help from ChatGPT 5.2
    def __init__(self, size, n):
        self.size, self.n = size, n
        self.data  = np.zeros((size, n))
        self.idx   = 0
        self.count = 0   # number of valid elements

    def append(self, row):
        self.data[self.idx] = row
        self.idx = (self.idx + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def get_k_behind(self, k):
        if k >= self.count:
            return None
        pos = (self.idx - 1 - k) % self.size
        return self.data[pos]

    def get_oldest(self):
        return self.get_k_behind(self.size - 1)
# ---------------------------------------------------------------------------
class SimLogger:
    '''
    Log simulation data to an HDF5 (H5) file. 

    Arguments:
    
        filename (str):     Name of H5 file.
        
        bag (Bag):          Bag containing attributes and datasets to save.
                            An attribute is a single instance of an item.
                            A dataset comprises a sequence of items, saved
                            with the write method.
                        
        attributes (list):  List specifying attributes from the bag to be
                            saved to the H5 file. 

        datasets (list):    List specifying datasets from the bag to be
                            saved to the H5 file using the write function.

        save_every (int):   Save dataset items every "save_every" times that
                            the write function is called. Default=10.

        flush_every (int):  Flush dataset items to file "flush_every" times that
                            the write function is called. Default=500.

    '''
    def __init__(self, filename, bag, attributes, datasets, 
                 save_every=10,
                 flush_every=500,
                 chunk=16,
                 debug=False, 
                 delete_old_file=True):

        self.filename = filename
        if delete_old_file:
            os.system(f'rm -rf {filename}')
        elif sys.path.exists(filename):
            raise RuntimeError(f'''
    File {filename} already exists. Either rename it or delete it!
            ''')

        self.chunk   = chunk
        self.funtype = re.compile('float|int|bool|str')
        self.alltype = re.compile('float|int|bool|str|array')
        self.getnptype= re.compile("'.*'")
        self.ftype = np.float32
        self.itype = np.int32
        self.debug = debug 
        
        self.attributes = attributes
        self.datasets   = datasets
        
        # Data are saved every "save_every" calls to write
        self.save_every  = save_every

        # Data are flushed to "disk" every "flush_every" calls to write
        self.flush_every = flush_every

        # Index of saved dataset element. This increments every
        # time an item is saved.
        self.index = 0

        # Loop index. This increments every time the 
        # write method is called.
        self.loop_index = 0

        # Open file
        self.f = h5py.File(filename, "w")
        
        # Attributes to be saved
        for key in self.attributes:
            obj = bag.get(key)
            if obj is None:
                raise RuntimeError(f"bag object missing item '{key}'")

            # The current item exists in bag
            self.__register_item(key, obj, which='attribute')

        self.__register_item('save_every', save_every, which='attribute')

        for key in self.datasets:
            obj = bag.get(key)
            if obj is None:
                raise RuntimeError(f"bag object missing item '{key}'")

            # The current item exists in bag
            self.__register_item(key, obj, which='dataset')

    def write(self, bg):
        if self.loop_index % self.save_every == 0:
            for key in self.datasets:
                obj = bg.get(key)
                self.f[key].resize(self.index + 1, axis=0)
                self.f[key][self.index] = np.asarray(obj, dtype=self.ftype)    
            self.index += 1
            
        if self.loop_index % self.flush_every == 0:
            self.flush()
            
        self.loop_index += 1

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()

    def __fun_type(self, obj):
        t = self.funtype.findall(str(type(obj)))
        if len(t) == 0:
            return None
        else:
            return t[0]

    def __all_type(self, obj):
        t = self.alltype.findall(str(type(obj)))
        if len(t) == 0:
            return None
        else:
            return t[0]
            
    def __get_nptype(self, obj):
        t = self.getnptype.findall(str(type(obj)))
        if len(t) == 0:
            return None
        else:
            return t[0][1:-1]
            
    def __register_item(self, key, obj, which):

        if which == 'attribute':
            
            the_type = self.__fun_type(obj)
            if the_type is None:
                raise RuntimeError(f'''
    The type of attribute {key}, {str(type(obj))}, must be either
    a float, int, bool, or str.
                ''')
            if self.debug:
                print('register(variable)', key, the_type)
            
            self.f.attrs[key] = eval(f'{the_type}(obj)')
            
            if the_type == 'float':
                self.f.attrs[key] = self.f.attrs[key].astype(self.ftype)
                
            elif the_type == 'int':
                self.f.attrs[key] = self.f.attrs[key].astype(self.itype)
        else:
            # Assume we have a dataset.
            # Either: 
            #  1  of a fundamental type (float, int, bool, or str), or
            #  2. of a numpy array
            
            # chunk frames for efficient saving

            the_type = self.__all_type(obj)

            if self.debug:
                print('register(dataset)', key, the_type)
            
            if the_type is None:
                raise RuntimeError("""
        Only fundamental types float, int, bool str and numpy arrays are 
        supported.
                    """)
 
            elif the_type == 'array':
                self.f.create_dataset(
                    key, shape=(0, *obj.shape), maxshape=(None, *obj.shape),
                    dtype=self.ftype, chunks=(self.chunk, *obj.shape),
                    compression="gzip", compression_opts=4)
                    # arrange bytes so that identical bytes are adjacent
                    #shuffle=True,  
                    #scaleoffset=4)  # 4 sig figs
                
            else:
                # This is a fundamental type
                self.f.create_dataset(
                    key, shape=(0,), maxshape=(None,),
                    dtype=self.ftype, chunks=(4096,),
                    compression="gzip", compression_opts=4)

    def __str__(self):
        s = f"\nFilename: {self.filename}\n"
        s += "  Attributes:\n"
        for name, obj in self.f.attrs.items():
            s += f"    {name:16s}: {self.__get_nptype(obj)}\n"
            
        s += "\n  Datasets:\n"
        for name, obj in self.f.items():
            s += f"    {name:16s}: {self.__get_nptype(obj)}\n"
        return s
# -------------------------------------------------------------------------- 
def require(f, key, where="dataset"):
    try:
        return f[key] if where=="dataset" else f.attrs[key]
    except KeyError:
        raise RuntimeError(
            f"{f.filename} missing required {where} '{key}'")

def request(f, key, where="dataset", crash=True):
    try:
        return f[key] if where=="dataset" else f.attrs[key]
    except KeyError:
        if crash:
            raise RuntimeError(
            f"{filename} does not contain requested {where} '{key}'")
        else:
            print(f"{filename} does not contain requested {where} '{key}'")
            return None
# -------------------------------------------------------------------------- 
class SimReader:
    """
    Read simulation data from an HDF5 (H5) file. 

    Arguments:
    
        filename (str): Name of H5 file.
        items (list):   Optional list specifying which items 
                        to read from file. Default: read everything.
    """
    def __init__(self, filename, items=[]):

        self.filename = filename
        self.getnptype= re.compile("'.*'")
        
        self.f = h5py.File(filename, "r")
                            
        # Determine items to read
        attributes = list(self.f.attrs.keys())  # attributes
        datasets   = list(self.f.keys())        # datasets/groups
        if len(items) == 0:
            items = attributes  + datasets

        # Maps to keep track of attributes/datasets
        self.attrs = {}
        self.datasets = {}
        
        for item in items:
            if item in attributes:
                cmd = f'self.{item} = request(self.f, "{item}", "attribute")'
                exec(cmd)
                self.attrs[item] = eval(f'self.{item}')
                
            elif item in datasets:
                cmd = f'self.{item} = request(self.f, "{item}", "dataset")'
                exec(cmd)
                self.datasets[item] = eval(f'self.{item}')
            else:
                raise KeyError(f'''
    In file {filename}, the item {item} not found!
                ''')
        try:
            # Compute time interval between saved frames
            self.Dt = self.dt * self.save_every
        except:
            self.Dt = 1.0

        try:
            self.max_frames = len(self.r)
        except:
            self.max_frames = -1
            
    def header(self):
        bg = Bag()
        for key, value in self.attrs.items():
            bg.set(key, value)
        bg.read = self.read    # read function of SimReader needed by 
        return bg
        
    def read(self, i):
        bg = Bag()
        bg.t = i * self.Dt
        for key, obj in self.datasets.items():
            bg.set(key, obj[i].astype(np.float32, copy=True))
        return bg

    def close(self):
        self.f.close()

    def __get_nptype(self, obj):
        t = self.getnptype.findall(str(type(obj)))
        if len(t) == 0:
            return None
        else:
            return t[0][1:-1]
            
    def __str__(self):
        s = f"\nFilename: {self.filename}\n"
        s += "  Attributes:\n"
        for name, obj in self.f.attrs.items():
            s += f"    {name:16s}: {self.__get_nptype(obj)}\n"
            
        s += "\n  Datasets:\n"
        for name, obj in self.f.items():
            s += f"    {name:16s}: {self.__get_nptype(obj)}\n"
        return s
# ---------------------------------------------------------------------------
class SimServer:
    """
    Broadcast a string to any connected clients.
    """
    def __init__(self, host="127.0.0.1", port=5000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen()
        self.sock.setblocking(False)
        self.connections = []

    def poll(self):
        """Accept any new clients."""
        try:
            conn, _ = self.sock.accept()
            conn.setblocking(False)
            self.connections.append(conn)
        except BlockingIOError:
            pass

    def send(self, string):
        """Send a string to all clients."""
        msg = f"{string}<eos>".encode()
        dead = []
        for c in self.connections:
            try:
                c.sendall(msg)
            except:
                dead.append(c)
        for c in dead:
            self.connections.remove(c)

class SimClient:
    """
    Receive a string from the server.
    """
    def __init__(self, host="127.0.0.1", port=5000, tries=10):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.buffer = ""
        self.tries = tries
        
    def recv(self):
        """Wait for the next string."""
        k = 0
        active = True
        while active:
            data = self.sock.recv(1024).decode()
            if not data:
                return self.buffer

            self.buffer += data

            if "<eos>" in self.buffer:
                line, self.buffer = self.buffer.split("<eos>", 1)
                self.buffer = "" # just to be sure!
                return line
            k += 1
            active = k < self.tries
            
        return self.buffer
# --------------------------------------------------------------------------
class Missing:
    pass
MISSING = Missing()
   
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