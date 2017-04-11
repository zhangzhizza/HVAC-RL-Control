import threading
import tensorflow as tf
import numpy as np
from multiprocessing import Value, Lock

class nparray():
    def __init__(self):
        self.a = [];
    def appendto(self, a):
        self.a.append(a);
    def geta(self):
        return self.a;
    
class worker():
        
    def train(self, nparray, lock, a):
        for _ in range(5):
            nparray.append(a);
             

        
global_lock = Lock();
workers = [worker() for _ in range(5)]
nparray = [];
threads = [];

counter = 0;
for worker in workers:
    worker_work = lambda: worker.train(nparray, global_lock, counter);
    thread = threading.Thread(target = (worker_work));
    thread.start();
    thread.join();
    counter += 1;

print (nparray)        