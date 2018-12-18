import threading
import tensorflow as tf
import numpy as np
import time
from multiprocessing import Value, Lock

class nparray():
    def __init__(self):
        self.a = [];
    def appendto(self, a):
        self.a.append(a);
    def geta(self):
        return self.a;
    
class worker():
    
    def __init__(self, var, id):
        self._op = tf.assign(var, var + 1, use_locking = True);
        self.id = id
    def train(self, sess):
        for i in range(1000):
            #time.sleep(0.01);
            #print (self.id)
            sess.run(self._op);
            
             

        
global_lock = Lock();
var = tf.Variable(0);
sess = tf.Session();
sess.run(tf.global_variables_initializer());
workers = [worker(var, id) for id in range(5)]
nparray = [];
threads = [];

coord = tf.train.Coordinator()
counter = 0;
for worker in workers:
    worker_work = lambda: worker.train(sess);
    thread = threading.Thread(target = (worker_work));
    thread.start();
    threads.append(thread);
    counter += 1;

coord.join(threads)

print (sess.run(var));