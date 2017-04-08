import threading
from multiprocessing import Value, Lock

class Thread_A(threading.Thread):
    def __init__(self, num, lock):
        threading.Thread.__init__(self);
        self.lock = lock
        self.num = num;
    
    def run(self):
        for i in range(100):
            with self.lock:
                self.num.value+=1;
            
num = Value('d', 0.0);
lock = Lock();

a = Thread_A(num, lock);
b = Thread_A(num, lock);

a.start();
b.start();

a.join()
b.join()

print (num.value)