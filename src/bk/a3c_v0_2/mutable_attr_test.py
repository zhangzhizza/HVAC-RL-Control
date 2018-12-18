import copy

class Mutable_attr():
    def __init__(self):
        self._attr = [1,2,3,4];
    
    @property
    def attr(self):
        return copy.deepcopy(self._attr);
    
    
cl = Mutable_attr();
attr = cl.attr;
print (attr);
attr.append(3);
print (cl.attr);