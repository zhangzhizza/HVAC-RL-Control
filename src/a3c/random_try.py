import numpy as np
def f():
    """
    Given a state, run stochastic policy network to give an action.
    
        Args:
            state: np.ndarray, 1*m where m is the state feature dimension.
                Processed normalized state.
            sess: tf.Session.
                The tf session.
        
        Return: int 
            The action index.
    """
    a = [0.1,0.2,0.5,0.2]
    uni_rdm = np.random.uniform();
    imd_x = uni_rdm;
    for i in range(len(a)):
        imd_x -= a[i];
        if imd_x <= 0.0:
            return i;