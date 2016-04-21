import numpy as np

def data8(n=100):
    # Generates '8' shaped data
    y = np.linspace(0,1, n)

    x = np.append(np.sin(2*np.pi*y), (-np.sin(2*np.pi*y)))

    return np.column_stack((x,np.append(y,y))).astype(dtype=np.float32)

