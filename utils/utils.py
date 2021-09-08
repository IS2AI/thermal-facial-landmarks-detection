import numpy as np

# compute mask for one landmark

# i - landmark's number
# landmarks - array with all landmarks
# m, n - dimensions of mask

def one_key_mask(i, landmarks, m, n):
    
    b = np.zeros((m, n))   
    
    for x in range(m):
        for y in range(n):
            b[x, y] = 0.5**(max([abs(landmarks[i, 0] - x), abs(landmarks[i, 1] - y)]))
            
    return b