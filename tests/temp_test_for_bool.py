# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:13:47 2024

@author: emili
"""
import numpy as np
from bisect import bisect

class StepInterpolate:
    def __init__(self,x,t):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(t, np.ndarray):
            t = np.array(t)
        ind = np.argsort(t)
        self.t = t[ind]
        self.x = x[ind]
        

    def __call__(self, times):
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        out = []
        for time in times:
            if np.any(np.isclose(time, self.t)):
                out.append(self.x[np.where(np.isclose(time,self.t))[0][0]])
            else:
                idx = bisect(self.t,time)
                if idx == 0:
                    out.append(None)
                else:
                    out.append(self.x[idx-1])
                    
        return out

def main():
    t = np.array([0.5, 1, 1.3, 1.5, 3.2, 5, 6, 7, 8, 9])
    x = np.array([0,   0,   0,   1,   0, 0, 0, 1, 1, 1])
    interp = StepInterpolate(x,t)
    print(interp([0,0,1,1.3,3.19999]))
    #print(interp([0,0.5,1.]))
if __name__ == "__main__":
    main()