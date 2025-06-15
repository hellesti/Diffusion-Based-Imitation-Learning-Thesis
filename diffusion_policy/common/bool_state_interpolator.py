import numpy as np
from bisect import bisect
import numbers
from typing import Union
import time



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
        if isinstance(times, numbers.Number):
            times = np.array([times])
        elif not isinstance(times, np.ndarray):
            times = np.array(times)
        out = []

        #print("fun. input:",times)
        #print("self.times:",self.t)
        #print("self.datapoints", self.x)

        for time_ in times:
            #print(time_)
            if np.any(np.equal(time_, self.t)):
                assert not (len(np.where(np.equal(time_, self.t))) > 1)
                out.append(self.x[np.where(np.equal(time_,self.t))[0][0]])
            else:
                #print("BISECTING")
                idx = bisect(self.t,time.monotonic())
                if idx == 0:
                    out.append(None)
                else:
                    out.append(self.x[idx-1])

        #print(out)
        #print()
        return out

class BoolStateInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        #print("boolinterp times:",times)
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            self._times = times
            self._poses = poses
        self.pos_interp = StepInterpolate(poses,times)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        if isinstance(t, numbers.Number):
            t = np.array([t])
        
        pose = self.pos_interp(t)
        return pose

    def __str__(self):
        return(str(self._poses)+"\n"+str(self._times))

    @property
    def times(self) -> np.ndarray:
        return self._times

    def try_trim(self, #Should trim according to this rule but NOT change the timing
            start_t: float, end_t: float
            ) -> "BoolStateInterpolator":
        assert start_t < end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        if np.any(should_keep):
            keep_times = times[should_keep]
            all_times = keep_times #np.concatenate([[start_t], keep_times, [end_t]])
            # remove duplicates, Slerp requires strictly increasing x
            all_times = np.unique(all_times)
            # interpolate
            all_poses = self(all_times)
            self._times = all_times
            self._poses = all_poses
            return BoolStateInterpolator(times=all_times, poses=all_poses)
        else:
            return BoolStateInterpolator(times=self.times, poses=self._poses)
    
    def drive_to_waypoint(self, pose, curr_time) -> "BoolStateInterpolator":
        final_interp = BoolStateInterpolator(curr_time, pose)
        return final_interp

    def schedule_waypoint(self,
            pose, 
            action_time, 
            curr_time = None
        ) -> "BoolStateInterpolator":


        if curr_time is not None:
            if action_time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                print("gripper: given time for waypoint < current time")
                return self
        #print("ACTION TIME:", action_time)
        #print("appending to:", self._times)

        if curr_time is not None:
            self.try_trim(curr_time-0.5, action_time)
        else:
            self.try_trim(-np.inf, action_time)

        out_times = self._times
        out_poses = self(out_times)
        out_times = np.append(out_times, action_time)
        out_poses = np.append(out_poses, pose)
        #print("resulting times:",out_times)
        #print("actions:", out_poses)
        out = BoolStateInterpolator(out_times, out_poses)
        # print("-----")
        # print(out_times)
        # print(out_poses)
        # print(out._times)
        # print(out._poses)
        # print("-----")
        return out