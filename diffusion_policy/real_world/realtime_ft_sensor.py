# -*- coding: utf-8 -*-

#=================

import NetFT
import os
import time
import enum
import numbers
from typing import Union
import multiprocessing as mp
from diffusion_policy.common.precise_sleep import precise_wait
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from diffusion_policy.shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Command(enum.Enum):
    STOP = 0
    SET_ZERO = 1

class FTSensor(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager, 
            ft_sensor_ip = None, 
            frequency = 60,
            ft_transform_matrix = np.identity(6),
            launch_timeout=3,
            verbose=False,
            #receive_keys=None,
            get_max_k=128,
            ):
        
        super().__init__(name="FTSensor")
        self.ft_sensor_ip = ft_sensor_ip
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.frequency = frequency
        self.ft_transform_matrix = ft_transform_matrix
        #build input queue
        example = {
            'cmd': Command.STOP.value,
        }

        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue

        example = dict()
        example['ft_ee_wrench'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        example['ft_sensor_receive_timestamp'] = time.time()
        
        
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.move_done_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        #self.receive_keys = receive_keys
        self.soft_real_time = False
        self.zero_point = np.zeros(6)
    # =========== miscellaneous =========== 
    def set_zero_point(self):
        message={
            'cmd': Command.SET_ZERO.value
        }
        self.input_queue.put(message)

        


    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FTSensor] Sensor process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self):
        self.stop()
        
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    
    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
        if np.all(self.ft_transform_matrix == np.identity(6)):
            print("Warning: FT sensor transform is identity. Did you forget to pass it to the controller?")
        # start rtde
        ft_sensor_ip = self.ft_sensor_ip
        ft_sensor = NetFT.Sensor(ft_sensor_ip)
    
        try:
            if self.verbose:
                print(f"[FTSensor] Connect to FT sensor: {ft_sensor_ip}")

            #init pose

            # main loop
            dt = 1. / self.frequency
            # use monotonic time to make sure the control loop never go backward
            t_start = time.monotonic()


            iter_idx = 0
            keep_running = True

            while keep_running:
                t_now = time.monotonic()

                t_next_loop = (t_start + (iter_idx+1)*dt)

                state = {}
                
                state["ft_ee_wrench"] = (self.ft_transform_matrix @ np.array(ft_sensor.getMeasurement()))/1000000-self.zero_point #1000000 is steps per N.
                #print(state["ft_ee_wrench"])

                if self.verbose:
                    print(state)

                state['ft_sensor_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    
                    elif cmd == Command.SET_ZERO.value:
                        wrench = state["ft_ee_wrench"]
                        self.zero_point = wrench
                        
                    else:
                        keep_running = False
                        break

                # regulate frequency
                precise_wait(t_next_loop)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    pass #not implemented
                    #print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            ft_sensor.sock.close() #library does not contain a function for closing the connection, so we close it manually
            self.ready_event.set()

            if self.verbose:
                print(f"[FTSensor] Disconnected from FT sensor: {ft_sensor_ip}")

def main():
    pass

if __name__ == "__main__":
    main()
