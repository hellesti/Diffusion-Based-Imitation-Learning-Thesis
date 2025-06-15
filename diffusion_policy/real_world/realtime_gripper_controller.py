
import os
import time
import enum
import multiprocessing as mp
from diffusion_policy.common.precise_sleep import precise_wait
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from diffusion_policy.shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.real_world.gripper_api.gripper_modbus_tcp import Gripper
from diffusion_policy.common.bool_state_interpolator import BoolStateInterpolator


class Command(enum.Enum):
    STOP = 0
    SCHEDULE_WAYPOINT = 1

class GripperController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager, 
            gripper_ip = None, 
            frequency = 30,
            launch_timeout=5,
            gripper_init_pos=None,
            verbose=False,
            #receive_keys=None,
            get_max_k=128,
            ):
 
        if gripper_init_pos is not None:
            assert isinstance(gripper_init_pos, bool)
        
        super().__init__(name="GripperController")
        self.gripper_ip = gripper_ip
        self.launch_timeout = launch_timeout
        self.gripper_init_pos = gripper_init_pos
        self.verbose = verbose
        self.frequency = frequency

        #build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': True,
            'target_time': time.time()
        }

        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue

        # build ring buffer
        # if receive_keys is None:
        #     receive_keys = [
        #         'gOBJ',
        #         'gSTA',
        #         'FaultStatus',
        #         'gPR',
        #         'Finger_Position',
        #         'Finger_Current'
        #     ]
            
        example = dict()
        #for key in receive_keys:
        #    example[key] = np.array(getattr(rtde_r, 'get'+key)())

        
        example['object_is_grabbed'] = np.array([True])
        example['gPR'] = np.array([255]) #Grippper position request echo, between 0 and 255
        example['gPO'] = np.array([255]) #Actual (encoder) gripper position, between 0 and 255
        example['gCU'] = np.array([255]) #Motor current, between 0 and 255. Actual current approximation is given in gripper manual

        example['gripper_receive_timestamp'] = time.time()
        
        
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
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[GripperController] Controller process spawned at {self.pid}")

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
        
    # ========= command methods ============
        

    def schedule_waypoint(self, pose, target_time):
        #assert target_time > time.time()
        #assert isinstance(pose, bool)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
        #print(f"instruction: {pose}")
    
    # def goto(self, pose):
    #     #assert isinstance(pose, Union[np.bool_,bool])
        
    #     message = {
    #         'cmd': Command.GOTO.value,
    #         'target_pose': pose,
    #         'target_time': -1.0
    #     }
        
    #     self.input_queue.put(message)

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

        # start rtde
        gripper_ip = self.gripper_ip
        
        gripper = Gripper(gripper_ip=gripper_ip)
        time.sleep(1)
        gripper.activate()
        try:
            if self.verbose:
                print(f"[GripperController] Connect to gripper: {gripper_ip}")

            #init pose
            gripper.gotobool(self.gripper_init_pos) 

            # main loop
            dt = 1. / self.frequency
            init_pose = self.gripper_init_pos
            # use monotonic time to make sure the control loop never go backward
            t_start = time.monotonic()
            pose_interp = BoolStateInterpolator(
                times=[t_start],
                poses=[init_pose]
            )
            #print(curr_pose)
            #print(t_start)
            iter_idx = 0
            keep_running = True
            prev_command = init_pose

            while keep_running:
                # start control iteration
                # send command to robot
                t_now = time.monotonic()
                pose_command = pose_interp(t_now)[0]
                gripper.gotobool(pose_command)

                t_next_loop = (t_start + (iter_idx+1)*dt)

                #prev_command = pose_command #TODO use this somehow?
                # update robot state
                state = gripper.get_state()
                if self.verbose:
                    print(state)
                #for key in self.receive_keys:
                #    state[key] = np.array(getattr(rtde_r, 'get'+key)())
                state['gripper_receive_timestamp'] = time.time()
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

                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        #print(curr_time, target_time, target_pose)
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            action_time=target_time,
                            curr_time=curr_time
                            #last_waypoint_time=last_waypoint_time
                        )

                        #last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                #rtde_c.waitPeriod(t_start)
                    
                precise_wait(t_next_loop)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    pass
                    #print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # mandatory cleanup
            # TODO do goto to storage position
            # terminate
            gripper.close()
            self.ready_event.set()

            if self.verbose:
                print(f"[GripperController] Disconnected from gripper: {gripper_ip}")

def main():
    pass

if __name__ == "__main__":
    main()
