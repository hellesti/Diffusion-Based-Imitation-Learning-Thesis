#Code modified by Niilo Emil Isosomppi from https://github.com/baha2r/robotiq3f_py which is by Bahadur Beigomi

#See link below for gripper documentation. Manual is also archived in the project files. Robotiq UI (Windows) for testing the gripper and configuring its IP is also archived in the apppended project files.
#Gripper: https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf
#RUI : 

from pyModbusTCP.client import ModbusClient
import time
import warnings
import numpy as np                                                          
#import keyboard

class Gripper:
    def __init__(self, gripper_ip):
        self.client = ModbusClient(host=gripper_ip, auto_open=True)
        self.client.open()
        self._running = True
        self.gripper_state = {}

    def _update_state(self):
        """Update and store the gripper status from Modbus server."""
        readData = self.client.read_input_registers(0,3)
        if readData is None:
            print("Tried reading gripper output registers unsuccessfully.")
        else:
            gOBJ, gSTA, gGTO, gACT = self.stat(self.add_leading_zeros(bin(readData[0])))
            FaultStatus, gPR = self.Byte_status(self.add_leading_zeros(bin(readData[1])))
            gPO, gCU = self.Byte_status(self.add_leading_zeros(bin(readData[2])))

            self.gOBJ, self.gSTA, self.gGTO, self.gACT = gOBJ, gSTA, gGTO, gACT
            self.FaultStatus, self.gripper_state["gPR"] = FaultStatus, np.array([gPR])
            self.gripper_state["gPO"], self.gripper_state["gCU"] = np.array([gPO]), np.array([gCU])

            if (self.gOBJ in [1,2]):
                self.gripper_state["object_is_grabbed"] = np.array([True])
            else:
                self.gripper_state["object_is_grabbed"] = np.array([False])

    def get_state(self):
        self._update_state()
        return self.gripper_state

    @staticmethod
    def add_leading_zeros(bin_num, total_length=16):
        """Ensure binary number string has correct number of digits."""
        bin_str = str(bin_num)[2:]
        return bin_str.zfill(total_length)

    @staticmethod
    def Byte_status(variable: int) -> str:
        """Split and parse byte status."""
        B1 = int(variable[0:8],2)
        B2 = int(variable[8:16],2)
        return B1, B2

    @staticmethod
    def stat(variable: int) -> str:
        """Split and parse status."""
        # Define gripper modes

        # Split and parse status bits
        gOBJ = int(variable[0:2],2)
        gSTA = int(variable[2:4],2)
        gGTO = int(variable[4],2)
        gACT = int(variable[7],2)

        return gOBJ, gSTA, gGTO, gACT
    
    def wait_until_connected(self):
        while True:
            readData = self.client.read_input_registers(0,3)
            if readData is None:
                print("Waiting for gripper connection..")
                time.sleep(1)
            else:
                pass
        print("Connected to gripper.")
        return True
    
    def activate(self):
        """Activate the gripper."""
        response = self.client.write_multiple_registers(0, [self._action_req_variable(request_ACTIVATE=1), 0, 0])
        print("Gripper activating")
        time.sleep(1)
        
    def goto(self, request_POSITION=50, request_SPEED=255, request_FORCE=255):
        """Send a command to the gripper."""
        response = self.client.write_multiple_registers(
            0,
            [self._action_req_variable(request_ACTIVATE=1, request_GOTO=1),
            self._position_req_variable(request_POSITION),
            self._write_req_variable(request_SPEED, request_FORCE)]
        )



    def gotobool(self, position=None, request_SPEED=255, request_FORCE=255, wait_timeout=None): 
        """Send a command to the gripper."""
        if wait_timeout is not None:
            timeout_monotonic= time.monotonic()+wait_timeout

        def write():
            response = self.client.write_multiple_registers(
                0,
                [self._action_req_variable(request_ACTIVATE=1, request_GOTO=1),
                self._position_req_variable(request_POSITION),
                self._write_req_variable(request_SPEED, request_FORCE)]
            )
            return response
        
        #print(position)
        if position is not None:
            if position == True:
                request_POSITION = 255
            else:
                request_POSITION = 0
            response=False
            if wait_timeout is not None:
                while(not response and time.monotonic()<timeout_monotonic):
                    response = write()
                    time.sleep(1/30)
            else:
                response = write()
        else:
            response = None
        return response
    
    def wait_for_grab_success(self,timeout=3):
        t_start = time.time()
        while True: #time.time()<(t_start+timeout):
            gripper_state=self.get_state()
            if gripper_state["object_is_grabbed"]:
                return True
            time.sleep(0.1)
        return False

            
    # def status(self):
    #     readData = self.client.read_input_registers(0,3)
    #     self.gOBJ, self.selfgSTA, self.gGTO, self.gACT = self.stat(self.add_leading_zeros(bin(readData[0])))
    #     self.FaultStatus, self.gripper_state["gPR"] = self.Byte_status(self.add_leading_zeros(bin(readData[1])))
    #     self.gripper_state["gPO"], self.gripper_state["gCU"] = self.Byte_status(self.add_leading_zeros(bin(readData[2])))

    #     if (self.gOBJ in [1,2]):
    #         self.gripper_state["object_is_grabbed"] = True
    #     else:
    #         self.gripper_state["object_is_grabbed"] = False


    def _action_req_variable(self, rARD: int = 0, rATR: int = 0, request_GOTO: int = 0, request_ACTIVATE: int = 0) -> str:
        """Build action request variable."""
        # Check if the input variables are either 0 or 1
        for var in [rARD, rATR, request_GOTO, request_ACTIVATE]:
            if var not in [0, 1]:
                raise ValueError("Input variables must be either 0 or 1.")
        # Construct the string variable
        string_variable = f"0b00{rARD}{rATR}{request_GOTO}00{request_ACTIVATE}00000000" 
        
        return int(string_variable,2)
    
    def _position_req_variable(self, POSITION_REQUEST: int = 0) -> str:
        """Build position request variable."""
        # Check if the input variables are between 0 or 255
        for var in [POSITION_REQUEST]:
            if var not in range(0,256):
                raise ValueError("Input variables must be between 0 and 255.")
        POSITION_REQUEST = format(POSITION_REQUEST, '08b')

        # Construct the string variable
        string_variable = f"0b00000000{POSITION_REQUEST}"
        
        return int(string_variable,2) 
    
    def _write_req_variable(self, X: int = 0, Y: int = 0) -> str:
        """Build write request variable."""
        # Check if the input variables are between 0 to 255
        for var in [X, Y]:
            if var not in range(0,256):
                raise ValueError("Input variables must be between 0 and 255.")
        X = format(X, '08b')
        Y = format(Y, '08b')
        # Construct the string variable
        string_variable = f"0b{X}{Y}"
        
        return int(string_variable,2)
    
    def close(self):
        """Stop the update thread and close the Modbus client."""
        response = self.client.write_multiple_registers(0, [self._action_req_variable(request_GOTO = 0, request_ACTIVATE=0), 0, 0])
        self._running = False
        self.client.close()
        print("Connection closed.")

def main():
    #import keyboard
    """Main function."""
    # Create and activate gripper controller
    gripper = Gripper("192.168.1.12")
    #assert gripper.wait_until_connected()
    #TODO assert connected
    gripper.activate()
    #final_position = [210,10,210,10,210,100]
    speed = 100
    force = 0
    grab = False
    prev = -1
    while True:
        # if keyboard.is_pressed('a'):
        #     grab = False
        # elif keyboard.is_pressed('s'):
        #     grab = True
        # elif keyboard.is_pressed('esc'):
        #     break
        if grab:
            gripper.goto(255, request_FORCE=force, request_SPEED = speed)
        else:
            gripper.goto(0, request_FORCE=force, request_SPEED = speed)
        
        obj = gripper.gOBJ
        if (obj != prev):
            if (obj == 0):
                print("Moving to position")
            elif (obj == 1):
                print("Object detected while opening")
            elif (obj == 2):
                print("Object detected while closing")
            elif (obj == 3):
                print("Reference value reached without object detection")
        prev = obj
        time.sleep(0.1)
        
    
    # for i in range(len(final_position)):
    #     # Send command to gripper and wait for it to reach final position
    #     gripper.command_gripper(final_position[i], speed, force)
    #     while (gripper.Finger_Position != final_position[i]):
    #         pass #print(f"FingerA_Position: {gripper.FingerA_Position}")
    #     print(f"Finger_Position: {gripper.Finger_Position}")
    #     time.sleep(1)
        
    # Close the controller when done

if __name__ == "__main__":
   main()