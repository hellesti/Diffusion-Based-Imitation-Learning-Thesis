

import serial.tools.list_ports
import serial
import numpy as np

import time

def get_suctioncup_port():
    usb_ports = []
    ports=serial.tools.list_ports.comports()
    for port in ports:
        if "USB" in port.device:
            usb_ports.append(port.device)

    assert len(usb_ports) == 1 #For the setup in the NTNU lab you may need to get patched drivers from https://github.com/juliagoda/CH341SER which correctly identify the module
    return usb_ports[0]




# value=0
# while True:
#     cont = input("press ENTER:")
#     value= 1-value
#     num=str(value)
#     write(num)


class Suctioncup:
    def __init__(self, ttyusb="auto", initpos=False):
        if ttyusb == "auto":
            ttyusb = get_suctioncup_port()
        self.arduino = serial.Serial(port=ttyusb, baudrate = 115200, timeout=3)
        self._running = True
        self.suctioncup_state = {}
        self.suctioncup_state["suctioncup_state"] = [initpos]

    def write(self,x):
        if x:
            x = 1
        else:
            x = 0
        ret = self.arduino.write(bytes(str(x),'utf-8'))
        self.suctioncup_state["suctioncup_state"] = [x]
        print(x, ret)


    def get_state(self):
        return self.suctioncup_state

    def gotobool(self, pose=None):
        if pose is not None:
            assert ((pose == 1) or (pose == 0))
            """Send a command to the suction cup."""
            self.write(pose)
            return True
        
    def close(self):
        self.arduino.close()

def main():
    pass
        
if __name__ == "__main__":
   main()