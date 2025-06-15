import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.real_world.suctioncup_usb import Suctioncup
import time

"""Main function."""
# Create and activate suctioncup controller
suctioncup = Suctioncup(ttyusb="auto")
#assert suctioncup.wait_until_connected()
grab = suctioncup.get_state()["suctioncup_state"]
while True:
    # if keyboard.is_pressed('a'):
    #     grab = False
    # elif keyboard.is_pressed('s'):
    #     grab = True
    # elif keyboard.is_pressed('esc'):
    #     break
    input("Press ENTER:")
    grab=1-grab

    suctioncup.gotobool(grab)

    print(suctioncup.get_state()["suctioncup_state"])
    time.sleep(0.1)
    