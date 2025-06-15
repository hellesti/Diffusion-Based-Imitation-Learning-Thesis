print_ee_pose           Prints copyable list containing the position of the current TCP relative to base coordinate system
print_ee_joints         Prints copyable list containing the joint angles of the current position (indifferent to TCP)
activate_tool           Grab with gripper or activate vacuum in suction cup (depending on real_env)
deactivate_tool         Release gripper or release suction cup vacuum (depending on real_env)
do_t_reset              Resets T from end position to valid starting position. Only valid if real_env == pusht
#TODO finish writing this