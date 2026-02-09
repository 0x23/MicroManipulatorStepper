from open_micro_stage import OpenMicroStageInterface

# create interface and connect
oms = OpenMicroStageInterface(show_communication=True, show_log_messages=True)
oms.connect("/dev/ttyACM0")

# run this once to calibrate joints
# for i in range(3): oms.calibrate_joint(i, save_result=True)

# home device
oms.home()

# move and wait
oms.move_to(0, 0, 0, f=10)
oms.wait_for_stop()

# print some info
oms.read_device_state_info()
