import rosbag
import csv

# Open bag
bag = rosbag.Bag('rosbag_files/Good/1_Gui.bag')

# Open CSV for writing
with open('franka_torque_and_gripper_log.csv', 'w', newline='') as csvfile:

    writer = csv.writer(csvfile)
    # Write header
    writer.writerow([

        'time',
        'tau_J_0', 'tau_J_1', 'tau_J_2', 'tau_J_3', 'tau_J_4', 'tau_J_5', 'tau_J_6',
        'gripper_position_left', 'gripper_position_right'

    ])

    # Initialize latest gripper states
    latest_gripper_position = [None, None]

    for topic, msg, t in bag.read_messages(topics=[

        '/franka_state_controller/franka_states',
        '/franka_gripper/joint_states'
    ]):

        timestamp = t.to_sec()

        if topic == '/franka_gripper/joint_states':

            print(msg)
            latest_gripper_position = list(msg.position) if msg.position else [None, None]
            #latest_gripper_effort = list(msg.effort) if msg.effort else [None, None]

        elif topic == '/franka_state_controller/franka_states':

            tau_J = list(msg.tau_J)
            row = [timestamp] + tau_J + latest_gripper_position #+ latest_gripper_effort
            writer.writerow(row)

bag.close()
