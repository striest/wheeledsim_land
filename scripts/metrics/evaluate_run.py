import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import rosbag

from common_msgs.msg import BoolStamped
from nav_msgs.msg import Odometry

"""
General evaluation script for intervention learning
Important metrics:
    1. Interventions/km of driving
    2. Interventions/minute of driving
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Dir containing the bags for the run')
    parser.add_argument('--odom_topic', type=str, required=False, default='/odometry/filtered_odom', help='Topic to read odom from')
    parser.add_argument('--intervention_topic', type=str, required=False, default='/mux/intervention', help='Topic to read intervention from')
    parser.add_argument('--distance_window', type=float, required=False, default=100., help='Sliding window to analyze interventions/km over (in m)')
    parser.add_argument('--time_window', type=float, required=False, default=10., help='Sliding window to analyze interventions/minute over (in s)')
    parser.add_argument('--start_time', type=float, required=False, default=0., help='Drop all data before this many seconds into the bag')
    parser.add_argument('--end_time', type=float, required=False, default=float('inf'), help='Drop all data after this many seconds')
    parser.add_argument('--title', type=str, required=False, default='', help='Plot title')
    args = parser.parse_args()

    bags = [x for x in os.listdir(args.run_dir) if x[-4:] == '.bag']

    interventions = [] # [t x 2] arr of timestamp, intervention
    positions = [] # [t x 4] arr of timestamp, x, y, intervention

    for bidx, bag_fp in enumerate(sorted(bags)):
        print('{} ({}/{})'.format(bag_fp, bidx+1, len(bags)), end='\r')
        bag = rosbag.Bag(os.path.join(args.run_dir, bag_fp), 'r')
        for topic, msg, t in bag.read_messages(topics=[args.odom_topic, args.intervention_topic]):
            if topic == args.odom_topic:
                time = msg.header.stamp.to_sec()
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                data = np.array([time, x, y, 0.])
                positions.append(data)
            elif topic == args.intervention_topic:
                time = msg.header.stamp.to_sec()
                intervention = msg.data
                data = np.array([time, intervention])
                interventions.append(data)

    interventions = np.stack(interventions, axis=0)
    positions = np.stack(positions, axis=0)

    #Start at 0, 0, 0
    positions[:, 1] -= positions[0, 1]
    positions[:, 2] -= positions[0, 2]
    positions[:, 0] -= interventions[0, 0]
    interventions[:, 0] -= interventions[0, 0]

    positions = positions[(positions[:, 0] >= args.start_time) & (positions[:, 0] <= args.end_time)]
    interventions = interventions[(interventions[:, 0] >= args.start_time) & (interventions[:, 0] <= args.end_time)]

    total_distance = np.linalg.norm(positions[1:, [1, 2]] - positions[:-1, [1, 2]], axis=1).sum()

    #Is intervention when the signal switches from 0 to 1
    intervention_mask = (interventions[1:, 1] - interventions[:-1, 1]) > 1e-2
    intervention_timestamps = interventions[:-1, 0][intervention_mask]
    intervention_positions = []
    #Do this the iterative way to save space
    for ts in intervention_timestamps:
        idx = np.argmin(abs(positions[:, 0] - ts))
        pos = positions[idx]
        intervention_positions.append(pos)
        positions[idx, 3] = 1.

    intervention_positions = np.stack(intervention_positions, axis=0)

    #Create figures
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    plt.suptitle(args.title)

    #Plot positions
    axs[0].plot(positions[:, 1], positions[:, 2], c='b', label='path (total {:.2f}km)'.format(0.001 * total_distance))
    axs[0].scatter(positions[0, 1], positions[0, 2], c='y', marker='^', label='start')
    axs[0].scatter(positions[-1, 1], positions[-1, 2], c='y', marker='o', label='end')
    axs[0].scatter(intervention_positions[:, 1], intervention_positions[:, 2], c='r', marker='x', label='intervention ({} total)'.format(intervention_positions.shape[0]))
    for i, pos in enumerate(intervention_positions):
        axs[0].text(pos[1], pos[2], i+1)
    axs[0].legend()
    axs[0].set_xlabel('X(m)')
    axs[0].set_ylabel('Y(m)')
    axs[0].set_title('Path')

    #Plot interventions/time
    axs[1].scatter(intervention_timestamps, np.ones_like(intervention_timestamps), c='r', marker='x', label='Interventions')
    axs[1].legend()
    axs[1].set_xlabel('T(s)')
    axs[1].set_ylabel('Interventions')
    axs[1].set_title('Interventions vs. Time')

    if intervention_positions.shape[0] == 0:
        print('no interventions')
        exit()

    if total_distance > args.distance_window:
        #Plot interventions/km
        interventions_km = []
        position_buffer = np.array([positions[0]])
        flag = False
        cumulative_distance = 0.
        for pos in positions[1:]:
            pos_old = position_buffer[-1]
            ds = np.linalg.norm(pos[[1, 2]] - pos_old[[1, 2]]) 
            cumulative_distance += ds
            position_buffer = np.concatenate([position_buffer, np.expand_dims(pos, axis=0)], axis=0)

            while cumulative_distance > args.distance_window:
                flag = True
                ds2 = np.linalg.norm(position_buffer[0][[1, 2]] - position_buffer[1][[1, 2]])
                cumulative_distance -= ds2
                position_buffer = position_buffer[1:]

            if flag:
                interventions_km.append(position_buffer[:, 3].sum() / (0.001 * args.distance_window))

        axs[2].plot(positions[-len(interventions_km):, 0], interventions_km)
    else:
        axs[2].text(0, 0, 'Distance too short to compute stats')

    axs[2].set_ylabel('Interventions/km (Window {:.2f}km)'.format(0.001 * args.distance_window))
    axs[2].set_xlabel('Time (s)')
    axs[2].set_title('Interventions/km')

    #Plot interventions/min
    if interventions[-1, 0] > args.time_window:
        interventions_min = []
        position_buffer = np.array([positions[0]])
        flag = False
        cumulative_time = 0.
        for pos in positions[1:]:
            pos_old = position_buffer[-1]
            dt = pos[0] - pos_old[0]
            cumulative_time += dt
            position_buffer = np.concatenate([position_buffer, np.expand_dims(pos, axis=0)], axis=0)

            while cumulative_time > args.time_window:
                flag = True
                dt2 = position_buffer[1][0] - position_buffer[0][0]
                cumulative_time -= dt2
                position_buffer = position_buffer[1:]

            if flag:
                interventions_min.append(position_buffer[:, 3].sum() / ((1./60.) * args.time_window))

        axs[3].plot(positions[-len(interventions_min):, 0], interventions_min)
    else:
        axs[3].text(0, 0, 'Time too short to compute stats')
    axs[3].set_ylabel('Interventions/min (Window {:.2f}min)'.format((1./60.) * args.time_window))
    axs[3].set_xlabel('Time (s)')
    axs[3].set_title('Interventions/min')
    plt.show() 
