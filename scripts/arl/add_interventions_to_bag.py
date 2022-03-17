import rospy
import rosbag
import argparse
import numpy as np

from common_msgs.msg import BoolStamped
from geometry_msgs.msg import Twist

"""
Simple script that adds an intervention topic to ARL rosbags
This is accomplished by getting the timestamps of the teleop messages and populating true/false with them

ALSO NOTE THAT I WILL MODIFY THE BAG DIRECTLY
"""

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_fp', type=str, required=True, help='Path to the bag')
    parser.add_argument('--teleop_topic', type=str, required=False, default='/warthog2/rc_teleop/cmd_vel')
    parser.add_argument('--intervention_topic', type=str, required=False, default='/mux/intervention')
    parser.add_argument('--rate', type=int, required=False, default=10, help='rate to publish interventions at')
    args = parser.parse_args()
    
    teleop_times = []
    teleop_timestamps = []

    bag = rosbag.Bag(args.bag_fp, 'a')

    start_time = bag.get_start_time()
    end_time = bag.get_end_time()
    dt = 1/args.rate

    times = np.arange(start_time, end_time, dt)

    for topic, msg, t in bag.read_messages(topics=[args.teleop_topic]):
        teleop_timestamps.append(t)
        teleop_times.append(t.to_sec())

    for i, time in enumerate(times):
        msg_out = BoolStamped()
        msg_out.header.stamp = rospy.Time.from_sec(time)
        msg_out.header.seq = i
        is_intervention = min(abs(teleop_times - time)) < dt
        msg_out.data = is_intervention
        bag.write(args.intervention_topic, msg_out, msg_out.header.stamp)
    
    bag.close()

    print('done')
