import rospy
import numpy as np
import copy

from std_msgs.msg import Float64, Float64MultiArray, ColorRGBA
from common_msgs.msg import BoolStamped, Int32Stamped
from geometry_msgs.msg import Vector3, Point, Pose, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

MARKERS_PER_ACTION = 100 #Just an upper bound on markers/action to avoid namespacing collisions

class EilVisualization:
    """
    Class that handles publishing rviz visualizations from EIL
    """
    def __init__(self, action_library, action_topic, q_values_topic, intervention_topic, base_frame):
        """
        Args:
            action_library: The library that EIL is learning over
            action_topic: The topic that EIL publishes the discrete action to
            q_values_topic: The topic that EIL publishes q values to
            intervention_topic: The topic that interventions are published on
            base_frame: The base frame of the robot (the frame to tie the visuals to)
        """
        self.action_library = action_library
        self.n_acts = len(action_library['library'])

        self.action_sub = rospy.Subscriber(action_topic, Int32Stamped, self.handle_action)
        self.q_sub = rospy.Subscriber(q_values_topic, Float64MultiArray, self.handle_q_values)
        self.intervention_sub = rospy.Subscriber(intervention_topic, BoolStamped, self.handle_intervention)
        self.base_frame = base_frame

        self.viz_pub = rospy.Publisher("/eil/viz", MarkerArray, queue_size=10)

        self.current_action = None
        self.current_q_values = None
        self.current_intervention = None

        self.rate = rospy.Rate(10)

        #Assign namespace ids via the following: act_idx*MARKERS_PER_ACTION + marker_type (marker_type defined below)
        self.namespace_idxs = {
            'path':0,
            'act_label':1,
            'q_value':2,
            'current_action':MARKERS_PER_ACTION * self.n_acts + 1
        }

        self.actlib_msg = self.get_action_library_msg()

    def get_action_library_msg(self):
        """
        Publish the action library to the robot (once)
        """
        path_msgs = MarkerArray()
        act_msgs = MarkerArray()
        qval_msgs = MarkerArray()
        for ai, action in enumerate(self.action_library['library']):
            positions = np.stack([action['viz']['x'], action['viz']['y'], action['viz']['z']], axis=-1)
            base_msg = Marker()
            base_msg.header.stamp = rospy.Time.now()
            base_msg.header.frame_id = self.base_frame
            base_msg.ns = "eil"
            base_msg.action = 0
            base_msg.frame_locked = True

            act_msg = copy.deepcopy(base_msg)
            act_msg.type = Marker.LINE_STRIP
            act_msg.id = MARKERS_PER_ACTION * ai + self.namespace_idxs['path']
            act_msg.color = ColorRGBA(r=0., g=1., b=0., a=1.)
            act_msg.scale = Vector3(x=0.05, y=0.05, z=0.05)
            act_msg.pose = Pose(orientation=Quaternion(w=1.0))
            act_msg.lifetime = rospy.Duration(0.)
            for pos in positions:
                x, y, z = pos
                act_msg.points.append(
                    Point(x=x, y=y, z=z)
                )

            label_msg = copy.deepcopy(base_msg)
            label_msg.type = Marker.TEXT_VIEW_FACING
            label_msg.id = MARKERS_PER_ACTION * ai + self.namespace_idxs['act_label']
            label_msg.color = ColorRGBA(r=1., g=1., b=1., a=1.)
            label_msg.scale = Vector3(x=0.5, y=0.5, z=0.5)
            label_msg.lifetime = rospy.Duration(0.)
            mid_pos = positions[int(positions.shape[0]/2)]
            label_msg.pose = Pose(position=Point(x=mid_pos[0], y=mid_pos[1], z=mid_pos[2]), orientation=Quaternion(w=1.))
            label_msg.text = "a{}".format(ai)

            q_value_msg = copy.deepcopy(base_msg)
            q_value_msg.type = Marker.TEXT_VIEW_FACING
            q_value_msg.id = MARKERS_PER_ACTION * ai + self.namespace_idxs['q_value']
            q_value_msg.color = ColorRGBA(r=1., g=1., b=1., a=1.)
            q_value_msg.scale = Vector3(x=0.5, y=0.5, z=0.5)
            q_value_msg.lifetime = rospy.Duration(0.)
            mid_pos = positions[-1]
            q_value_msg.pose = Pose(position=Point(x=mid_pos[0], y=mid_pos[1], z=mid_pos[2]), orientation=Quaternion(w=1.))
            q_value_msg.text = "{:.2f}".format(0.0)

            path_msgs.markers.append(act_msg)
            act_msgs.markers.append(label_msg)
            qval_msgs.markers.append(q_value_msg)

        current_action_msg = copy.deepcopy(base_msg)
        current_action_msg.type = Marker.CUBE
        current_action_msg.id = self.namespace_idxs['current_action']
        current_action_msg.color = ColorRGBA(r=0., g=0., b=0., a=1.)
        current_action_msg.scale = Vector3(x=0.25, y=0.25, z=0.25)
        current_action_msg.lifetime = rospy.Duration(0.)
        current_action_msg.pose = Pose(orientation=Quaternion(w=1.0))
            
        return {
            'path':path_msgs,
            'act_label':act_msgs,
            'q_value':qval_msgs,
            'current_action':MarkerArray(markers=[current_action_msg])
        }

    def update_actlib_msg(self):
        """
        Update the actlib message according to current data
            1. Relabel the q values and color the axes accordingly
            2. Put a marker on the current action. Color it differently if it is/isnt an intervention
            3. Restamp everything
        """
        #relabel q
        if self.current_q_values is not None:
            qmin = self.current_q_values.min() - 0.5
            qmax = self.current_q_values.max() + 0.5
            for i, val in enumerate(self.current_q_values):
                nval = (val - qmin) / (qmax - qmin)
                self.actlib_msg['q_value'].markers[i].text = "{:.2f}".format(val)
                self.actlib_msg['path'].markers[i].color = ColorRGBA(r=nval, g=0., b=1.-nval, a=1.)

        #Current action marker
        if self.current_action is not None and self.current_intervention is not None:
            x = self.action_library['library'][self.current_action]['viz']['x'][-1]
            y = self.action_library['library'][self.current_action]['viz']['y'][-1]
            z = self.action_library['library'][self.current_action]['viz']['z'][-1]
            self.actlib_msg['current_action'].markers[0].pose = Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(w=1.0))
            self.actlib_msg['current_action'].markers[0].color = ColorRGBA(r=1., g=0., b=0., a=1.) if self.current_intervention else ColorRGBA(r=0., g=0., b=1., a=1.)

        #restamp
        for k in self.actlib_msg.keys():
            for i in range(len(self.actlib_msg[k].markers)):
                self.actlib_msg[k].markers[i].header.stamp = rospy.Time.now()

    def handle_action(self, msg):
        self.current_action = msg.data

    def handle_q_values(self, msg):
        self.current_q_values = np.array(msg.data)

    def handle_intervention(self, msg):
        self.current_intervention = msg.data

    def spin(self):
        while not rospy.is_shutdown():
            self.update_actlib_msg()
            for msg in self.actlib_msg.values():
                self.viz_pub.publish(msg)
            self.rate.sleep()
