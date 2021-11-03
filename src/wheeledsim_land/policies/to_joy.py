import rospy

from sensor_msgs.msg import Joy

class ToJoy:
    """
    Wrapper from the standard torch interface to joy messages
    """
    def __init__(self, policy, taxis=1, saxis=3):
        self.policy = policy
        self.taxis = taxis
        self.saxis = saxis
        self.device = self.policy.device

    def action(self, obs=None):
        act = self.policy.action(obs)
        msg = Joy()
        msg.header.stamp = rospy.Time.now()
        msg.buttons = [0] * 12
        msg.axes = [0.] * 6
        msg.axes[self.taxis] = act[0].cpu().item()
        msg.axes[self.saxis] = act[1].cpu().item()
        return msg

    def to(self, device):
        self.policy = self.policy.to(device)
        self.device = device
        return self
