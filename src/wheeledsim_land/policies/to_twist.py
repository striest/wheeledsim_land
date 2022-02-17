import rospy

from geometry_msgs.msg import Twist

class ToTwist:
    """
    Wrapper from the standard torch interface to twist messages
    """
    def __init__(self, policy, tscale=1.0, sscale=1.0):
        self.policy = policy
        self.tscale = tscale
        self.sscale = sscale

        self.device = self.policy.device

    def action(self, obs=None, return_info=False):
        if return_info:
            act, info = self.policy.action(obs, return_info=True)
            return self.act_to_twist(act), info
        else:
            act = self.policy.action(obs, return_info=False)
            return self.act_to_twist(act)

    def act_to_twist(self, act):
        msg = Twist()
        msg.linear.x = act[0].cpu().item()
        msg.angular.z = act[1].cpu().item()
        return msg

    def to(self, device):
        self.policy = self.policy.to(device)
        self.device = device
        return self
