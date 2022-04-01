import rospy

from common_msgs.msg import Int32Stamped

class ToInt32:
    """
    Wrapper from the standard torch interface to int32 messages
    """
    def __init__(self, policy):
        self.policy = policy
        self.device = self.policy.device

    def action(self, obs=None, return_info=False):
        if return_info:
            act, info = self.policy.action(obs, return_info=True)
            return self.act_to_int(act), info
        else:
            act = self.policy.action(obs, return_info=False)
            return self.act_to_int(act)

    def act_to_int(self, act):
        msg = Int32Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = act.long().item()
        return msg

    def to(self, device):
        self.policy = self.policy.to(device)
        self.device = device
        return self
