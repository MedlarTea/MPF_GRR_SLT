#! /usr/bin/env python3
import numpy as np
import rospy
import math

# Standard message
from geometry_msgs.msg import *
from sensor_msgs.msg import Joy

# Our message
from mono_following.msg import *

# Our class
# from pid import PID_controller

class MonoController:
    def __init__(self):
        ### hyper parameters ###
        self.STOP = False
        # self.rx = 0.
        # self.ry = 0.

        ### PID ### 
        # Kp = 0.6
        # Kd = 0.1
        # Ki = 0.2
        # th_pid = PID_controller(Kp, Ki, Kd, deadband = 0., u_min = -1., u_max = 1., e_int_min = -0.2, e_int_max = 0.2, dt = dt)

        # Kp = 0.3
        # Kd = 0.0
        # Ki = 0.1
        # xy_pid     = PID_controller(Kp, Ki, Kd, deadband = 0., u_min = -1., u_max = 1, e_int_min = -0.1, e_int_max = 0.1, dt = dt)
        
        self.enable_back = rospy.get_param('~enable_back', True)
        self.max_vx = rospy.get_param('~max_vx', 0.2)
        self.max_va = rospy.get_param('~max_va', 0.5)
        self.gain_vx = rospy.get_param('~gain_vx', 0.3)
        self.gain_va = rospy.get_param('~gain_va', 0.3)
        self.distance = rospy.get_param('~distance', 1)
        self.timeout = rospy.get_param('~timeout', 1.0)

        self.last_time = rospy.Time(0)
        self.target_id = -1

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.target_sub = rospy.Subscriber('/mono_following/target', Target, self.target_callback)
        # joystick callback
        self.joystick = rospy.Subscriber('/bluetooth_teleop/joy', Joy, self.joystick_callback)
        # self.joystick = rospy.Subscriber('/joy', Joy, self.joystick_callback)

        ### Node is already set up ###
        rospy.loginfo("Mono Controlling Node is Ready!")
    

    def target_callback(self, target_msg):
        """
        - Our controlling strategy: keep certain `pos_x` distance and decrease `pos_y` to 0 to ensure our distance estimation is accurate enough
        - Our robot is Omni-directional
        """
        if self.STOP == True:
            self.cmd_vel_pub.publish(Twist())
            print("JOY STOP")
            return

        target_id = target_msg.target_id
        if len(target_msg.position) == 0:
            self.cmd_vel_pub.publish(Twist())
            return
        px, py = target_msg.position  # [x, y]
        
        theta = math.atan2(py, px)
        print('x, y, theta: {:.3f}, {:.3f}, {:.3f}'.format(px, py, theta))

        va = min(self.max_va, max(-self.max_va, theta * self.gain_va))
        vx = 0.0

        if abs(theta) < math.radians(45):
            vx = (px - self.distance) * self.gain_vx
            print('vx, va: {:.3f}, {:.3f}'.format(vx, va))
            min_vx = -self.max_vx if self.enable_back else 0.0
            vx = min(self.max_vx, max(min_vx, vx))
        else:
            print('rotation too big')
        
        twist = Twist()
        twist.linear.x = vx
        twist.angular.z = va

        self.cmd_vel_pub.publish(twist)
        self.last_time = target_msg.header.stamp
        
    def spin(self):
        if (rospy.Time.now() - self.last_time).to_sec() > self.timeout:
            # print(self.last_time)
            # print((rospy.Time.now() - self.last_time).to_sec())
            print('timeout!!')
            self.cmd_vel_pub.publish(Twist())

    # For safety
    def joystick_callback(self, joy_msg):
        button_states = joy_msg.buttons # (cha, circle, triangle, square)--(buttons 0,1,2,3)
        print(button_states)
        if button_states[0] == 1:
            self.STOP = True
        if button_states[1] == 1:
            self.STOP = False



def main():
    rospy.init_node('mono_controller_node')
    node = MonoController()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        node.spin()
        rate.sleep()

if __name__ == '__main__':
	main()
