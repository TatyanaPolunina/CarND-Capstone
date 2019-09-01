#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree;
from std_msgs.msg import Int32

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish.
MAX_DECEL = 0.5

def get_waypoint_velocity(waypoint):
    return waypoint.twist.twist.linear.x

def set_waypoint_velocity(waypoint, velocity):
    waypoint.twist.twist.linear.x = velocity

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None;
        self.base_waypoints = None;
        self.waypoints_2d = None;
        self.waypoint_tree = None;
        self.stop_line_wp = -1;

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():  
            if self.pose and self.base_waypoints:
                closest_waypoint_index = self.get_closestwaypoint_index();
                self.publish_waypoints(closest_waypoint_index);              
            rate.sleep()
    
    def get_closestwaypoint_index(self):
        pos = [self.pose.pose.position.x, self.pose.pose.position.y];
        closest_idx = self.waypoint_tree.query(pos, 1)[1]
        
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[(closest_idx - 1) % len(self.waypoints_2d)]
        cl_vec = np.array(closest_coord)
        prev_vec = np.array(prev_coord)
        curren_pos = np.array(pos)
        val = np.dot(cl_vec - prev_vec, curren_pos - cl_vec);
        if (val > 0):
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def pose_cb(self, msg):
        self.pose = msg;
        
    def publish_waypoints(self, closest_idx):
        lane = self.get_lane()
        self.final_waypoints_pub.publish(lane)    
        
    def get_lane(self):
        lane = Lane()
        closest_wp_index = self.get_closestwaypoint_index()
        last_index = closest_wp_index + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_wp_index: last_index]
        if (self.stop_line_wp == -1 or self.stop_line_wp > last_index):
            lane.waypoints = base_waypoints
        else:
            dist_to_light = self.stop_line_wp - closest_wp_index
            lane.waypoints = self.decelerate(base_waypoints, dist_to_light)
        return lane
    
    def decelerate(self, waypoints, dist_to_light):
        lane_wps = []
        dist_to_stop = max(0, dist_to_light - 2)
        for i in range(len(waypoints)):
            wp = Waypoint()
            wp.pose = waypoints[i].pose
            dist = self.distance(waypoints, i, dist_to_stop)
            dec_vel = math.sqrt(2 * MAX_DECEL * dist)
            if dec_vel < 1:
                dec_vel = 0.0
            set_waypoint_velocity(wp, min(dec_vel, get_waypoint_velocity(wp)))
            lane_wps.append(wp)
        return lane_wps
               
        

    def waypoints_cb(self, waypoints):
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        self.base_waypoints = waypoints

    def traffic_cb(self, msg):
       if (msg.data != self.stop_line_wp):
            self.stop_line_wp = msg.data
            rospy.loginfo("new stop line recieved {0}".format(self.stop_line_wp))


    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
