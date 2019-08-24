from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self,   vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                 accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle);
        
        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0
        mx = 0.2
        self.pid_controller = PID(kp, ki, kd, mn, mx);
        self.lpf = LowPassFilter(0.5, 0.02)
        self.last_time = rospy.get_time();
        self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        
        
    def control(self, dbw_enabled, current_vel, linear_vel, angular_vel):
        if not dbw_enabled:
            self.pid_controller.reset();
            return 0., 0., 0
            
        current_vel = self.lpf.filt(current_vel)
        
        cur_time = rospy.get_time()
        time_diff = cur_time - self.last_time
        self.last_time = cur_time
        
        vel_diff = linear_vel - current_vel
        
        throttle = self.pid_controller.step(vel_diff, time_diff)
                
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        brake = 0.0
        
        if linear_vel == 0.0 and current_vel < 0.1:
            brake = 400
            throttle = 0.0
        elif vel_diff < 0 and throttle < 0.1:
            throttle = 0.0
            decel = max(self.decel_limit, vel_diff)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
        return throttle, brake, steering
