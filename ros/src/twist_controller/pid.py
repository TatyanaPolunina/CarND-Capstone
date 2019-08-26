
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        self.reset()

    def reset(self):
        self.int_val = 0.0
        self.num_iter = 0
        self.total_i = 0
        self.last_error  = 0
        self.common_error = 0
        self.prev_error = 0.0
        self.tollerance = 0.001;
        self.grad_delta = 0.01;
        self.num_iter_to_tune = 100
        self.total_i = 0.0

    def step(self, error, sample_time):

        integral = self.int_val + error * sample_time;
        derivative = (error - self.last_error) / sample_time;

        val = self.kp * error + self.ki * integral + self.kd * derivative;

        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min
        else:
            self.int_val = integral
        
        self.total_i += abs(error * sample_time)
        self.last_error = error
        self.num_iter+=1
        
        self.common_error += abs(error);
        #exit if don't need to tune params
        if  self.num_iter % self.num_iter_to_tune != 0.0:        
            #rospy.logwarn(" {0} {1}".format(self.nub_iter, self.num_iter_to_tune))
            return val;
        
        self.common_error = self.common_error / self.num_iter
        if self.common_error > self.tollerance:
            diff_error = (self.prev_error - self.common_error) / (sample_time * self.num_iter_to_tune);
            #use -derivatives for backpropagation
            self.tuneParams(error, self.total_i, derivative, diff_error * self.grad_delta, sample_time * self.num_iter_to_tune)
            self.prev_error = self.common_error
            #clean up for the next iteration
            self.common_error = 0;
            self.num_iter = 0;
            self.total_i = 0;
   

        return val
        
    def tuneParams(self, change_p, change_i, change_d, delta, sample_time):
        self.kp += self.kp * change_p * delta / sample_time
        self.ki += self.ki * change_i * delta / sample_time
        self.kd += self.kd * change_d * delta / sample_time
