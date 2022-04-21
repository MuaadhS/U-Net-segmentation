from PCA9685 import PCA9685
import time
 
pwm = PCA9685(0x40)
pwm.setPWMFreq(50)

def init():
#initial positon
    global step0,step2, step4, step6, step8, step10
    step0 = 1600
    step2 = 500
    step4 = 750
    step6 = 500
    step8 = 1000 # 1700
    step10 = 1100

    pwm.setServoPulse(0,step0)
    pwm.setServoPulse(2,step2)
    pwm.setServoPulse(4,step4)
    pwm.setServoPulse(6,step6)
    pwm.setServoPulse(8,step8)
    pwm.setServoPulse(10,step10)


def move(channel, step, pos):
    global step0,step2, step4, step6, step8, step10
    if step < pos:
        for i in range(step,pos,10):
            pwm.setServoPulse(channel,i)
            time.sleep(0.05)
    elif step > pos:
        for i in range(step,pos,-10):
            pwm.setServoPulse(channel,i)
            time.sleep(0.05)
    if channel == 0:
        step0 = pos
    elif channel == 2:
        step2 = pos;
    elif channel == 4:
        step4 = pos
    elif channel == 6:
        step6 = pos;
    elif channel == 8:
        step8 = pos;
    elif channel == 10:
        step10 = pos;

def motion():
    move(6, step6, 1000)
    move(2, step2, 1500)
    move(10, step10, 750)
    move(2, step2, 1000)
    move(0, step0, 1200)
    move(2, step2, 1500)
    move(10, step10, 1100)
    #move(2, step2, 1500)
    #move(10, step10, 1000)
    #move(2, step2, 2000)
    
    #back to origin
    pos0 = 1600
    pos2 = 500
    pos4 = 750
    pos6 = 500
    pos8 = 1000
    pos10 = 1100
    #move(0, step0, pos0)
    move(2, step2, pos2)
    move(0, step0, pos0)
    move(4, step4, pos4)
    move(6, step6, pos6)
    move(8, step8, pos8)
    move(10, step10, pos10)
    
init()