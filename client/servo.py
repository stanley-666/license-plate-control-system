from gpiozero import Servo
from time import sleep

servo = Servo(17)


def sg90_open() :
    print("[SERVO] OPEN GATE!!!")
    servo.min()
    sleep(0.5)
    servo.value = None


def sg90_close() :
    print("[SERVO] CLOSE GATE!!!")
    servo.max()
    sleep(0.5)
    servo.value = None


