#!/bin/python

import sys


timeCharged = float(raw_input().strip())

if (timeCharged > 4):
    life =  8.00
else :
    life =  timeCharged * 2


rounded_life = "%.4f" % life

print life

