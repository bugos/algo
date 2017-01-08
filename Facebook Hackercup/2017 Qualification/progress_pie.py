'''
|C| BugOS
https://www.facebook.com/hackercup/problem/1254819954559001/'''
import math, fileinput

cycleX = cycleY = cycleRadius = 50
def getColour(progress, X, Y):
  X -= cycleX #axes on the center of the cycle(50, 50)
  Y -= cycleY
  inCircle = cycleRadius**2 >= X**2 + Y**2
  atan = math.atan2(Y, X)
  if atan < 0: atan += 2 * math.pi
  inProgress = atan <= 2 * math.pi * progress / 100
  #print(inProgress, inCircle)
  if inProgress and inCircle:
    return "black"
  else:
    return "white"
  

fin = fileinput.input() #open('small2.in', 'r')
T = int(fin.readline())
for testCase in range(T):
    line = map(int, fin.readline().split()); # parse 3 ints
    print("Case #%d: %s" % (testCase+1, getColour(*line)));
