'''
Developed by Evangelos "bugos" Mamalakis, 12-04-2014
https://code.google.com/codejam/contest/2974486/dashboard#s=p1
'''
def find_needed_friends(shyness):
    friends = 1
    for people in shyness:
        friends = friends + 1
        friends = friends - people
    return t + X / f

def output(test_case, time):
    more_dec_places = repr(time)
    out = "Case #%d: %s" % (test_case, more_dec_places)
    print out

#Input
fin = open('small.in', 'r')
fin.readline()
for test_case, line in enumerate(fin, 1):
    line = map(float, line.split())
    _, shyness = line
    friends = find_needed_friends(shyness)
    output(test_case, time)
