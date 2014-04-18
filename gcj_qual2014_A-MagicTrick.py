'''
|C| BugOS
https://code.google.com/codejam/contest/2974486/dashboard#s=p0
'''

def read_answer():
    answer = int(fin.readline())
    for i in range(4):
        if i+1 == answer: 
            line =  map(int, fin.readline().split())
        else:
            fin.readline()
    return line

fin = open('small2.in', 'r')
T = int(fin.readline())
for testCase in range(T):
    list1 = read_answer()
    list2 = read_answer()
    intersection = list(set(list1).intersection(list2))
    size = len(intersection)
    if size == 0:
        print "Case #%d: Volunteer cheated!" % (testCase+1)
    elif size > 1:
        print "Case #%d: Bad magician!" % (testCase+1)
    else: 
        print "Case #%d: %d" % (testCase+1, intersection[0])
