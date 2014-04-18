'''
Developed by Evangelos "bugos" Mamalakis, 12-04-2014
https://code.google.com/codejam/contest/2974486/dashboard#s=p1
'''
def simulate(C, F, X):
    #C: Farm Price
    #F: Farm Cookies per Second
    #X: Wanted Cookies

    t = 0. #Time
    f = 2. #Cookies per Second
    
    while is_efficient(f):
        #buy and update
        t = t + C / f #mazevoume ta aparaithta mpiskota
        f = f + F #anevazoyme thn apodosh agorazontas

    return t + X / f

def is_efficient(f):
    cookies_needed = X - C
    time_needed = cookies_needed / f
    return time_needed * F > C

def output(test_case, time):
    more_dec_places = repr(time)
    out = "Case #%d: %s" % (test_case, more_dec_places)
    print out

#Input
fin = open('large.in', 'r')
fin.readline()
for test_case, line in enumerate(fin, 1):
    line = map(float, line.split())
    C, F, X = line
    time = simulate(C, F, X)
    output(test_case, time)
