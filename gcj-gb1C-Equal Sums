# |C| BugOS
# http://code.google.com/codejam/contest/1836486/dashboard#s=p2&a=2
import itertools

def find_equal_sum_subsets(S):
    #S: the set of numbers 
    
    subsets = (c
               for length in range(1,len(S)+1)
               for c in itertools.combinations(S,length)
               )

    #finds a good subset pair
    result = ((i,j)
              for i,j in itertools.combinations(subsets,2)
              if sum(i)==sum(j)
              )
    

    print "Case #%d:" % (test_case) #, no_new_line
    try:
        a, b = next(result)
        print ' '.join(map(str, a))
        print ' '.join(map(str, b))
    except StopIteration:
        print 'Impossible'

#Input
fin = open('big.in', 'r')
fin.readline()
for test_case, line in enumerate(fin, 1):
    line = map(int, line.split())[1:20]
    find_equal_sum_subsets(line)
