#Note: make sure wordlist and input files end with a newline.

import sys
from bisect import insort

wordlist = list(open('dictionary.txt'))
sys.stdin = open('Test_1_s_0.in')
#sys.stdout = open('Test_1_s_0.out.txt', 'w')

wordlist.sort()

def cost(word):
    minimum = len(word) #minimum cost
    current = 0 #current letter = current cost
    while current < minimum: #possible to find a new minimum
        #get cost to select word from current position
        l = wordlist.copy()
        w = word[:current]
        
        insort(l, w)

        index = wordlist.index(word)
        i = l.index(w)

        select = index - i
        
        minimum = min(minimum, current + select)
        current = current + 1
    return minimum

if __name__ == "__main__":
    
    cases = int(input())

    for case, word in enumerate(sys.stdin, 1):
        print("Case #{}: {}".format(case, cost(word)))        
    assert(case == cases)
