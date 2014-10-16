import sys
from bisect import bisect_left
from itertools import islice

def cost(word, wordlist):
    try:
        index = wordlist.index(word)
    except ValueError:
        return len(word) + 1
    
    current = 0 #current letter = current cost
    minimum = len(word) #minimum cost

    while current < minimum: #possible to find a new minimum
        #get cost to select word from current position
        w = word[:current]       
        i = bisect_left(wordlist, w)

        select = (index + 1) - i
        #print(w, i, select, minimum)
        minimum = min(minimum, current + select)
        current = current + 1
    return minimum

def main():
    nCases = int(fin.readline())
    cases = (c.strip() for c in islice(fin, nCases))

    wordlist = [w.strip() for w in fwordlist]
    wordlist.sort()
    
    for case, word in enumerate(cases, 1):
        fout.write("Case #{}: {}\n".format(case, cost(word, wordlist)))        
    
if __name__ == "__main__":
    with open('Test_1_0_0.in') as fin:
        with sys.stdout as fout:
        #with open('Test_1_0_0.out', 'w') as fout:
            with open('dictionary_small.txt') as fwordlist:
                main()
