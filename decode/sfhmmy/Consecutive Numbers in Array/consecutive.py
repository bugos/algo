import sys
from itertools import islice
sys.stdin = open('Test_0_s_0.in')
sys.stdout = open('Test_0_s_0.out.txt', 'w')

def cons(array):
    count = 1
    maximum = 1
    
    for i in range(1, len(array)):
        if array[i] == array[i-1] + 1:
            count = count + 1
        else:
            maximum = max(maximum, count)
            count = 1
            
    return max(maximum, count)

def getInts():
    return [int(n) for n in fin.readline().split()]

def main():
    length, Nqueries = getInts()

    array = getInts()
    assert len(array) == length
    
    print("Initial array: {}".format(cons(array)))

    queries = islice(fin, Nqueries)
    for query, line in enumerate(queries, 1):
        x, y = map(int, line.split())
        array[x] = y
        print("Change #{}: {}".format(query, cons(array)))

    
if __name__ == "__main__":
    with sys.stdin as fin:
        with sys.stdout as fout:
            main()
