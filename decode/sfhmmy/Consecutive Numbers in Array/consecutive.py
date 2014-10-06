import sys
sys.stdin = open('Test_0_s_0.in')
sys.stdout = open('Test_0_s_0.out.txt', 'w')

def getInts():
    return list(map(int, input().split()))

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

if __name__ == "__main__":
    length, queries = getInts()

    array = getInts()
    print("Initial array: {}".format(cons(array)))

    for query, line in enumerate(sys.stdin, 1):
        x, y = map(int, line.split())
        array[x] = y
        print("Change #{}: {}".format(query, cons(array)))

    sys.stdout.flush()
    
