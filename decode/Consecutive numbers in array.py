##sys.stdin = open('input.txt', 'r')

def getInts():
    return list(map(int, input().split()))

def cons(array):
    count = 1
    maximum = 1
    for i in range(1, len(array)):
        if array[i] == array[i-1] + 1:
            count = count + 1
            maximum = max(maximum, count)
        else:
            count = 1
    return maximum

if __name__ == "__main__":
    length, queries = getInts()

    array = getInts()
    print("Initial array: {}".format(cons(array)))
    
    for query in range(1, queries+1):
        x, y = getInts()
        array[x] = y
        print("Change #{}: {}".format(query, cons(array)))

    

    
