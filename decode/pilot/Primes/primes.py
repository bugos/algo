from itertools import count, islice
from math import sqrt
from fractions import gcd
from operator import mul
from functools import reduce
from time import time

#c`andidates = (6 * n + pm for n in count(1) for pm in [-1,1])
#coprime30 = (1, 7, 11, 13, 17, 19, 23, 29)
#candidates = (30 * n + c for n in count(1) for c in coprime30)

PRIMORIAL_FACTORS = 2, 3, 5, 7, 11,# 13, 17
PRIMORIAL = reduce(mul, PRIMORIAL_FACTORS)#1, 2, 6, 30, 210, 2310,...
COPRIMES =tuple(n for n in range(1,PRIMORIAL,2) if gcd(n, PRIMORIAL) == 1)
print(len(COPRIMES))

#Returns an estimation -incl. false positives- of all primes
def candidateGenerator():
    yield from PRIMORIAL_FACTORS # drop the 1
    generator = (k * PRIMORIAL + c for k in count() for c in COPRIMES)
    next(generator) # drop the 1
    yield from generator
#print([i for i in islice(candidateGenerator(), 20)])

# For n > 2    
def isPrime(n):
    sqrtn = int(sqrt(n))
    for p in candidateGenerator():
        if p > sqrtn:
            return True
        if n % p == 0:
            return False

def getPrime():
    for candidate in candidateGenerator():
        if isPrime(candidate):
            yield candidate

start = time()
for i in islice(getPrime(), 100000):
    pass
end = time()
print(i)
print(end - start)
