import sys
#Input
fin = sys.stdin
fin.readline()
for test_case, line in enumerate(fin, 1):
    line = line.rstrip('\n')
    stack = list(line);
    stack.reverse()
    swaps = 0
    prevChar = '+'
    for char in stack:
    	if (char != prevChar):
    		swaps += 1
    		prevChar = char;
    print("Case #%d: %d" % (test_case, swaps))
    	