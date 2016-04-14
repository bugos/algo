import sys

def countSheep(N):
	trackedDigits = set()
	lastNamedNumber = 0
	while len(trackedDigits) < 10: #we dont have all 0..9
		lastNamedNumber += N
		trackedDigits.update(str(lastNamedNumber))
	return lastNamedNumber

#Input
fin = sys.stdin
fin.readline()
for test_case, line in enumerate(fin, 1):
	number = int(line)
	if ( 0 == number ):
		print("Case #%d: %s" % (test_case, "INSOMNIA"))
	else:
		print("Case #%d: %d" % (test_case, countSheep(number)))