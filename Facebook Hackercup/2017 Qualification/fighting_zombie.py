'''
|C| BugOS
https://www.facebook.com/hackercup/problem/326053454264498/
'''
import math, fileinput, re, operator, itertools, functools

def getOptimalProbability(damage, spells):
  # pprint(spells)
  probabilities = []
  for spell in spells:
    # get spells from string
    values = re.split("d|", spell)
    diceRolls, remaining = spell.split('d');
    diceRolls = int(diceRolls);
    if len(remaining.split('+')) == 2:
      diceSides, diceAdder = map(int, remaining.split('+'));
    elif len(remaining.split('-')) == 2:
      diceSides, diceAdder = map(int, remaining.split('-'));
      diceAdder = - diceAdder;
    else:
      diceSides, diceAdder = [int(remaining), 0];
    
    # print(diceRolls, diceSides, diceAdder)
    
    # probabilities.append(getProbability(damage, diceRolls, diceSides, diceAdder))
    probabilities.append(getProbability(diceRolls, diceSides, diceAdder, damage))
  return max(probabilities)



# http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def nChooseR(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(operator.mul, range(n, n-r, -1))
    denom = functools.reduce(operator.mul, range(1, r+1))
    return numer//denom

#http://mathworld.wolfram.com/Dice.html
def getProbability(diceRolls, diceSides, diceAdder, damage):
  wantedValue = damage-diceAdder
  sumMin = diceRolls
  sumMax = diceRolls * diceSides
  if(sumMin >= wantedValue):
    return 1;
  if(sumMax < wantedValue):
    return 0;
  s = diceSides
  n = diceRolls
  sumProbability = 0
  for p in range(wantedValue, sumMax + 1):
    for k in range(math.floor((p-n)/s) + 1):
      sumProbability += (-1)**k * nChooseR(n, k) * nChooseR(p-s*k-1, n-1)
  sumProbability /= (s**n)
  return sumProbability
  

def main():
  fin = fileinput.input() #open('small2.in', 'r')
  T = int(fin.readline())
  for testCase in range(T):
      damage, spellCount = map(int, fin.readline().split())
      spells = fin.readline().split() # [rolls]d[sides]+[adder]
      print("Case #%d: %s" % (testCase+1, round(getOptimalProbability(damage, spells), 8)))

#if __name__ == "__main__":
main()

  # ∫∫∫...(x+y+...)*f(x)*f(y)*...dxdydz...


# https://www.reddit.com/r/learnpython/comments/36qhmz/python_probability_of_dice_function/
def diceSumProbability(dice_number, sides, diceAdder, target):
    rollAmount = sides**dice_number
    targetAmount = len([comb for comb in itertools.product(range(1, sides+1), repeat=dice_number) if sum(comb)+diceAdder >= target]) # changed = to >=
    return targetAmount / rollAmount
  

