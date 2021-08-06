import itertools
counter = itertools.count(start=5, step = -2.5)
#print(counter)  Runs till infinity
print(next(counter))  # runs step by step
print(next(counter))
print(next(counter))

data = [100, 200,300, 400]
daily_data = list(zip(itertools.count(),data))   # I think same as enumerate counts all in the list. This is done by zip function.
print(daily_data)

daily_data = list(zip(range(10),data)) # Runs counter and ends where the data input stopped.
print(daily_data)

daily_data = list(itertools.zip_longest(range(10),data)) # Runs counter till where the range stops innstead of data input.
print(daily_data)

#cycle also returns an iterator that goes to infinity. It takes on an iterator and cycles through it till infinity.

counter = itertools.cycle([1,2,3])
#print(counter)  Runs till infinity
print(next(counter))  # runs step by step
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))

counter = itertools.cycle(('ON','OFF'))
print(next(counter))  # runs step by step
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))

# Another infinity iteratable function is repeat

counter = itertools.repeat(2, times=3)
print(next(counter))
print(next(counter))
print(next(counter))

squares = map(pow, range(10), itertools.repeat(2))
print(list(squares))
#print(squares)

# itertools.starmap
squares = itertools.starmap(pow, [(0,2),(1,2),(2,2)])
print(list(squares))


letters = ['a','b','c','d']
numbers = [0,1,2,3]
names = ['corey','adam','ikenna']

result = itertools.combinations(letters,2)  # combinations the order doesn't matter
for item in result:  
    print(item)

result = itertools.permutations(letters,2)  # permutations the order does matter
for item in result:  
    print(item)

# itertools.product allows for repeat of the order
result = itertools.product(letters,repeat = 2) 
for item in result:  
    print(item)

result = itertools.combinations_with_replacement(numbers,4)  # combinations_with_replacement allows for repeat of the order where order doesn't matter.
for item in result:  
    print(item)

#using addition to add list creates another list in memory so to avoid that, we can use itertools.chain to combine the lists without adding then together.
#combine = letters  + numbers + names
combined = itertools.chain(letters, numbers, names)
for item in combined:  
    print(item)

# islice would slice a porting in an iterator. itertools.islice
result = itertools.islice(range(10),5)

for item in result:
    print(item)

result = itertools.islice(range(10),1,6,2) # first value is start, second is stop and third is space. when it has only 1 value, it assumes stop with start at 0 and increment of 1

for item in result:
    print(item)

with open('test.log','r') as f:
    header = itertools.islice(f,3)  # Grabs the first 3 lines of the file.

    # for line in header:
    #     print(line)

    for line in header:
        print(line, end='')  # with no spaces in between the line.

# itertools.compress allows us to select certain values from iterables.
selectors = [True, True,False,True]
letters = ['a','b','c','d']
numbers = [0,1,2,3]

results =  itertools.compress(letters,selectors)

for item in results:
    print(item)

# compare to filter. you have to create a function.
def lt_2(n):
    if n < 2:
        return True
    return False

results = filter(lt_2, numbers) # filters the True values
for item in results:
    print(item)

results = itertools.filterfalse(lt_2, numbers) # filters the False values
for item in results:
    print(item)

# itertools.dropwhile drops all the flase value until it sees a True and then returns the rest of the iterables
numbers = [0,1,2,3,2,1,0]

results = itertools.dropwhile(lt_2, numbers) 
for item in results:
    print(item)

# itertools.takewhile takes all Flase values and stop when it reaches a True value.
results = itertools.takewhile(lt_2, numbers) 
for item in results:
    print(item)

#####################################################################
# Counter tool for homework.

from collections import Counter 
c = Counter([1, 2, 21, 12, 2, 44, 5, 
              13, 15, 5, 19, 21, 5]) 
  
for i in c.elements(): 
    print ( i, end = " ") 
print()
####################################################################

# itertools.accumulate adds all values by default can also take a multiplication on subtraction argument
results = itertools.accumulate(numbers) 
for item in results:
    print(item)

# # To use multiplication, or any other operator besides addition, we would eed to import operator module
# import operator

# numbers = [1,2,3,2,1,0]
# results = itertools.accumulate(numbers, operator.mul) 
# for item in results:
#     print(item)

# # itertools.groupby would go through an iterable and group values based on certain key. Returns a tuple of key and a tuple of all the values that were grouped by the key

