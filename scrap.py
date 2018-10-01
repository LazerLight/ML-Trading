# import pandas as pd


# def test_run():
#     """Function called by Test Run."""
#     df = pd.read_csv("data/AAPL.csv")
#     # TODO: Print last 5 rows of the data frame


# if __name__ == "__main__":
#     test_run()

print('fuck')

# This is a comment
"""This is a comment"""

print('Filks'[3])
print('Filks'[1:3])
print(1,2,3,'fuck')

greeting ='hi'
greeting = 'hello' #reassing

print(greeting)

#Data Types

myStr = 'jk'
myInt = 5
myFloat = 1.2
myList = [1,2,4,'5']
myDict = {'a':3, 'h': 4, 'c': 4}

print(type(myStr), myStr)
print(type(myInt), myInt)
print(type(myFloat), myFloat)
print(type(myList), myList)
print(type(myDict), myDict)


#Accessing Dictionaries and Lists

print(myList[2])
print(myDict['a'])


#Concat
greeting = myStr + '444rkgjgtjgjgtkgkj'
print(greeting)

#Conditionals
x = 4

if x < 6:
    print('this is tru')
    print('stills tru')

if x < 6:
    print('this is trueeeee')
else: 
    print('this is false')

color = 'redfd'

if color == 'red':
    print('Color is red')
elif color == 'blue':
    print('color is blue')
else: 
    print('color is neither red nor blu')


if color == 'red' and x < 10:
    print("color is red and number is less than 10")

if color == 'red': 
    if x < 10: 
        print ("color is red and number is less than 10")

#Loops

people = ['Jehn', 'Sere', 'Tem', 'Beb']

for person in people:
    print ('Current person: ', person)

for i in range(len(people)):
    print('Current person: ', people[i], i)

for i in range(0, 10):
    print(i)




count = 1
while count < 34:
    print (count)
    count = count + 1
    if count == 7:
        break


#Functions
def sayFuck(name = 'Lo'):
    print('Fuck you', name)

sayFuck('Helga')
sayFuck()


def sums(val1, val2):
    return val1 + val2

numSums = sums(2,5)

print(numSums)


#Mutable vs Immutable

def addOne(number):
    number = number + 1
    print('Value inside', number)

number = 5

addOne(number)
print('Value outside', number)

def addToList(list):
    list.append(5)
    print('Value inside', list)

list = [1,4,5,5,4]
addToList(list)

print('Value inside', list)


#String functions

myfunk = 'funK Me'

print(myfunk.capitalize())
print(myfunk.swapcase())
print(len(myfunk))
print(myfunk.replace('funK', 'fuck'))
print(myfunk.count('M'))
print(myfunk.startswith('fun'))
print(myfunk.endswith('e'))
print(myfunk.split())
print(myfunk.find('Me'))
print(myfunk.index('Me'))

#Modules
def sayK(name):
    print('fuck you', name)
    return


#Classes & Objects
class Person:
    __name = ''
    __email = '' #Double underscore means the property is private

    def __init__(self,name,email):
        self.__name = name
        self.__email = email

    def set_name(self,name):
        self.__name = name
    
    def get_name(self):
        return self.__name
    
    def toString(self):
        return '{} can be contacted at {}'.format(self.__name,self.__email)


# yeet = Person()
# yeet.set_name('Yee Tee')

#or
yeet = Person('Fuck Yeet', 'fuck@you.com')
print(yeet.get_name())
print(yeet.toString())