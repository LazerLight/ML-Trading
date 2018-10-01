import numpy as np

#Creating Arrays with initial values
def test_run():
    #List to 1D array
    print np.array([1,5,4]) 
    #List of tuples to 2D array
    print np.array([(1,5),(66,55,4)]) 


def test_run1():
    #Creating empty array
    print np.empty(5)
    print np.empty((6,9), dtype=np.int_)

    #Creating an array of 1s
    print np.ones((5,3))

def test_run2():
    #Generate an array of random numbers with range [0.0, 1.0)
    print np.random.random((5,4)) 

    #A single int in [0,10)
    print np.random.randint(10)
    #Same as above, specifying [low, high) explicit
    print np.random.randint(0,10)
    #5 random ints as a 1D array
    print np.random.randint(0,10,size=5)
    #2x3 array of random ints
    print np.random.randint(0,10,size=(2,3,3))    

#Displaying Array Attributes

def test_run3():
    a = np.random.random((5,4)) 

    #Get number of rows and columns and amount of dimensions (2D,3D etc)
    print a.shape[0] #Rows
    print a.shape[1] #Columns
    print len(a.shape) #Number of dimensions

    #Get shape of the array (tuple with the dimensions)
    print a.shape
    #Return amount of elements in an array
    print a.size 
    # Get data type of elements
    print a
    print a.dtype


# Operations on ndarrays
def test_run4():
    np.random.seed(693) # Seed the random number generator
    a = np.random.randint(0,10, size=(5,4)) # 5x4 random int in [0,10)
    print "Array:\n", a

    # Sum of all elements
    print "Sum of all elements: ", a.sum()

    #Iterate over rows to compute sums of each column. Axis = 0 refers to downards across rows
    #Minimum of each column
    print "Sum of each column:\n", a.sum(axis=0)
    print "Minumum of each column:\n", a.min(axis=0)

    #Iterate over columns to compute sums of each row. Axis = 1 refers to horizontally across columns
    #Max of each row
    print "Sum of each row:\n", a.sum(axis=1)
    print "Maximum of each row:\n", a.max(axis=1)

    #Mean of all elements(leave out axis argument)
    print "Mean of all elements: ", a.mean()

    #Get index of max value of a 1D array 
    print np.argmax(np.array([1,2,3]))



# Accessing array elements
def test_run5():
    a = np.random.rand(5, 4)
    print "Array:\n", a

    # Accessing element at position (3,2)
    element = a[3, 2]
    print element

    #Slice Elements in the 1 through 3rd column in the 0th row
    print a[0,1:3]

    #Slicing n:m:t specifies a range that starts at n and stops before m, in steps size t
    print a[:, 0:3:2]

    #Accessing using list of indices (Will access element at index 1, 1, 2 and then 3)
    b = np.random.rand(5)
    indices = np.array([1,1,2,3])
    print b[indices]


#Modifying Array Elements
def test_run6():
    a = np.random.rand(5,4)
    print "Array:\n", a

    #Assigning a value to a particular location
    a[0,0] = 1
    print "\n Modified :\n",a 

    #Assigning a single value to an entire row
    a[0,:] = 2
    print "\n Modified :\n",a 
    
    #Assigning a list to a column in an array (Sizes must match)
    a[:,3] = [1,2,3,4,5]
    print "\n Modified :\n",a 

#Boolean or 'Mask' Index Arrays
def test_run7():
    a = np.array([(20,25,10,23,26,32,10,5,0),(0,2,50,20,0,1,28,5,0)])
    print a

    #calculate mean
    mean = a.mean()
    print mean

    #masking
        #to return an array containing only values less than the mean:
    print a[a<mean]
        #to return the arrays with the values less than the mean replaced by the mean:
    a[a<mean] = mean
    print a

#Arithmetic Operations

def test_run8():
    a = np.array([(1,2,3,4,5),(10,20,30,40,50)])
    print "Original array a:\n", a

    #Multiply a by 2
    print "\nMultiply a by 2:\n", 2*a

    #Divide a by 2
    print "\nMultiply a by 2:\n", a/2
    print "\nMultiply a by 2:\n", a/2.0

    b = np.array([(100,200,300,400,500),(1,2,3,4,5)])
    print "Original array b:\n", b
    
    #Add the two arrays
    print "\nAdd a+b:\n", a+b

    #Multiply the two arrays (Element-wise multiplication)
    print "\nMultiply a+b:\n", a*b


test_run8()