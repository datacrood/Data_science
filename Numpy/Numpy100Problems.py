# %% [markdown]
# # 100 numpy exercises
# 
# This is a collection of exercises that have been collected in the numpy mailing list, on stack overflow
# and in the numpy documentation. The goal of this collection is to offer a quick reference for both old
# and new users but also to provide a set of exercises for those who teach.
# 
# 
# If you find an error or think you've a better way to solve some of them, feel
# free to open an issue at <https://github.com/rougier/numpy-100>.

# %% [markdown]
# File automatically generated. See the documentation to update questions/answers/hints programmatically.

# %% [markdown]
# Run the `initialize.py` module, then for each question you can query the
# answer or an hint with `hint(n)` or `answer(n)` for `n` question number.

# %%
%run initialise.py

# %% [markdown]
# #### 1. Import the numpy package under the name `np` (★☆☆)

# %%
import numpy as np

# %% [markdown]
# #### 2. Print the numpy version and the configuration (★☆☆)

# %%
print(np.__version__) #np.show_config())

# %% [markdown]
# #### 3. Create a null vector of size 10 (★☆☆)

# %%
null_array = np.zeros(10)
null_array

# %% [markdown]
# #### 4. How to find the memory size of any array (★☆☆)

# %%
print(null_array.size, null_array.itemsize, null_array.nbytes) 

# %% [markdown]
# #### 5. How to get the documentation of the numpy add function from the command line? (★☆☆)

# %%
# print(np.add.__doc__)
help(np.add)
# np.info(np.add)

# %% [markdown]
# #### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

# %%
zer = np.zeros(10, dtype=int)
# zer = np.empty(10)
zer[4]=1 #Indexing changes the original array
# zer1 = np.insert(zer, 4, [5,6]) Note that insert don't change on spot rather returns a copy
zer

# %% [markdown]
# #### 7. Create a vector with values ranging from 10 to 49 (★☆☆)

# %%
np.arange(10, 49)
# np.array(range(10,49))
# np.linspace(10, 48, 39) #Note start = 10, stop = 48 (not inclusive of 49), size = (49-10) 

# %% [markdown]
# #### 8. Reverse a vector (first element becomes last) (★☆☆)

# %%
norm = np.arange(10)
norm[::-1] # Using slicing, it only creates a view and not change norm permanently.
# norm.sort() # Perform sorting by traversing

# %% [markdown]
# #### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

# %%
np.arange(9).reshape(3,3)
# np.random.randint(0,8, (3,3))
# matrix = np.fromfunction(lambda i, j: i * 3 + j, (3, 3), dtype=int)

# %% [markdown]
# #### 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)

# %%
arr = np.array([1,2,0,0,4,0])
# np.where(arr!=0) For 2-d array it returns two arrays one telling x-coordinates and other about y-coordinate.
np.nonzero(arr)

# %% [markdown]
# #### 11. Create a 3x3 identity matrix (★☆☆)

# %%
np.identity(3, dtype=int) # It specifically creates a sqaure identity matrix
np.eye(3, dtype=int) # More generalised form to create ones on diagonal and zeros elsewhere
np.diag([1,1,1]) #Puts 1d into diagonals and zeros elsewhere.
np.diag(np.ones(3)) # Fetches/Extracts the diagonal elements in a matrix

# %% [markdown]
# #### 12. Create a 3x3x3 array with random values (★☆☆)

# %%
np.random.random((3,3,3)) # while np.random.rand(3,3,3) takes only one positional argument and not a tuple; randn for normal distibution
# random.uniform (creating floating random values within a range); random.randint  (for integer values), 

# %% [markdown]
# #### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

# %%
t = np.random.uniform(0,50, (10,10,10))
t[0]=np.nan
print(np.max(t)) #alias to np.amax()
print(np.nanmax(t)) # to ignore nan values if present
# np.maximum takes two arrays and tells the max sum of the argument

# %% [markdown]
# #### 14. Create a random vector of size 30 and find the mean value (★☆☆)

# %%
np.random.random(30).mean() #np.nanmean() to deal with nan values; can also provide axis. 
# np.average(weights = []) to calculate average based on customised weights.

# %% [markdown]
# #### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

# %%
s = np.ones((4,4), dtype=int)
# s[0, :], s[-1, :]=1,1 #More short: s[(0,-1), :]=1 
# s[:, 0], s[:, -1]=1,1 # More short s[:, (0,-1)]=1
# or s[0, :]= s[-1, :]=s[:, 0]= s[:, -1]=1
s[1:-1, 1:-1]=0 #Note: last element is not inclusive. therefore -3-1 = -4 will be inclusive. We only add one if steps = negative
print(s[1:-1, 1:-1].base,s[(0,1), :].base) #Difference between a copy and a base

# %% [markdown]
# #### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

# %%
# This time we have to add another row and column. 
x = np.random.randint(1, 50, (4,4)) #(existing_array, pad_width, constant_values, mode)
np.pad(x, pad_width=(2,3), mode='constant', constant_values=(9,8))
# ANother way is n_row, n_col = arr.shape ->
# new = np.zeros((n_row+padding width, n_col+width)) -> new[row_width:-row_width, col_width:-col_width] = arr

# %% [markdown]
# #### 17. What is the result of the following expression? (★☆☆)
# ```python
# 0 * np.nan
# np.nan == np.nan
# np.inf > np.nan
# np.nan - np.nan
# np.nan in set([np.nan])
# 0.3 == 3 * 0.1
# ```

# %%
print(0 * np.nan) #Any arithmetic operation with nan =nan
print(np.nan == np.nan) # nan = not a number so can't compare  
print(np.inf > np.nan) #again, comparison not possible
print(np.nan - np.nan) #arithmetic
print(np.nan in set([np.nan])) # in a set, np.nan is considered equal to itself
print(0.3 == 3 * 0.1) #It's important to be aware about floats due to small rounding error therfore, use np.isclose() 3*0.1 error will progate further if continued multiplying

# %% [markdown]
# #### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

# %%
np.diag([1,2,3,4], k=-1) #k =+1 will check above the diagonal.

# %% [markdown]
# #### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

# %%
gg = np.zeros((8,8), dtype=int)
gg[::2, ::2]=1
gg[1::2, 1::2]=1
print(gg[::2, ::2].base) #Note: It's creating a view.

# %% [markdown]
# #### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)

# %%
np.unravel_index(335, (6,7,8)) # often we flatten the 2d array into 1-d and then we want what it's coordinates could be in 2-d
#For example,So, in the 3-by-4 block the 2D index (1, 2) would correspond to the linear index 6 which is 1 x 4 + 2.
# unravel_index does the inverse. Given a linear index, it computes the corresponding ND index. Since this depends on the block dimensions, 
# these also have to be passed. So, in our example, we can get the original 2D index (1, 2) back from the linear index 6

# %% [markdown]
# ![sxwBU.png](attachment:sxwBU.png)

# %%
row, col = np.indices((4, 5), sparse=True) #np.indices is used to create matrix indexes, i and j. 
# np.meshgrid is used just to return individual coordinates. which otheriwise if we do manually will take a lot of time. (use for plotting and numerical computation)
hs = np.arange(44, 64).reshape(4,5)
hs[row, col]
hs1 = 3*row+5*col # equivalent to 3i+5j kind of matrix done using help of index of row, col. 
hs1

# %% [markdown]
# ![8Mbig.png](attachment:8Mbig.png)

# %% [markdown]
# #### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

# %%
# used to repeat patterns or create larger structures. For repeating an entire array along multiple dimensions. np.newaxis; np.repeat can also be used
arr = np.array([[1,0], [0,1]])
np.tile(arr, (4,4)) # How many repetitions along each dimension i.e., axis=0 and axis=1. Here 4 times along rows and 4 times aloong cols.

# %% [markdown]
# #### 22. Normalize a 5x5 random matrix (★☆☆)

# %%
z = np.random.uniform(50, 80, (5,5))
# No. 1 way is Z-score nomralisatioon: to subtract mean from each and dividing all by standard deviation.
(z-np.mean(z))/np.std(z) 
# Second way using min-max normalistion
(z-np.min(z))/np.max(z)-np.min(z)
# L2 normalisation
q3, q1 = np.percentile(z, (75, 25))
(z-q1)/(q3-q1) #Note IQR = q3-q1

# %% [markdown]
# #### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

# %%
# using dictioanary
# custom1 = np.dtype()
# uSing tuples
# custom2 = np.dtype()
# np.array(, dtype= custom1)

# %% [markdown]
# #### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

# %%
import numpy as np
np.random.seed(0)
np.random.randint(10, 30, (5,3)).dot(np.random.randint(10, 30, (3,2))) # Either use np.dot or @ (It's the matrix multiplication)
np.random.randint(10, 30, (5,3))@(np.random.randint(10, 30, (3,2))) # Either use np.dot or @
np.dot(np.random.randint(10, 30, (5,3)), np.random.randint(10, 30, (3,2))) # Either use np.dot or @
# THis is the hadamard product that is simply multiplying each element. Note: the shape must match to perform this
np.random.randint(10, 30, (3,3))*(np.random.randint(10, 30, (3,3))) 

# %% [markdown]
# #### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

# %%
import numpy as np
s = np.arange(1, 20)

mask = (s>=3) & (s<=8) #Note 'and' don't work in numpy, we have to use & or |. This is boolean way to index elements
# s[mask]=-1 # Note: doing s[mask].base is not returning a view. This doesn't mean that it is always creating a copy. May be at the backend numpy is optimising but not returning the base as view
# This is permanently changing the values.

# 2nd way
s[(s>=3)&(s<=8)]*= -1

# Third way
np.where(mask, -s, s)

# %% [markdown]
# #### 26. What is the output of the following script? (★☆☆)
# ```python
# # Author: Jake VanderPlas
# 
# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))
# ```

# %%
print(sum(range(5),-1)) #Sum takes an iterable (range(5)) in this case and with initial value of -1 adds the iterables: -1+0+1+2+3+4
from numpy import * #ALways avoid as this can lead to namespace pollution, casting numpy keywords over buil-in python
print(sum(range(5),-1)) #numpy's sum function sums along the -1 axis therefore = 0+1+2+3+4
# %reset -f THis resets the namespaces

# %% [markdown]
# #### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
# ```python
# Z**Z
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# Z<Z>Z
# ```

# %%
import numpy as np
Z = np.arange(10)
Z**Z # Raises each element to the power of itself NOte: 0**0 = 1
2<<Z>>2 # << left shift; >>right shift are bitwise operations.
Z<-Z # compares each element with self
# Z<Z<Z # Chained comparisons are not possible, need to use separatly (may be using parenthesis)

# %% [markdown]
# #### 28. What are the result of the following expressions? (★☆☆)
# ```python
# np.array(0) / np.array(0)
# np.array(0) // np.array(0)
# np.array([np.nan]).astype(int).astype(float)
# ```

# %%
np.array(0) +9 # Scalar value, Note: Different from np.array([0])
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0)) # SUrprisingly, This doesn't result nan because numpy handles integer division by 0 differently
np.array([np.nan]).astype(int).astype(float) # np.nan is represented by a very small floating point value. to git it a representation, but any operation with nan = nan

# %% [markdown]
# #### 29. How to round away from zero a float array ? (★☆☆)

# %%
# z = np.random.uniform(-10, 10, (5,3))
z = np.array([-5.5, -5.56, -5.53, -5.43, -5.001, -0.7, -0.3, 3.5, 3.57, 3.3, 0.7, 0.3, 0])
print(z, np.where(z>1, np.ceil(z), np.floor(z)))
# No.2 approach
# np.copysign(np.)
np.copysign(np.ceil(abs(z)), z[::-1]) #Copysign takes two argument, new iterable, and what sign to copy from (Either provide single sign or iterable same size to new)

# %% [markdown]
# #### 30. How to find common values between two arrays? (★☆☆)

# %%
import numpy as np
z1 = np.random.randint(0, 10, 10)
z2 = np.random.randint(0, 10, 10)
print(list(set(z1).intersection(set(z2)))) 
print(set(z1) & set(z2)) #Note and don't work
print(np.intersect1d(z1, z2)) # Note: 1d because finding intersect of 2d don't bother about the shape, but the elements. 

# %% [markdown]
# #### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

# %%
# np.seterr(all="ignore")
np.ones(1) / 0
# Back to sanity
np.seterr(all='warn')
np.ones(1)/0

# %% [markdown]
# #### 32. Is the following expressions true? (★☆☆)
# ```python
# np.sqrt(-1) == np.emath.sqrt(-1)
# ```

# %%
# np.sqrt(-1) == np.emath.sqrt(-1) # np.sqrt(-1) returns nan with warning whereas np.emath handles it efficiently
np.sqrt(-1) #returns nan
np.emath.sqrt(-1) #returns 1j
np.nan==float("-inf") # False for comparing nan with any thing.

# %% [markdown]
# #### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

# %%
import numpy as np
yesterday = np.datetime64('today')-np.timedelta64(1)
tomorrow = np.datetime64('today')+np.timedelta64(1)
print(yesterday, np.datetime64('today'), tomorrow)

# %% [markdown]
# #### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

# %%
# Note: Stick to python looping only if no other way in numpy library because you are sacrificing your speed
# start = np.datetime64('2016-07-01')
# dates = []
# for i in range(31):
#     dates.append(start)
#     start+=np.timedelta64(1)
np.arange('2016-07-01', '2016-07-31', dtype='datetime64') #However we generallyy use pandas to deal with time-series data

# %% [markdown]
# #### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)

# %%
A = np.ones((4,4))
B = np.ones((4,4))
(A+B).astype(int) #cast dtype using astype!
# Can also use np.add(out=), np.multiply, np.divide, np.negative(out=)

# %% [markdown]
# #### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)

# %%


# %% [markdown]
# #### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

# %%


# %% [markdown]
# #### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

# %%


# %% [markdown]
# #### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

# %%


# %% [markdown]
# #### 40. Create a random vector of size 10 and sort it (★★☆)

# %%


# %% [markdown]
# #### 41. How to sum a small array faster than np.sum? (★★☆)

# %%


# %% [markdown]
# #### 42. Consider two random array A and B, check if they are equal (★★☆)

# %%


# %% [markdown]
# #### 43. Make an array immutable (read-only) (★★☆)

# %%


# %% [markdown]
# #### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

# %%


# %% [markdown]
# #### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

# %%


# %% [markdown]
# #### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)

# %%


# %% [markdown]
# #### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)

# %%


# %% [markdown]
# #### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

# %%


# %% [markdown]
# #### 49. How to print all the values of an array? (★★☆)

# %%


# %% [markdown]
# #### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

# %%


# %% [markdown]
# #### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

# %%


# %% [markdown]
# #### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

# %%


# %% [markdown]
# #### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

# %%


# %% [markdown]
# #### 54. How to read the following file? (★★☆)
# ```
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11
# ```

# %%


# %% [markdown]
# #### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

# %%


# %% [markdown]
# #### 56. Generate a generic 2D Gaussian-like array (★★☆)

# %%


# %% [markdown]
# #### 57. How to randomly place p elements in a 2D array? (★★☆)

# %%


# %% [markdown]
# #### 58. Subtract the mean of each row of a matrix (★★☆)

# %%


# %% [markdown]
# #### 59. How to sort an array by the nth column? (★★☆)

# %%


# %% [markdown]
# #### 60. How to tell if a given 2D array has null columns? (★★☆)

# %%


# %% [markdown]
# #### 61. Find the nearest value from a given value in an array (★★☆)

# %%


# %% [markdown]
# #### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

# %%


# %% [markdown]
# #### 63. Create an array class that has a name attribute (★★☆)

# %%


# %% [markdown]
# #### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

# %%


# %% [markdown]
# #### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

# %%


# %% [markdown]
# #### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)

# %%


# %% [markdown]
# #### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

# %%


# %% [markdown]
# #### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)

# %%


# %% [markdown]
# #### 69. How to get the diagonal of a dot product? (★★★)

# %%


# %% [markdown]
# #### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

# %%


# %% [markdown]
# #### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)

# %%


# %% [markdown]
# #### 72. How to swap two rows of an array? (★★★)

# %%


# %% [markdown]
# #### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)

# %%


# %% [markdown]
# #### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

# %%


# %% [markdown]
# #### 75. How to compute averages using a sliding window over an array? (★★★)

# %%


# %% [markdown]
# #### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is  shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)

# %%


# %% [markdown]
# #### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

# %%


# %% [markdown]
# #### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)

# %%


# %% [markdown]
# #### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)

# %%


# %% [markdown]
# #### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

# %%


# %% [markdown]
# #### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)

# %%


# %% [markdown]
# #### 82. Compute a matrix rank (★★★)

# %%


# %% [markdown]
# #### 83. How to find the most frequent value in an array?

# %%


# %% [markdown]
# #### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

# %%


# %% [markdown]
# #### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)

# %%


# %% [markdown]
# #### 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)

# %%


# %% [markdown]
# #### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

# %%


# %% [markdown]
# #### 88. How to implement the Game of Life using numpy arrays? (★★★)

# %%


# %% [markdown]
# #### 89. How to get the n largest values of an array (★★★)

# %%


# %% [markdown]
# #### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)

# %%


# %% [markdown]
# #### 91. How to create a record array from a regular array? (★★★)

# %%


# %% [markdown]
# #### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

# %%


# %% [markdown]
# #### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

# %%


# %% [markdown]
# #### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)

# %%


# %% [markdown]
# #### 95. Convert a vector of ints into a matrix binary representation (★★★)

# %%


# %% [markdown]
# #### 96. Given a two dimensional array, how to extract unique rows? (★★★)

# %%


# %% [markdown]
# #### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

# %%


# %% [markdown]
# #### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

# %%


# %% [markdown]
# #### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

# %%


# %% [markdown]
# #### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)

# %%



