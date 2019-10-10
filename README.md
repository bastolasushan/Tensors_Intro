# Tensors_Intro
In general all current machine learning systems use tensors as their basic data structure. Tensors are fundamental to the field 


So exactly # What are tensors??

At its core a tensor is a container for data almost always numerical data. So, it's a container for numbers. You may be already familiar with matrices, which are 2D tensors: tensors are a generalization of matrices to an arbitary number of dimensions

If you don't know what matrices is :
         Those are the matematical term you studied in your high school maths
         
  In theory:
      A matrix is a collection of numbers arranged into a fixed number of rows and columns. Usually the numbers are real numbers.
     ex:
     X = [ 1, 2, 3]  Y= [2, 4, 5]
     
     
     
     
     
# Scalars( 0 D tensors)
     
     A tensor that contains only one number is called a scalar ( or scalar tensor , or 0-dimensional tensor, 0D )  
In , numpy a float32 or float64 number is a scalar tensor ( or scalar array).
You can display the number of axes of a Numpy tensor vis the ndim attribute 
a scalar tensor has 0 axes (ndim == 0) { ndim is basically a syntax which is number of dimensional in python}

```python
import  numpy as np
x = np.array(12)
x
>> array(12)     #Output
x.ndim
>> 0  # indicating as 0th dimension
```

# Vectors (1D tensors)
An array of numbers is called a vector, or 1D tensor. A 1D tensor is said to have exactly one axis. Following is a Numpy vector

```Python
x = np.array([12,3,6,14])
x
>> array ([12,3,6,14])
x.ndim
>> 1  # 1th dimension tensor
```


This vector has five entries and so is called a 5 dimensional vector. Don't confuse a 5D vector with a 5D tensor! A 5D vector has only one axis and has five dimensions along its axis, whereas a 5D tensor has five axes ( and may have any number of dimensions along each axis). Dimensionally can denote either the number of entries along a specific axis (as in the case of our 5D vector) or the number of axes in a tensor (such as 5D tensor), which can be confusing at times. In the latter case, it's tecnically more correct to talk about a tensor of rank 5 ( the rank of a tensor being the number of axes). but the ambiguous notation 5D tensor is common regardless

# Matrices (2D tensors)

An array of vectors is a matrix, or 2D tensor. A matrix has two axes (often referred to rows and columns). You can visually interpret a matrix as a rectangular grid of numbers. This is a Numpy matrix:

```Python
x= np.array ([5, 78, 2, 42, 1],
	     [6, 23, 4, 12, 4],
	     [3, 23, 42, 12, 5])
x.dim

>> 2 
```


# 3D tensors and higher dimensional tensors

If you pack such matrices in a new array , you obtain a 3D tensor, which you can visually interpret as a cube of numbers. 
Following is a Numpy 3D tensor:

```Python

x = np.array([ [[5, 12, 23, 24, 1],
                [5, 12, 23, 24, 1],
                [5, 12, 23, 24, 1]],
                
                 [[5, 12, 23, 24, 1],
                 [5, 12, 23, 24, 1],
                 [5, 12, 23, 24, 1]],
                 
                 [[5, 12, 23, 24, 1],
                 [5, 12, 23, 24, 1],
                 [5, 12, 23, 24, 1]] ])
                 
x.ndim     
>> 3 # It is indicating as 3 dimensional array
```


A tensor is defined by three key attribute
- number of axes ( rank) for instance , a 3d tensor has three axes and a matrix has two axes
- Shape - this is  a tuple of integers that describes how many dimensions that tensor has along each   axis  eg:  x = [2,3] means it has 2 shape and x= [2,3,4,5] means it has 4 shape
- Data type : (usually created dtype in Python libraries) - This is the type of the data contained in   the tensor; for instance , a tensor's type could be float32, unit8, float64, and so on. On rare       occasion you may see a char tensor. Note that string tensors don't exist in Numpy ( or in most       other libraries), because tensors live in preallocated, contiguous memory segments: and strings,     being variable length , would preclude the use of this implementation


So visualizing it in code:
----------------------------------
```Python
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# we display number of tensor in train_images data (remember the dimension thing)
print(train_images.ndim)
>> 3  # this means it has 3 dimension , 3D or 3 tensor)

# the shape
print(train_images.shape)
>> (60000, 28,28)  # which has 60K images with 28*28 pixels

# the data type

print(train_images.dtype)
>> unit8 # it's 8-bit integers
```

In short is is a 3D tensor of 8-bit integers. More precisely , it's an array of 60,000 matrices of 28*28 integers. Each such matrix is a grayscale image, with coefficients between 0 and 255.


Real world examples of data tensors
----------------------------------------
- Vector data - 2D tensors of shape (samples features)
- Timeseries data or sequence data - 3D tensors of shape (samples, timesteps, features)
- Images - 4D tensors of shape ( samples, height, width, channels) or (samples, channels, height,        width)
- Video - 5D tensors of shape ( samples, frames, height, width , channels) or (samples, frame,          channels, height, width)
