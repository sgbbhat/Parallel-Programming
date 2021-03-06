Data Parallel Reduction


1) Unpack the tarball into your SDK "C/src" directory.

2) Edit the source files "vector_reduction.cu" and "vector_reduction_kernel.cu" 
   to complete the functionality of the parallel addition reduction on the 
   device, assuming an input array of any size. Compiling your code will
   produce the executable "mp5.1-reduction" in your SDK "C/bin/linux/release"
   directory.

3) There are two modes of operation for the application.  

   No arguments: the application will create a randomly initialized array to 
   process. After the device kernel is invoked, it will compute the correct
   solution value using the CPU, and compare it with the device-computed
   solution. If it matches (within a certain tolerance), it will print out
   "Test PASSED" to the screen before exiting.  

   One argument: the application will initialize the input array with the
   values found in the file specified by the argument.

   In either case, the program will print out the final result of the CPU and
   GPU computations, and whether or not the comparison passed.  

4) Submit your solution via Moodle as a tarball. Your submission should
   contain the "mp5.1-reduction" folder provided, with all changes and
   additions you made to the source code. In addition, add a text file, Word
   document, or PDF file with your answers to the following questions:

   1. How many times does your thread block synchronize to reduce its portion
      of an array to a single value?

   2. What is the minimum, maximum, and average number of "real" operations 
      that a thread will perform? "Real" operations are those that directly 
      contribute to the final reduction value.


Grading:

Your submission will be graded based on the following parameters.  

Functionality/knowledge: 65%
    - Uses an O(N) data-parallel reduction algorithm.
    - Produces correct result output for test inputs.

Report: 35%
    - Answer to Question 1: 15%, answer to Question 2: 20%

