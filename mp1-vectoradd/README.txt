EE5940 Machine Problem 1
Vector Additions

1) Untar mp1-vectoradd.tgz into your SDK C/src directory.

2) Edit the vectorAddOnDevice( ... ) function in vectoradd.cu and the
   vectorAddKernel( ... ) function in vectoradd_kernel.cu to complete the
   functionality of the vector addition on the device. Do not change
   the source code elsewhere. The length of the vector is defined such that one
   thread block will be sufficient to compute the entire solution vector.
   Compiling your code will produce the executable mp1-vectoradd in your SDK
   C/bin/linux/release directory. 

3) There are several modes of operation for the application.  

   No arguments: the application will create two randomly initialized vectors
   to add. After the device addition is invoked, it will compute 
   the correct solution vector using the CPU, and compare that solution with 
   the device-computed solution. If it matches (within a certain tolerance), 
   it will print out "Test PASSED" to the screen before exiting.  

   One argument: the application will use the random initialization to 
   create the input vectors, and write the device-computed output to the file 
   specified by the argument.  

   Two arguments: the application will initialize the two input vectors with 
   the values found in the files provided as arguments. No output is written
   to file.

   Three arguments: the application will read its inputs from the files provided 
   by the first two arguments, and write its output to the file provided in the
   third.  

   Note that if you wish to use the output of one run of the application as an
   input, you must delete the first line in the output file, which displays
   the accuracy of the values within the file. The value is not relevant for
   this application.  

4) Submit your solution via Moodle as a .tgz file. The file should contain
   the mp1-vectoradd folder provided, with all the changes and additions you
   have made to the source code.  In addition, provide a text file, Word
   Document, or PDF file with your answers to the following question.  

   1. What is the memory-access to floating-point computation ratio in each
      thread? Ignore the storing of the result. Only global memory loads should be
      counted towards your off-chip bandwidth.

Grading:  

Your submission will be graded based on the following parameters.  

Functionality/knowledge: 75%
    - Correct result output file for our test inputs.
    - Correct usage of CUDA library calls and C extensions.
    - Correct usage of thread id's in vector computation.

Report: 25%
    - Question 1: 25%
