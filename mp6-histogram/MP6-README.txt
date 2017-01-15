ECE5940 Machine Problem 6: Histogramming

Introduction:
-------------
Histograms are a commonly used analysis tool in image processing and data mining
applications. They show the frequency of occurrence of data elements over
discrete intervals, also known as bins. A simple example for the use of
histograms is determining the distribution of a set of grades.

Example:

Grades: 0, 1, 1, 4, 0, 2, 5, 5, 5

The above grades ranging from 0 to 5 result in the following 6-bin histogram:

Histogram: 2, 2, 1, 0, 1, 3


The MP directory contains the following files:
----------------------------------------------
   *) MP6-README.txt:          This file.
   *) Makefile:                The makefile to compile your code.
   *) util.h:                  Header file with some utility macros and function
			       prototypes.

   *) util.c:                  Source file with some utility functions.
   *) ref_2dhisto.h:           Header file for the reference kernel.
   *) ref_2dhisto.cpp:         Source file for the scalar reference
			       implementation of the kernel.

   *) test_harness.cpp:        Source file with "main()" method that has sample
			       calls to the kernels.

   *) opt_2dhisto.h:           Header file for the parallel kernel (currently
			       empty).

   *) opt_2dhisto.cu:          Source file for the parallel implementation of
			       the kernel (currently empty).


Your Task:
----------
"ref_2dhisto(...)" constructs a histogram from the bin ids passed in 'input'.

   *) 'input' is a 2D array of input data. These will all be valid bin ids, so
       no range checking is required.
   *) 'height' and 'width' are the height and width of the input.
   *) 'bins' is the histogram. HISTO_HEIGHT and HISTO_WIDTH are the dimensions
      of the histogram (and are 1 and 1024 respectively, resulting in a 1K-bin
      histogram).

Your task is to implement an optimized function: "void opt_2dhisto(...)"

Assumptions/constraints:
------------------------
    (1) The 'input' data consists of index values into the 'bins'.
    (2) The 'input' bins are *NOT* uniformly distributed. This non-uniformity is
        a large portion of what makes this problem interesting for GPUs.
    (3) For each bin in the histogram, once the count reaches 255, then
        no further incrementing should occur. This is sometimes called a
        "saturating counter". DO NOT "ROLL-OVER".

You should only edit the following files: "opt_2dhisto.h", "opt_2dhisto.cu",
and "test_harness.cpp". Do NOT modify any other files. Furthermore, only modify
"test_harness.cpp" where instructed to do so (view the comments in the file).
You should only measure the runtime of the kernel itself, so any GPU allocations
and data transfers should be done outside the function "opt_2dhisto". The
arguments to the function "opt_2dhisto" have been intentionally left out for you
to specify based on your implementation.

You may not use anyone else's histogramming solution; however, you are allowed
to use third-party implementations of primitive operations in your solution. If
you choose to do so, it is your responsibility to include these libraries in the
tarball you submit, and modify the Makefile so that your code compiles and
runs. You must also mention any use of third-party libraries in your report.
Failure to do so may be considered plagiarism. If you are uncertain whether
a function is considered a primitive operation or not, please inquire about it
on the discussion board.

To Run:
-------
The provided Makefile will generate an executable in the usual SDK binary
directory with the name "mp6-histogram".  There are two modes of operation for
the application.

    No arguments: The application will use a default seed value for the random
                  number generator when creating the input image.

    One argument: The application will use the seed value provided as
		  a command-line argument. When measuring the performance of
		  your application, we will use this mode with a set of
		  different seed values.

When run, the application will report the timing information for the sequential
code followed by the timing information of the parallel implementation. It will
also compare the two outputs and print "Test PASSED" if they are identical, and
"Test FAILED" otherwise.

The base code provided to you should compile and run without errors or warning, but will fail the comparison.

Submitting the MP:
------------------
Submit your solution via Moodle as a tarball containing the "mp6-histogram"
directory provided, with all the changes and additions you made to the source
code. In addition, provide a text file for your report. Your report should
simply contain a journal of all optimizations you tried, including those that
were ultimately abandoned or worsened the performance. Your report should have
an entry for every optimization tried, and each entry should note:

   1) The changes you made for the optimization.
   2) Any difficulties with completing the optimization correctly.
   3) The amount of time spent on the optimization (even if it was abandoned
      before working).
   4) If finished and working, the speedup of the code after the optimization
      was applied.

Grading:
--------
Your submission will be graded based on the following parameters:

Demo: 15%
   - Produces correct result output files for our test inputs.

Functionality:  50%
   - This is a qualitative portion of your grade. For this portion, we will
     grade the thoughtfulness you put into speeding up the application.

Report: 35%
   - Complete an accurate journal. We will at least check for discrepancies,
     optimizations that you did but didn't report, etc.
