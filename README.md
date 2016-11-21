CS2024 Project: 
===================================

Team members: Michael Choi (cc2373), Weilong Guo (wg97)

Introduction
------------
The aim of this project is to explore the famous wine dataset <sup>[2]</sup> using k-means clustering and Naive Bayes classification in mlpack. The data set contains result of chemical analysis of three types of wines grown in a certain plantation in Italy. The goal is to identify the three types of wine via clustering method, and evaluate the model by its prediction performance. 
We learnt about the C++ packages mlpack <sup>[1]</sup> and Armadillo <sup>[3]</sup>.

Description of algorithms
-------------------------
Our main reference for k-means clustering and Naive Bayes classification can be found in the documentation of <sup>[1]</sup>. 


Description of files
--------------------
* [main.cpp](main.cpp) -- main file 
* [wine.data](wine.data) -- Wine data from <sup>[2]</sup>
* [wine.names](wine.names) -- Wine data description from <sup>[2]</sup>
* [assignments_default](assignments_default) -- Clustering result from k-means clustering with old start
* [assignments_withguess](assignments_withguess) -- Clustering result from k-means clustering with warm start
* [assignments_nbc](assignments_nbc) -- Clustering result from Naive Bayes classification

Results
-------
To compile our code, we can execute 
```
g++ main.cpp -lmlpack -O2 -larmadillo
```


References
----------
<sup>[1]</sup>: "[Mlpack](http://www.mlpack.org)"

<sup>[2]</sup>: "[Wine dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/wine/)"

<sup>[3]</sup>: "[Armadillo](http://arma.sourceforge.net/)"

