// CS2024 Project - Investigating the k-means and Naive Bayes algorithm on the WINE dataset in mlpack
// Name: Chek Hin Choi (cc2373), Weilong Guo (wg97)
#include <iostream>
#include <algorithm>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
using namespace std;
using namespace mlpack::kmeans;
using namespace mlpack::naive_bayes;

// Create 3 centroids from subdata at index i,j,k
arma::mat create_centroid(const arma::mat& subdata,int i, int j, int k) {
  arma::mat out = join_horiz(join_horiz(subdata.cols(i,i),subdata.cols(j,j)),subdata.cols(k,k));
  return out;
}

// Calculate the classification rate from a given assignment vector
vector<double> classifyrate(const arma::Col<size_t>& assignments) {
  vector<double> out(4);
  arma::mat confusion(3,3,arma::fill::zeros);

  for(int i = 0; i < 59; ++i) 
    confusion(0,assignments(i))++;
  
  // Class 0 correct classification rate
  out[0] = confusion.row(0).max()/59.0;

  for(int i = 59; i < 130; ++i) 
    confusion(1,assignments(i))++;

  // Class 1 correct classification rate
  out[1] = confusion.row(1).max()/71.0;
 
  for(int i = 130; i < 178; ++i) 
    confusion(2,assignments(i))++;

  // Class 2 correct classification rate
  out[2] = confusion.row(2).max()/48.0;

  // Overall correct classification rate
  out[3] = (confusion.row(0).max() + confusion.row(1).max() + confusion.row(2).max())/178.0;
  
  return out;
}

int main()
{
  // Load data
  arma::mat data;
  data.load("wine.data");
  arma::Col<size_t> label(178,arma::fill::zeros);
  arma::Col<double> label2(data.col(0));
  for (int i = 0; i < 178; i++)
    label(i) = (int) label2(i)-1;
  data = data.t(); // Armadillo is column-based
  arma::mat subdata = data.rows(1,13); // Remove the lablels
  size_t clusters = 3; // Set initial cluster to 3
  
  // Create assignment vector
  arma::Col<size_t> assignments;

  // 1. Perform k-means clustering using the default algorithm in mlpack
  KMeans<> k;
  k.Cluster(subdata,clusters,assignments);
  vector<double> classes = classifyrate(assignments);
  cout << "Default k-means in mlpack with cold start (i.e. without specifying initial centroids): ";
  cout << classes[0] << ", " << classes[1] << ", " << classes[2] << ", " << classes[3] << endl;
  assignments.save("assignments_default",arma::arma_ascii);

  // 2. Perform k-means clustering with specified cluster centroid
  arma::mat centroid = create_centroid(subdata,0,59,130);
  assignments.clear();
  k.Cluster(subdata,clusters,assignments,centroid,false,true);
  classes = classifyrate(assignments);
  cout << "k-means in mlpack with warm start (i.e. specifying the correct initial centroids): ";
  cout << classes[0] << ", " << classes[1] << ", " << classes[2] << ", " << classes[3] << endl;
  assignments.save("assignments_withguess",arma::arma_ascii);

  // 3. Perform k-means clustering with random cluster centroid
  vector<int> idx(178);
  for(int i = 0; i < idx.size(); ++i)
    idx[i] = i;

  // Perform Monte Carlo simulation for 10,000 times and compute the average classification rate for each class 
  vector<double> rate(4);
  for(int i = 0; i < 10000; ++i) { 
    random_shuffle(idx.begin(),idx.end());
    centroid = create_centroid(subdata,idx[0],idx[1],idx[2]);
    k.Cluster(subdata,clusters,assignments,centroid,false,true);
    classes = classifyrate(assignments);
    rate[0] += classes[0];
    rate[1] += classes[1];
    rate[2] += classes[2];
    rate[3] += classes[3];
  }
  for (int i = 0; i < 4; ++i) 
    rate[i] /= 10000;
  
  cout << "k-means in mlpack with random start (i.e. specifying random initial centroids): ";
  cout << rate[0] << ", " << rate[1] << ", " << rate[2] << ", " << rate[3] << endl;

  // 4. Perform naive Bayes classification
  NaiveBayesClassifier<arma::mat> nbc(subdata,label,clusters);
  nbc.Classify(subdata,assignments);
  classes = classifyrate(assignments);
  cout << "Default Naive Bayes classifier in mlpack: ";
  cout << classes[0] << ", " << classes[1] << ", " << classes[2] << ", " << classes[3] << endl;  
  assignments.save("assignments_nbc",arma::arma_ascii);
  
  return 0;
}
