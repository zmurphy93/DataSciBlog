"""
K-Nearest Neighbors
"""

# Author: Zachary B. Murphy


import numpy as np
import math

class KNearestNeighborsClassifier():
    """K-Nearest Neighbors Classification
    
    A classifier that uses the labels of a training dataset to classify unseen
    data based on their similarity(distance) to the unseen data point in
    question.
    
    
    Parameters
    ----------
    k : int, optional (default=5)
        Number of neighbors used in the classification process.
          
    similarity_metric : string, optional (default="euclidean")
       The metric used to evaluate distances during the classification process.
       Possible values:
           
           - "euclidean": computes the Euclidean distance measure (sqrt(sum((x - y)^2)))
               
           - "manhattan": computes the Manhattan distance measure (sum(|x - y|))
               
           - "minkowski": computes the Minkowski distance measure (sum(|x - y|^p)^(1/p))
               
           - "chebyshev": computes the Chebyshev distance measure (max(|x - y|))
    
    weights: string, optional (default="uniform")
        Weights to assign to the distances of the k-nearest neighbors during 
        the classification process. Possible values:
            -"uniform": Uniform weights. All points are weighted equally.
            
            -"distance": weight distances of each neighbor by the inverse of 
            their distance to the point in question. In this case, closer 
            neighbors of a query point will have a greater influence than 
            those which are further away.
    
    p: float, optional (default=5.0)
    Parameter for the Minkowski similarity metric. When p = 1, this is 
    equivalent to using "manhattan" similarity, and "euclidean" similarity 
    for p = 2.
        
        """
    def __init__(self, k=3,  similarity_metric = "euclidean", 
                 weights="uniform", p=5.0):
        
        self.k = k
        self.similarity_metric = similarity_metric
        self.weights = weights
        self.p = p
        
    def train(self, Xtrain, ytrain):
         """Fit K-Nearest Neighbors model according to the given training data.
           
         Parameters
         ----------
        
         Xtrain : array-like, shape (n_samples, n_features)
            Training data.
         ytrain : array, shape (n_samples,)
            Target values.
            
        Returns
        ----------
        self: A trained classifier object with Xtrain and ytrain stored as 
        attributes.
         """
         self.Xtrain=Xtrain
         self.ytrain = ytrain
         return self
    
    def _euclidean_distance(self, Xtrain, X):
        """ Euclidean Distance Measure.
        
        Computes the Euclidean distance measure (sqrt(sum((x - y)^2)))
        
        Parameters
        ----------
        Xtrain : array-like, shape (n_samples, n_features)
            Training data.
        X: array-like, shape (n_samples, n_features)
            Array of testing points
        
        Returns
        ----------
        distance: array-like, shape (n_samples, n_features)
        The distances between each point between the two sets.
        """
        distance = 0
        for i in range(len(X)):
                distance += pow((X[i] - self.Xtrain[i]), 2)
        return math.sqrt(distance)
    
    
    def _manhattan_distance(self, Xtrain, X):
        """ Manhattan Distance Measure.
        
        Computes the Manhattan distance measure (sum(|x - y|))
        
        Parameters
        ----------
        Xtrain : array-like, shape (n_samples, n_features)
            Training data.
        X: array-like, shape (n_samples, n_features)
            Array of testing points
        
        Returns
        ----------
        distance: array-like, shape (n_samples, n_features)
        The distances between each point between the two sets.
        """
        distance = 0
        for i in range(len(X)):
            distance += abs((X[i] - self.Xtrain[i]))
        return distance
    
    
    def _minkowski_distance(self, p, Xtrain, X):
        """ Minkowski Distance Measure.
        
        Computes the Minkowski distance measure (sum(|x - y|^p)^(1/p))
        
        Parameters
        ----------
        p: float
        Parameter for the Minkowski similarity metric. When p = 1, this is 
        equivalent to using "manhattan" similarity, and "euclidean" similarity 
        for p = 2.
        
        
        Xtrain : array-like, shape (n_samples, n_features)
            Training data.
        X: array-like, shape (n_samples, n_features)
            Array of testing points
        
        Returns
        ----------
        distance: array-like, shape (n_samples, n_features)
        The distances between each point between the two sets.
        """
        distance = 0
        for i in range(len(X)):
            distance += pow(abs(X[i] - Xtrain[i]), self.p)**(1/(self.p))
        return math.sqrt(distance)


    def _chebyshev_distance(self, Xtrain, X):
        """ Chebyshev distance measure.
        
        Computes the Chebyshev distance measure (max(|x - y|))
        
        Parameters
        ----------
        Xtrain : array-like, shape (n_samples, n_features)
            Training data.
        X: array-like, shape (n_samples, n_features)
            Array of testing points
        
        Returns
        ----------
        distance: array-like, shape (n_samples, n_features)
        The distances between each point between the two sets.
        
        """
        distance = 0
        for i in range(len(X)):
            distance = max(abs((X[i] - self.Xtrain[i])))
        return distance


    def _compute_similarity(self, Xtrain, similarity_metric, X):
        """ Compute similarity given a similarity_metric.
        
        
        Parameters
        ----------
        Xtrain : array-like, shape (n_samples, n_features)
            Training data.
            
        similarity_metric : string, optional (default="euclidean")
        The metric used to evaluate distances during the classification process.
        Possible values:
           
           - "euclidean": computes the Euclidean distance measure 
           (sqrt(sum((x - y)^2)))
               
           - "manhattan": computes the Manhattan distance measure 
           (sum(|x - y|))
               
           - "minkowski": computes the Minkowski distance measure 
           (sum(|x - y|^p)^(1/p))
               
           - "chebyshev": computes the Chebyshev distance measure 
           (max(|x - y|))
           
           
        X: array-like, shape (n_samples, n_features)
            Array of testing points
        
        Returns
        ----------
        distance: array-like, shape (n_samples, n_features)
        The distances between each point between the two sets.
        
        """
        if self.similarity_metric == "euclidean":
            distance = self._euclidean_distance(self.Xtrain, X)
        elif self.similarity == "manhattan":
            distance = self._manhattan_distance(self.Xtrain, X)
        elif self.simlilarity == "minkowski":
            distance = self._minkowski_distance(self.p, self.Xtrain, X)
        elif self.similarity =="chebyshev":
            distance = self._chebyshev_distance(self.Xtrain, X)
        else:
            print("Invalid similarity measure. Use 'euclidean', 'manhattan', 'minkowski', or 'chebyshev'")
        return distance
            
            
    def predict(self, Xtest):
        """Perform classification on an array of test vectors Xtest.
        The predicted class ypred for each sample in X is returned.
        
        Parameters
        ----------
        Xtest : array-like, shape = [n_samples, n_features]
        
        Returns
        -------
        ypred : array, shape = [n_samples]
        """
        
            
        ypred = np.empty(Xtest.shape[0])
        if self.weights=='uniform':
            return None
        elif self.weights == 'distance':
            for i, test_sample in enumerate(Xtest):
                for x in self.Xtrain:
                    distances = self._compute_similarity(x, self.similarity_metric, test_sample)
                    distances=1./distances
                    inf_mask=np.isinf(distances)
                    inf_row=np.any(inf_mask, axis=1)
                    distances[inf_row]=inf_mask[inf_row]
                    idx = np.argsort(distances)
                idx = np.argsort([self._compute_similarity(x, self.similariy_metric, test_sample) for x in self.Xtrain])[:self.k]
                dist = 1. / dist
            inf_mask = np.isinf(dist)
            inf_row = np.any(inf_mask, axis=1)
            dist[inf_row] = inf_mask[inf_row]
        return dist
                k_nearest_neighbors = np.array([self.ytrain[i] for i in idx])
                ypred[i] = self._vote(k_nearest_neighbors)
        return ypred
  
# =============================================================================
# Testing section below    
# =============================================================================
if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    data=pd.read_csv("C:/Users/zmurp/github/Machine_Learning_Algorithms/datasets/diabetes.csv")
    data.median()
    dataset = data.fillna(data.median())
    X = dataset.iloc[:500, 0:8].values
    ytrain = dataset.iloc[:500, 8].values
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(X)

    Xtest=scaler.fit_transform(dataset.iloc[500:, 0:8].values)
    ytest=dataset.iloc[500:, 8].values
    
KNearestNeighborsClassifier().train(Xtrain, ytrain)._chebyshev_distance(Xtrain, Xtest)
print(dir(KNearestNeighborsClassifier))