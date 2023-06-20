Ensemble Learning

Ensemble learning can only be used with supervised learning

Ensemble learning can be used for both regression and classification problems.

Ensemble learning can help to reduce overfitting.



Clustering 无监督

不需要先验知识，高维，可扩展，K-mean不能处理缺失值，层次可以，多种类型数据，噪声敏感，初始值敏感

Euclidean distance（欧几里得距离）：k-mean和层次都commonly用它

Mahalanobis distance（马氏距离）：层次也commomly用它

Hamming distance（汉明距离）：用于计算两个等长字符串在相同位置上不同字符的个数，适用于文本分类、信息检索等。

Cosine similarity（余弦相似度）：通过计算两个向量之间的夹角余弦值来衡量它们之间的相似性，适用于文本分类、推荐系统等。



层次：bottom-up approach

k-mean：partition

决策树 监督

可以处理各种类型的变量，对数据分布没有要求，可以处理missing value

特征关联度低，过拟合，噪声敏感，局部优化

DBSCAN vs k-mean：

DBSCAN can identify arbitrary shaped clusters

DBSCAN can handle noise effectively

DBSCAN does not require specifying the number of clusters in advance



What are the types of hierarchical cluster analysis? (Choose all that apply) A. Agglomerative B. Divisive



C. Centroid-based: This clustering method forms clusters by assigning data points to the nearest centroid (cluster center), such as the k-means algorithm.

D. Density-based: This clustering method forms clusters by grouping data points that are closely packed together, separated by areas of lower point density, such as the DBSCAN algorithm.



What are the applications of Bayesian statistics? (Choose all that apply)  C. Bayesian networks D. Bayesian updating

Gibbs sampling and B. Markov Chain Monte Carlo are computational methods used in Bayesian statistics.



D，层次：全局最优

K-mean：局部最优



在层次聚类中，树状图提供了合并过程的可视化表示，可用于通过在所需级别切割树来确定适当的聚类数。



回归模型

在回归问题中，残差图（residual plot）应服从均值为零且方差恒定的正态分布

在回归模型中，通常使用均方误差（Mean Squared Error，MSE）作为衡量模型性能的指标。



In logistic regression, the dependent variable is categorical

independent 自变量，dependent因变量

决策树

叶子节点：predictions or class labels

以下都可以减少overfitting：

A. Pre-pruning B. Post-pruning C. Bagging D. Boosting E. Regularization

随机森林是ensemble model，用决策树做base learner，随机森林可以减少overfitting

entropy（熵）

Gini Index is to measure the degree of impurity（杂质） or uncertainty within a dataset

CHAID is a decision tree algorithm that is primarily used for categorical data.



Monte-Carlo methods
用来模拟数值计算



In hypothesis testing, what does a Type I error represent?
拒绝 null hypothesis 即使是正确的

Type II

接受 错误的

In hypothesis testing, increasing the sample size will always reduce the probability of a Type II error.

还有几个影响Type II的都是越大，越小

PCA is a dimensionality reduction technique that can be used to reduce the number of features in a dataset while retaining as much of the variation in the data as possible.



Which of the following is a method of missing data imputation? 

A. k-Nearest Neighbors



Information Theory

Entropy

Mutual information

Conditional entropy



Which of the following are examples of ensemble learning methods? (Choose all that apply) A. Bagging B. Boosting C. Random Forest D. Stacking

Information gain is the difference between the parent node's entropy and the weighted average entropy of the child nodes.

The Rule Fit procedure is used to build a predictive model that consists of a set of rules

The MICE package in R can be used for missing data imputation.

In similarity and distance measures, high similarity always corresponds to low distance.  False

The Rule Fit procedure is used to extract linear models from an ensemble of decision trees. It is a type of ensemble learning method that combines the strengths of decision trees and linear regression models.