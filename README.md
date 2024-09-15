# Different Approaches for Time-Series Forecasting Stock Market Price

**Probabilistic Machine Learning** project for Master's degree in *Data Science and Artificial Intelligence* at the *University of Studies of Trieste*, lectures by prof. Luca Bortolussi.

The aim of this project is to evaluate the efficacy of various methodologies for forecasting stock market prices of diverse companies, utilizing different models:

## HMM with Expectation-Maximization (EM) Algorithm

HMMs are statistical models that represent systems with hidden states through observable sequences, where $$ \begin{cases} x_{i}=\text{observations}\\z_{i}=\text{hidden states} \end{cases}$$

Hidden Markov Models can be described as $$ \lambda=(A,B,\pi) $$ where $$ \begin{cases}A=\text{transition probability matrix}\\ B=\text{emission probability matrix}\\ \pi=\text{ initial probabilities of the states at }t=0\end{cases} $$

Training of the above HMM from given sequences of observations is done using the Baum-Welch algorithm which uses Expectation-Maximization (EM) to arrive at the optimal parameters for the HMM:

- Expectation (E) Step: Calculate the expected value of the log-likelihood with current parameter estimates.

- Maximization (M) Step: Maximize the expected log-likelihood to update the parameter estimates.

## Gaussian Hidden Markov Models(HMM) with K-Means Clustering Inizialization

Another method used in [1](https://ieeexplore.ieee.org/document/6199099) to initialize the HMMs is based on K-Means Clustering Algorithm.

K-Means is an unsupervised learning algorithm used to partition data into K clusters, where each data point belongs to the cluster with the nearest mean.

It’s based on the following steps:

- Initialization: Randomly select K initial cluster centroids.
- Assignment: Assign each data point to the nearest centroid.
- Update: Recalculate the centroids as the mean of all points in each cluster.
- Repeat: Iterate the assignment and update steps until convergence.

In this analysis, we will assume the following:

- Prior and Transition Probabilities: These are uniformly distributed across all states.
- Means and Variances: Each cluster found from K-Means is assumed to be a separate state component.
- Weights: Computed as the weights of the clusters, divided equally between the states to obtain the initial emission probabilities.

## Hidden Markov Model with Multivariate Gaussian Emissions trained using Variational Inference

Usually Second-Order Techniques are used to estimate parameters of probabilistic models from sample data, treating parameters as random variables with defined distributions.

Variational Bayesian Inference (VI) is a second-order approach that can be viewed as a variant of the EM algorithm, used for parameter estimation in probabilistic models.[2](https://arxiv.org/abs/1605.08618)

Advantages:

- Uncertainty Quantification: Allows numerical expression of the uncertainty associated with parameter estimation.
- Incorporation of Prior Knowledge: Enables the inclusion of prior knowledge about parameters in the estimation process.
- Automated Model Selection: the training process can easily be extended to automate the search for an appropriate number of model components in a mixture density model (e.g., Gaussian mixture model).
- Training in a Variational Framework: VI is used to perform HMM training within a variational framework, allowing for more flexible and robust parameter estimation.
Univariate Output Distributions: Typically applied to models with univariate output distributions (i.e., scalar values).

Any other information, can be found in the powerpoint presentation inside this repo!
