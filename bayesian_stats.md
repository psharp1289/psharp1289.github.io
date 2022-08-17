---
layout: page
title: Bayesian statistics
---

##### *"Conditional probabilities play an important role in investigating causal questions, as we often want to compare how the probability (or, equivalently, risk) of an outcome changes under different filtering, or exposure, conditions."* - Judea Pearl

The analogy to keep with us when parsing the language of probabilities is a *frequency table.* Here, imagine we're pulling random objects out of a hat, each with a shape and color. The frequency table below lists the number of objects with a given shape and color residing in the hat. From this information, we'll explore notions of conditional probabilities, independence, probability mass and density. 



```python
import pandas as pd
data=pd.read_csv('sample_data_class_1.csv',index_col=0)
data #print out contigency table
```


<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    jax: ["input/TeX","input/MathML","output/SVG", "output/CommonHTML"],
extensions: ["tex2jax.js","mml2jax.js","MathMenu.js","MathZoom.js", "CHTML-preview.js"],
TeX: {
  extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"]
},
  tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true,
      processEnvironments: true
    },
    "HTML-CSS": { availableFonts: ["TeX"] }
  });
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML-full"></script>


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>blue</th>
      <th>red</th>
      <th>marginal (shape)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>square</th>
      <td>10</td>
      <td>40</td>
      <td>50</td>
    </tr>
    <tr>
      <th>circle</th>
      <td>5</td>
      <td>15</td>
      <td>20</td>
    </tr>
    <tr>
      <th>triangle</th>
      <td>5</td>
      <td>25</td>
      <td>30</td>
    </tr>
    <tr>
      <th>marginal (color)</th>
      <td>20</td>
      <td>80</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>

### How to find the conditional probability


```python
data.loc['square'] #filter data to only view objects that are squares
```
By conditioning on square, we filter the data and only consider objects that are squares. We then get the following conditional probability $$p(blue|square) = 10/50 = 0.2$$. We computed this by taking the joint probability $$\begin{aligned} p(blue \cap square) \end{aligned}$$ and dividing it by the marginal probability $$p(square)$$. One can think of this as normalizing the frequency we get after filtering by the amount of items we've filtered. Colloquially, we could say, "Given that we're only considering square objects, what is the probability I find a blue object." As such, all probabilities must equal 1, or in terms of frequencies, the sum of frequencies must equal the marginal frequency of square objects.

Now let's assess the marginal probability of $$p(blue)$$. The marginal is the total of all blue objects. The probability is this total divided by how many objects there are, or in this case, $$p(blue)= \frac{20}{100} = 0.2$$. Here $$p(blue) = p(blue / square)$$, which means that event *blue* is independent of  event *square*.

Of note, I use the term *event* when a random variable is assigned to a given value. So, here, if shape and color are our two random variables, the two events are color=blue and shape=square. Alternatively, one could ask the question, are shapes independent of colors, which is at the level of random variables. Although we won't delve into it, you can see perhaps that the independence of a given color from a shape disappears when using different events. This is an indication of an interaction between these two random variables: that the effect of conditioning on shape depends on which level of color one is considering and vice versa. 

### Probability Mass

Probability mass and density are two related but different concepts. Mass is defined as the amount of 'stuff' in a given object. Here, our objects are *intervals*, which merely are arbitrary bins of data. Above, the data is categorical, and thus the categories define the data. But for interval-scale data, these bins could be any numerical range. If the scale went from 0 to 100 (let's say we're talking about test scores), we could denote probability masses for each 2 points on the scale. That is, for example, $$p([95,96])=.05$$ means that 5% of the test-takers scored between 95% and 96% on their exam.

$$\delta x=intervalWidth$$ 

To continue our exams example, this $$\delta x$$ is every 2-point interval starting at 0. Summing up the probability mass is a matter of summing up the ratio of exam scores in a given interval over the total number of test takers *for each interval*. This amounts to the equation below:

$$ Mass_{total}= \sum_{i=interval} \, p([x_i, \, x_i+\delta x]) $$ = 1

The summation of the probability mass of all intervals within a dataset must add up to 1.

### Probability Densities

The interesting case we're about to see is when data is continuous and each interval of data is as tiny as we can model it (a la what we do in calculus with continuous data). When you model data as such, we tend to discuss these probabilities as *densities* as opposed to *mass*. Recall, mass parses the total probability into intervals. By contrast, densities are the ratio of mass to the size of incredibly tiny intervals (the limit).The major takeaway is that probability densities can be *much greater than one* either in total or for a given interval.  

$$Density= \frac{p([x_i,\, x_i+\delta x])}{\delta x} $$

Here, densities are the probability mass of a very tiny interval divided by the size of that interval (i.e., how much mass per space). Importantly, these density values may be denoted by the exact same nomenclature as masses, i.e., by $$p(x)$$.

Like summing up all probability masses, the *integration* of a probability density distribution is also 1. This is because an integral mutliplies a density by the tiny width of the interval (in mathematical notation, $$dx$$, but we'll keep our notation, $$\delta x$$):

$$ Density_{total}=\int \delta x\ p(x)\ = \ 1$$

This is the same as:

$$\sum_{i=interval}\ \delta x\ \frac{p([x_i , \, x_i+\delta x])}{\delta x}$$

This equivalence delivers an intuition about integration. That is, integrating is a weighted sum (here the weighting is the change in x over the tiniest of regions) of densities.

### Deriving bayes theorem with a symmetry: filtering the data in two ways to compute joint probabilities

```python
import pandas as pd
data=pd.read_csv('sample_data_class_1.csv',index_col=0)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>blue</th>
      <th>red</th>
      <th>marginal (shape)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>square</th>
      <td>10</td>
      <td>40</td>
      <td>50</td>
    </tr>
    <tr>
      <th>circle</th>
      <td>5</td>
      <td>15</td>
      <td>20</td>
    </tr>
    <tr>
      <th>triangle</th>
      <td>5</td>
      <td>25</td>
      <td>30</td>
    </tr>
    <tr>
      <th>marginal (color)</th>
      <td>20</td>
      <td>80</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



Above we again see the contingency table of pulling objects from hats. Notice that there are two ways to arrive at the same joint probability of a square blue object. 

1. $$p(square \cap blue)=p(blue|square)\,p(square)$$

2. $$p(square \cap blue)=p(square|blue)\,p(blue)$$

As you can see, a joint probability can be computed by filtering either by the row or column. This is the point from which one can derive Bayes' theorem by equating the two filterings.


$$p(blue|square)\,p(square)=p(square|blue)\,p(blue)$$

By dividing each side by $$p(square)$$ we have Bayes' theorem which is a translation between conditional probabilities, or filters.

$$p(blue|square)=\frac{p(square|blue)\,p(blue)}{p(square)}$$

### Using Bayes' theorem to work with parameters and data

Moving forward, we'll use what we've learned but replace a row of the frequency table with data and the columns by parameters. Eventually, we'll also nest parameters within a given model that provides constraints on which parameters are under consideration, and how they function in a data-generating process.

Another way of putting this is that we'll be using Bayes techniques to infer latent values that explain the data we get. The ultimate goal of this project is to infer these latent quantities given the data, or evidence we have. The key terms in this enterprise are nomenclature attached to conditional and marginal probabilities found in the equation we've been working with:

$$\underbrace{p(\theta|Data)}_\text{posterior}=[\underbrace{p(Data|\theta)}_\text{likelihood} \,\,\underbrace{p(\theta)}_\text{prior}]\, / \, \underbrace{p(Data)}_\text{evidence}$$


Note that the denominator that we call "evidence" is simply a marginal probability. In many cases it's called the marginal likelihood given that it is a weighted average of all possible likelihoods, where the weighting is accomlished by the prior probability of a given set of parameters:

$$\sum_{\theta}p(Data|\theta)p(\theta)$$

An alternative way to think about the marginal likelihood is to sum up all the joint probabilities in a given row (if columns are $$\theta\text{'s}$$ and rows are datasets).


### Bayesian updating of hypotheses: models, parameters, and posteriors
To see Bayes in action, consider an example in which we flip a coin (get Data) to see if the coin is biased (make inferences about latent quantities, which we call $$\theta$$).

We have a prior belief in certain possible values of the bias of the coin, with 1.0 meaning the coin is biased always to be heads, and 0.0 meaning the coin is always biased to show tails. An unbiased coin has a parameter of 0.5. We will see how likelihood functions are dictated by a model of how to generate data, or what has been called a "generative model." 

The first thing we'll do is define a prior distribution. The beauty of this interactive code is that we can see how the posterior is affected by the prior simply by changing the relevant code. For now, we'll start off with a prior that favors the belief that the coin is unbiased: the peak probability over $$\theta=0.5$$ and the PMF decays towards 0 the more distant from 0.5 $$\theta$$ gets.


```python
"""
Bayesian updating of beliefs about the bias of a coin. The prior and posterior
distributions indicate probability masses at discrete candidate values of theta.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# theta is the vector of candidate values for the parameter theta.
n_theta_vals = 11
# Now make the vector of theta values:
theta = np.linspace(0, 1, n_theta_vals )

# p_theta is the vector of prior probabilities on the theta values.
p_theta = np.minimum(theta, 1-theta)  # Makes a triangular belief distribution.
p_theta = p_theta / np.sum(p_theta)     # Makes sure that beliefs sum to 1.


# Plot the results.
plt.figure(figsize=(12, 11))
plt.subplots_adjust(hspace=0.7)

# Plot the prior:
plt.subplot(3, 1, 1)
plt.stem(theta, p_theta, markerfmt=' ')
plt.xlim(0, 1.2)
plt.xlabel('$$\\theta$$')
plt.ylabel('$$P(\\theta)$$')
plt.title('Prior')
plt.show()
```
    

You'll notice that this is a probability *mass* function, with probability values at discrete values that sum to one. 

### Data
Below in the next bit of code is the Data we observe. Here, we observe 1 coin-toss, which came up heads.

```python
n_heads = 100
n_tails = 1
```

### Likelihood function and generative models
Likelihood functions denoted as $$p(Data|\theta)$$ can be thought of as part of the essence of a generative model. A generative model is a formal description of a mechanism that can generate observable data and is contingent upon parameter settings within the mechanism. For instance, in a reinforcement learning setting, the mechanism generating the data could be Q-learning algorithms. Inputting data and parameters into these algorithms can generate probabilities in the form $$p(data \ \theta,model)$$.


Here, we'll use the Bernoulli likelihood function as a way to generate data. We use this because it's suited exactly for what we're studying: a single, binary response. One can think of Bernoulli as defining the likelihood that one of the two binary outcomes is favored due to a bias term, denoted here as $$\theta$$.


```python
# Compute the likelihood of the data for each value of theta:
p_data_given_theta = theta**n_heads * (1-theta)**n_tails

#Plot the likelihood
plt.subplot(3, 1, 2)
plt.stem(theta, p_data_given_theta, markerfmt=' ')
plt.xlim(0, 1.2)
plt.xlabel('$$\\theta$$')
plt.ylabel('$$P(D|\\theta)$$')
plt.title('Likelihood')
plt.show()
```    


Note that although we previously computed Bayes' theorem on a single value in the example with objects being drawn from a hat, here, Bayes' theorem is computed on a *distribution*.

One line of code accomplishes this for the likelihood function above, but it's helpful to think of Bayes' theorem being computed iteratively on all candidate $$\theta$$ values defined by the prior distribution over $$\theta$$.

### Marginal likelihood
Think about how hard this would be to the extent that we have multiple parameters...


```python
# Compute the marginal likelihood
p_data = np.sum(p_data_given_theta * p_theta)
```

### Posterior: Re-allocating credibility
Now since we have all the pieces of Bayes theorem we can use it to reallocate credibility (probability) to candidate $$\theta$$ values that serve to explain our data (again that we saw a coin-flip come up heads). 


```python
# Compute the posterior:
p_theta_given_data = p_data_given_theta * p_theta / p_data   # This is Bayes' rule!

# Plot the posterior:
plt.subplot(3, 1, 3)
plt.stem(theta, p_theta_given_data, markerfmt=' ')
plt.xlim(0, 1.2)
plt.xlabel('$$\\theta$$')
plt.ylabel('$$P(\\theta|D)$$')
plt.title('Posterior')
plt.show()
```    


### How the prior affects the posterior

(1) Try plugging in 1000 heads for the data. Then run the rest of the code. What do you notice, and why?

(2) Change the prior to favor tails. What do you notice about the posterior with 10 heads, and why?

### How sample size affects precision
Notice how the number of heads and tails affects the distribution by changing the "data" block. 

### Specifying a prior distribution by finding a meaningful functional form

Although we've dealth with a simple prior distribution to illustrate Bayesian re-allocation of credibilities across possible parameters, we have not yet dealt with the idea of specifying typical prior, continuous distributions. 

To do this, we need to find a distribution that satisfies two conditions for a good 'mathematical description' of the data: (1) The form of the distribution should be "comprehensible with meaningful parameters" (Kruschke p.24). (2) the distribution should be "descriptively accurate" in that it "looks like" the distribution of one's actual data. 

The beta distribution is a good candidate because it satisfies these conditions **for our data**. The beta distribution has a scale ranging from 0 to 1 (which is the natural range of coin biases) and is defined by two meaningful parameters, $$a$$ and $$b$$, that control the prior bias to favor either $$\theta=0$$, via an increasing $$b$$ or favoring $$\theta=1$$ via an increasing $$a$$.

Formally the beta distribution is defined as such:

$$\theta^{(a-1)}\,(1-\theta)^{(b-1)} \, / \, B(a,b)$$

where $$B(a,b)=\int_{0}^{1}{d\theta}\,\theta^{(a-1)}\,(1-\theta)^{(b-1)}$$


The mathematical details aren't of much interest, but what is is that the beta *distribution* is different from the beta function. The latter, defined as $$B(a,b)$$ ensures that the probability density function $$B(\theta / a,b)$$ integrates to 1 (see point above on probability denisities).

For more info on using beta distirbution to compute a posterior over likely $$\theta$$ values, see here: https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/master/05_BernBeta.py

Let's run the following code to see what happens when we input the $$\theta=[0.9,0.7,0.5,0.3,0.1]$$ into a beta density functions with parameters $$a=4\,,\,b=4$$



```python
from scipy.stats import beta

beta.pdf([0.9,0.7,0.5,0.3,0.1],4,4)
```

Assuming the higest number is the peak of the distribution, what can we say with the output of the function? Why are some outputs greater than 1? What can we infer about the coin if this is the true underlying distribution of real $$\theta$$ values?


### Ways to solve for a posterior distribution over parameters

The most difficult part of estimating posteriors is to solve for the marginal likelihood, given that it requires integration over potentially many parameters. 

1. **Analytical**
    - When the functional form of the likeihood * prior is equivalent to the prior, the prior and likelihood are said to be "conjugate" and the marginal likelihood can be computed analytically. This means there is a mathematical equation that one can leverage to solve for the posterior without difficult or incalculable integration. Such is the case with a beta prior and Bernoulli likelihood as was the case in the example above tossing coins. 
   
   
2. **Numerical approximation via a grid of relevant parameters points**
    - The integral is approximated by a sum of several discrete values. This is what we did in the example last class when we dealth with 11 candidate values for $$\theta$$, or the coin's bias. We could have computed the posterior analytically, which just means plugging in one's evidence into an equation to compute the posterior. When in continuous space, one can discretize it by choosing e.g., 1,000 values spanning the full space of 0 to 1 in the beta distribution, and used a weighted sum to approximate the integral. Remember, the integral

The marginal likelihood involves estimating the probability of the data given all relevant parameter settings under consideration. Many models have **multiple parameters**: $$(Data|\theta,\beta,\alpha,\gamma,\delta,\epsilon)$$. 

In this situation, if one uses 1,000 parameter settings for each parameter, **the number of combinations of parameters = $$1000^6$$**, which is a number many computers cannot chug through when approximating an integral numerically.


### Using MCMC to compute the posterior without Bayes

Sampling $$\theta$$ from the posterior distribution. You do this via an algorithm that essentially uses randomness and simple decision rules to visit the relative frequency of candidate $$\theta$$ values that are present in the posterior distribution.

### Algorithm

*(AKA Paul's 5 Step Guide to a Perfect Posterior)*

1. Start with a guess of $$\theta$$ and calculate its $\theta$ by computing $$p(Data|\theta)p(\theta)$$. Thus, one still needs to specify a prior and a likleihood function to do MCMC. 


2. Randomly choose a new $$\theta$$ from a "jumping" distribution (also known as a proposal distribution). Jumping just means, am I going to jump to larger or smaller $\theta$ value from my current $$\theta$$.


3. A jumping distribution could be $$ jump \sim  \mathcal{N} (0,\sigma)$$ where a given jump, either larger (positive) or smaller (negative) from one's current $\theta$ is determined from randomly sampling from this distribution (centered on staying where one is at). 


4. Use this equation to decide to jump or stay where one is currently at:
    - $$p_{jump}=\text{min}\,\,(1,\frac{p(Data|\theta_{jump})p(\theta_{jump})}{p(Data|\theta_{current})p(\theta_{current})})$$
    
    Let's break this down a bit: The probability of jumping to a new value is determined by the ratio of the "probability" (it's the joint probability of $$\theta$$ and one's data) of the current $\theta$ and the proposed $\theta$ one can jump to. What happens if the proposed value is greater than current value?


5. Each time you sample a given $\theta$, record it: this is a **sample** from the posterior! In the long-run, sampling in this way will approximate the **actual posterior distribution**. Pretty cool.


> **_Note on terms:_**   If you see ["metropolis"](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) or ["gibbs"](https://en.wikipedia.org/wiki/Gibbs_sampling) algorithms, these are *instances* of MCMC.


### Using MCMC to recover two parameters: 2 coins, flipped simultaneously, each with their own bias

$$p(\theta_1,\theta_2|Data)$$. The data will consist of {H,T} for N times we flip both coins.
* note here that $\theta_1$ and $\theta_2$ are independent of each other, but they need not be. If, for instance, they were made by the same mint, they would not be. This would require hierarchical modelling, which I promise we will get to soon!


```python
"""
Use this program as a template for experimenting with the Metropolis algorithm
applied to 2 parameters called theta1,theta2 defined on the domain [0,1]x[0,1].

"""

from __future__ import division
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
import ipympl



# Define the likelihood function.
# The input argument is a vector: theta = [theta1 , theta2]

def likelihood(theta):
    # Data are constants, specified here:
    z1, N1, z2, N2 = 24, 100, 82, 100
    likelihood = (theta[0]**z1 * (1-theta[0])**(N1-z1)
                 * theta[1]**z2 * (1-theta[1])**(N2-z2))
    return likelihood


# Define the prior density function.
# The input argument is a vector: theta = [theta1 , theta2]
def prior(theta):
    # Here's a beta-beta prior:
    a1, b1, a2, b2 = 3, 3, 3, 3
    prior = beta.pdf(theta[0], a1, b1) * beta.pdf(theta[1], a2, b2) 
    return prior


```

### Defining our priors and parameters when there are multiple parameters

Why is there a multiplication symbol in both the prior and the likelihood?

What condition must be met in order to derive the distributions in this way?

Building blocks:

1. $$p(\theta_{1}|\theta_{2})=p(\theta_{1})$$

2. $$p(\theta_{1} \, \cap \, \theta_{2})= ?$$



```python

# Define the relative probability of the target distribution, as a function 
# of theta.  The input argument is a vector: theta = [theta1 , theta2].
# For our purposes, the value returned is the UNnormalized posterior prob.
def target_rel_prob(theta):
    if ((theta >= 0.0).all() & (theta <= 1.0).all()):
        target_rel_probVal =  likelihood(theta) * prior(theta)
    else:
        # This part is important so that the Metropolis algorithm
        # never accepts a jump to an invalid parameter value.
        target_rel_probVal = 0.0
    return target_rel_probVal

# Specify the length of the trajectory, i.e., the number of jumps to try:
traj_length = 200 # arbitrary large number

# Initialize the vector that will store the results.
trajectories = np.zeros((3,traj_length,2))

# Specify where to start the trajectory
trajectories[0,0,] = [0.50, 0.50] # arbitrary start values of the two param's
trajectories[1,0, ] = [0.1, 0.9] # arbitrary start values of the two param's
trajectories[2,0, ] = [0.05, 0.1] # arbitrary start values of the two param's

# Specify the burn-in period.
burn_in = 0

# Initialize accepted, rejected counters, just to monitor performance.
n_accepted = [0,0,0]
n_rejected = [0,0,0]

# Specify the seed, so the trajectory can be reproduced.
np.random.seed(47405)
```

### A multivariate normal jumping distribution

Why multivariate?
See below, run the code on the covariance matrix. Why do we need a covariance matrix?


```python
# This is the variance of the jumping distribution.
# It must be a covariance matrix
n_dim, sd1, sd2 = 2, 0.2, 0.2
covar_mat = [[sd1**2, 0], [0, sd2**2]]

for starting_point in range(3):
# Now generate the random walk. step is the step in the walk.
    for step in range(traj_length-1):
        current_position = trajectories[starting_point,step, ]
        # Use the proposal distribution to generate a proposed jump.
        # The shape and variance of the proposal distribution can be changed
        # to whatever you think is appropriate for the target distribution.
        proposed_jump = np.random.multivariate_normal(mean=np.zeros((n_dim)),
                                                     cov=covar_mat)
        # Compute the probability of accepting the proposed jump.
        prob_accept = np.minimum(1, target_rel_prob(current_position + proposed_jump)
                                / target_rel_prob(current_position))
        # Generate a random uniform value from the interval [0,1] to
        # decide whether or not to accept the proposed jump.
        if np.random.rand() < prob_accept:
            # accept the proposed jump
            trajectories[starting_point,step+1, ] = current_position + proposed_jump
            # increment the accepted counter, just to monitor performance
            if step > burn_in:
                n_accepted[starting_point] += 1
        else:
            # reject the proposed jump, stay at current position
            trajectories[starting_point,step+1, ] = current_position
            # increment the rejected counter, just to monitor performance
            if step > burn_in:
                n_rejected[starting_point] += 1
            

# # Extract just the post-burnIn portion of the trajectory.
# accepted_traj = trajectories

# # Compute the means of the accepted points.
# mean_traj =  np.mean(accepted_traj, axis=0)
# # Compute the standard deviations of the accepted points.
# stdTraj =  np.std(accepted_traj, axis=0)

time_points=np.linspace(1,traj_length,traj_length)

fig = plt.figure()
ax = fig.gca(projection='3d')
#trajectory 1
x = time_points
y = trajectories[0,:,0]#bias of coin1
z = trajectories[0,:,1] #bias of coin2

#trajectory 2
y2 = trajectories[1,:,0]#bias of coin1
z2 = trajectories[1,:,1] #bias of coin2

#trajectory 3
y3 = trajectories[2,:,0]#bias of coin1
z3 = trajectories[2,:,1] #bias of coin2
ax.plot(x, y, z, label='trajectory 1')
ax.plot(x,y2,z2,label='trajectory 2')
ax.plot(x,y3,z3,label='trajectory 3')

ax.legend()
ax.set_xlabel('Time Steps')
ax.set_ylabel('Bias'  r'$\theta_{1}$')
ax.set_zlabel('Bias'  r'$\theta_{2}$')
#mat
plt.show()

```


### Things to check

Do the chains overlap (representativeness)

When the chains do not overlap at the beginning of the chains, this is called a "burn in" period. It's typical to remove these samples from the collection of samples that will eventually comprise the estimate of the posterior.

### What it looks like done well


```python
"""
Use this program as a template for experimenting with the Metropolis algorithm
applied to 2 parameters called theta1,theta2 defined on the domain [0,1]x[0,1].

"""

from __future__ import division
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-darkgrid')
import ipympl



# Define the likelihood function.
# The input argument is a vector: theta = [theta1 , theta2]

def likelihood(theta):
    # Data are constants, specified here:
    z1, N1, z2, N2 = 24, 100, 82, 100
    likelihood = (theta[0]**z1 * (1-theta[0])**(N1-z1)
                 * theta[1]**z2 * (1-theta[1])**(N2-z2))
    return likelihood


# Define the prior density function.
# The input argument is a vector: theta = [theta1 , theta2]
def prior(theta):
    # Here's a beta-beta prior:
    a1, b1, a2, b2 = 3, 3, 3, 3
    prior = beta.pdf(theta[0], a1, b1) * beta.pdf(theta[1], a2, b2) 
    return prior
         
# Define the relative probability of the target distribution, as a function 
# of theta.  The input argument is a vector: theta = [theta1 , theta2].
# For our purposes, the value returned is the UNnormalized posterior prob.
def target_rel_prob(theta):
    if ((theta >= 0.0).all() & (theta <= 1.0).all()):
        target_rel_probVal =  likelihood(theta) * prior(theta)
    else:
        # This part is important so that the Metropolis algorithm
        # never accepts a jump to an invalid parameter value.
        target_rel_probVal = 0.0
    return target_rel_probVal

# Specify the length of the trajectory, i.e., the number of jumps to try:
traj_length = 5000 # arbitrary large number

# Initialize the vector that will store the results.
trajectory = np.zeros((traj_length, 2))

# Specify where to start the trajectory
trajectory[0, ] = [0.50, 0.50] # arbitrary start values of the two param's

# Specify the burn-in period.
burn_in = int(np.ceil(.1 * traj_length)) # arbitrary number

# Initialize accepted, rejected counters, just to monitor performance.
n_accepted = 0
n_rejected = 0

# Specify the seed, so the trajectory can be reproduced.
np.random.seed(47405)

# This is the variance of the jumping distribution.
# It must be a covariance matrix
n_dim, sd1, sd2 = 2, 0.2, 0.2
covar_mat = [[sd1**2, 0], [0, sd2**2]]
print(covar_mat)

# Now generate the random walk. step is the step in the walk.
for step in range(traj_length-1):
    current_position = trajectory[step, ]
    # Use the proposal distribution to generate a proposed jump.
    # The shape and variance of the proposal distribution can be changed
    # to whatever you think is appropriate for the target distribution.
    proposed_jump = np.random.multivariate_normal(mean=np.zeros((n_dim)),
                                                 cov=covar_mat)
    # Compute the probability of accepting the proposed jump.
    prob_accept = np.minimum(1, target_rel_prob(current_position + proposed_jump)
                            / target_rel_prob(current_position))
    # Generate a random uniform value from the interval [0,1] to
    # decide whether or not to accept the proposed jump.
    if np.random.rand() < prob_accept:
        # accept the proposed jump
        trajectory[step+1, ] = current_position + proposed_jump
        # increment the accepted counter, just to monitor performance
        if step > burn_in:
            n_accepted += 1
    else:
        # reject the proposed jump, stay at current position
        trajectory[step+1, ] = current_position
        # increment the rejected counter, just to monitor performance
        if step > burn_in:
            n_rejected += 1

            
# End of Metropolis algorithm.

#-----------------------------------------------------------------------
# Begin making inferences by using the sample generated by the
# Metropolis algorithm.

# Extract just the post-burnIn portion of the trajectory.
accepted_traj = trajectory[burn_in:]

# Compute the means of the accepted points.
mean_traj =  np.mean(accepted_traj, axis=0)
# Compute the standard deviations of the accepted points.
stdTraj =  np.std(accepted_traj, axis=0)

# Plot the trajectory of the last 500 sampled values.
plt.plot(accepted_traj[:,0], accepted_traj[:,1], marker='o', alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'$\theta1$')
plt.ylabel(r'$\theta2$')

# Display means in plot.
plt.plot(0, label='M = %.3f, %.3f' % (mean_traj[0], mean_traj[1]), alpha=0.0)
# Display rejected/accepted ratio in the plot.
plt.plot(0, label=r'$N_{pro}=%s$ $\frac{N_{acc}}{N_{pro}} = %.3f$' % (len(accepted_traj), (n_accepted/len(accepted_traj))), alpha=0)
# Evidence for model, p(D).
# Compute a,b parameters for beta distribution that has the same mean
# and stdev as the sample from the posterior. This is a useful choice
# when the likelihood function is binomial.
a =   mean_traj * ((mean_traj*(1-mean_traj)/stdTraj**2) - np.ones(n_dim))
b = (1-mean_traj) * ( (mean_traj*(1-mean_traj)/stdTraj**2) - np.ones(n_dim))
# For every theta value in the posterior sample, compute 
# beta.pdf(theta, a, b) / likelihood(theta) * prior(theta)
# This computation assumes that likelihood and prior are properly normalized,
# i.e., not just relative probabilities.

wtd_evid = np.zeros(np.shape(accepted_traj)[0])
for idx in range(np.shape(accepted_traj)[0]):
    wtd_evid[idx] = (beta.pdf(accepted_traj[idx,0],a[0],b[0] )
        * beta.pdf(accepted_traj[idx,1],a[1],b[1]) /
        (likelihood(accepted_traj[idx,]) * prior(accepted_traj[idx,])))

p_data = 1 / np.mean(wtd_evid)
# Display p(D) in the graph
plt.plot(0, label='p(D) = %.3e' % p_data, alpha=0)
plt.legend(loc='upper left')
plt.savefig('Figure_8.3.png')

# Estimate highest density region by evaluating posterior at each point.
accepted_traj = trajectory[burn_in:]
npts = np.shape(accepted_traj)[0] 
post_prob = np.zeros((npts))
for ptIdx in range(npts):
    post_prob[ptIdx] = target_rel_prob(accepted_traj[ptIdx,])
print(post_prob)
# Determine the level at which credmass points are above:
credmass = 0.95
waterline = np.percentile(post_prob, (credmass))

HDI_points = accepted_traj[post_prob > waterline, ]

plt.figure()
plt.plot(HDI_points[:,0], HDI_points[:,1], 'C1o')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel(r'$\theta1$')
plt.ylabel(r'$\theta2$')

# Display means in plot.
plt.plot(0, label='M = %.3f, %.3f' % (mean_traj[0], mean_traj[1]), alpha=0.0)
# Display rejected/accepted ratio in the plot.
plt.plot(0, label=r'$N_{pro}=%s$ $\frac{N_{acc}}{N_{pro}} = %.3f$' % (len(accepted_traj), (n_accepted/len(accepted_traj))), alpha=0)
# Display p(D) in the graph
plt.plot(0, label='p(D) = %.3e' % p_data, alpha=0)
plt.legend(loc='upper left')

plt.savefig('Figure_8.3_HDI.png')

plt.show()


```
  
## Ways to determine if MCMC "worked"

### Representativeness

Are values representing those in the posterior? Does the initial value of the chain influence the sampled distribution? The answers to these questions explain when the sampling procedure "gets stuck" in regions that prevent it from fully sampling the posterior.

### Accuracy

Are the central tendencies and range of the distribution across multiple intiations of MCMC?

### Efficiency

Is it computable within a reaonable time?


