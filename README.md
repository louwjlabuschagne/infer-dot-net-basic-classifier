# Basic Classifier using Infer.Net

## Introduction
This repo looks at building a `1-feature` probabilistic classifier. Whilst reading the <a href='https://mbmlbook.com/EmailClassifier_A_model_for_classification.html'>Model Based Machine Learning</a> ebook by <a href="https://www.microsoft.com/en-us/research/people/jwinn/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fpeople%2Fjwinn%2F">John Winn</a> et al. from <a href="https://www.microsoft.com/en-us/research/">Microsoft Reseach</a> I found it very confusing diving right into the model based machine learning approach and was looking for a way to tie MBML with traditional ML - this repo is the result.

Although the <a href="https://mbmlbook.com/index.html">MBML book</a> is a brilliant book, the first 3 chapters cover very probabilistic approaches to solving problem - as it should, as that is what the book is all about 😅. However, coming from traditional machine learning I was glad to see that in Chapter 4 they build a binary classifier, albeit in a probabilistic fashion. The MBML book uses <a href="https://en.wikipedia.org/wiki/Factor_graph#:~:text=A%20factor%20graph%20is%20a,the%20factorization%20of%20a%20function.&text=Factor%20graphs%20generalize%20constraint%20graphs,where%20all%20factors%20are%20constraints.">factor graphs</a> across the board - if you're not aquited with factor graphs, the TLDR from  <a href="https://en.wikipedia.org/wiki/Factor_graph#:~:text=A%20factor%20graph%20is%20a,the%20factorization%20of%20a%20function.&text=Factor%20graphs%20generalize%20constraint%20graphs,where%20all%20factors%20are%20constraints.">wikipedia</a>:

A factor graph is a bipartite graph representing the factorization of a function. Given a factorization of a function $g$,

```math
$$g(X_1,X_2,\dots,X_n) = \prod_{j=1}^m f_j(S_j),$$
```

where ${\displaystyle S_{j}\subseteq \{X_{1},X_{2},\dots ,X_{n}\}}$, the corresponding factor graph ${\displaystyle G=(X,F,E)}$ consists of variable vertices ${\displaystyle X=\{X_{1},X_{2},\dots ,X_{n}\}}$, factor vertices ${\displaystyle F=\{f_{1},f_{2},\dots ,f_{m}\}}$, and edges ${\displaystyle E}$. The edges depend on the factorization as follows: there is an undirected edge between factor vertex ${\displaystyle f_{j}}$ and variable vertex ${\displaystyle X_{k}}$  iff ${\displaystyle X_{k}\in S_{j}}$.

In other words, factor graphs are a way to factorize functions, and factor graphs as used in MBML are used specifically where $g(X_1,X_2,\dots,X_n)$ is a joint distribution or a joint likelihood function, and the factorization depends on the conditional independencies among the variables. These factor graphs can be solved using various algorithms like Belief propagation, also known as sum-product message passing, and a go-to platform to perform these calculations is C#'s dotnet using Microsoft's inferDotNet package. 


## The Setup
Chapter 4 of MBML is titled: `Uncluttering Your Inbox` and is all about building a classifier that can classify whether a person is going to reply to an email or not, and then using this classifier to, well, uncluttering your inbox. The chapter starts off by building a naive 1 feature classifier with the following factor graph:


<p align="center">
    <img src='assets/email-classifier.jpg' width='300px'>
</p>

The factor graph translates into the following. We've got some `featureValue` that we observer (that's why it is grayed out - it's "observed") which is our feature value, i.e. $X_1$. Then we go and create the calculate $score = featureValue \times weight$, i.e. $y=w\times x_1$. After that it gets a bit funky. We are trying to build a classifier, in other words, we want to "learn" some mapping, $f(X)$, that we can pass our feature value(s) $x_1$ through and get a score that we can then threshold to say whether our feature value(s) comes from the one class or the other. The most basic classifier to do this job is Logistic Regression. Here we assume a mapping from $x$ to $y$ as

$${\displaystyle y={\begin{cases}1&\beta _{0}+\beta _{1}x+\varepsilon >0\\0&{\text{else}}\end{cases}}}$$

where $\beta_0$ and $\beta_1$ are parameters that we will estimate in our training process and $\epsilon$ is the standard error term. This is basically what this factor graph is trying to do, build a similiar classifier, although in a very different approach - the Model Based Machine Learning approach. From the <a href="https://mbmlbook.com/EmailClassifier_A_model_for_classification.html">MBML book</a>:


> You may be surprised to learn that many classification algorithms can be interpreted as doing approximate inference in some probabilistic model. So rather than running a classification algorithm, we can instead build the corresponding model and use an inference algorithm to do classification. Why would we do this instead of using the classification algorithm? Because a model-based approach to classification gives us several benefits:
> * The assumptions in the classifier are made explicit. This helps us to understand what the classifier is doing, which can allow us to improve how we use it to achieve better prediction accuracy.
> * We can modify the model to improve its accuracy or give it new capabilities, beyond those of the original classifier.
> * We can use standard inference algorithms both to train the model and to make predictions with it. This is particularly useful when modifying the model, since the training and prediction algorithms remain in sync with the modified model.


They MBML book does come with code snippets, however most of the time it is very objectified code, which doesn't really help to learn the basics of inferDotNet, you can see <a href='https://github.com/dotnet/mbmlbook/blob/master/src/4.%20Uncluttering%20Your%20Inbox/Models/OneFeatureModel.cs'>here</a>.

So instead of using their code, this repo looks at creating the above factor graph, but instead of classifying whether an email is replied to or not, we dumb it down even further and aim to classify flowers from the Iris dataset as Setosa or Virginica using their sepal length attribute. If you haven't used the Iris dataset before, it consists of 150 flower oberservations for three different Iris species: Setosa, Versicolor and Virginica with their associated attributes: sepal length, sepal width, petal length and petalwidth. Here we are only using the Setosa and Verginica species and their sepal length attribute that is disibtrubed as 

<p align="center">
    <img src='./notebooks/sepal-length-dist.jpg' width='70%'>
</p>

So our factor graph looks like this:

<p align="center">
    <img src='assets/iris-classifier.jpg' width='300px'>
</p>

So what is happening here? Model based machine learning is all about *assumptions*. The assumptions chapter 4 makes to create this factor graph is:

1. The feature values can always be calculated, for any email.
2. Each email has an associated continuous score which is higher when there is a higher probability of the user replying to the email.
3. If an email’s feature value changes by $x$, then its score will change by $weight \times x$ for some fixed, continuous weight.

In our flower case:

1. The feature values can always be calculated, for any flower.
2. Each flower has an associated continuous score which is higher when there is a higher probability of the flower being Setosa.
3. If an flower's feature value changes by $x$, then its score will change by $weight \times x$ for some fixed, continuous weight.

Assumption 1 and 2 is just house keeping, but assumption 3 gives rise to:

<p align="center">
    <img src='assets/assumption-3.jpg' width='300px'>
</p>

From MBML:
> In drawing the factor graph, we’ve had to assume some prior distribution for weight. In this case, we have assumed that the weight is drawn from a Gaussian distribution with zero mean, so that it is equally likely to be positive or negative.

Assumption 4 becomes: The weight for a feature is equally likely to be positive or negative.

We might be tempted to do the following:

<p align="center">
    <img src='assets/threshold.jpg' width='300px'>
</p>

Just add a threshold with a lot of variance centered around 0 and let the model go. However, MBML warns against this as any inference algorithm will fail trying to compute this model. The book gives the reason

> Unfortunately, if we attempt to run inference on this model then any inference algorithm we try will fail. This is because some of the observed values have zero probability under the model. In other words, there is no way that the data-generating process encoded by our model could have generated the observed data values. When your data has zero probability under your model, it is a sure sign that the model is wrong!

What this means, is there are some outliers appear to come from one side of the classification, but are really from the other class. Looking at our sepal length distribution below, we highlight where the generation part of our model based approach falls short.

<p align="center">
    <img src='assets/sepal-length-dist-with-overlap.jpg' width='500px'>
</p>

As is common in MBML, to combat this, we add some noise to our observation's score calculation. In other words, we take the score that is generated by taking the product of the sepal length ($x$) and the weight ($w$) and we add some Gaussian noise by fixing the mean of a Gaussian random variable to the calculated score ($x \times w$) with variance 10. This 10 is quite arbitrary


We've explained the reasoning behind the `weight`, according the MBML §4    
Back to the factor graph. Using factor graphs to model the problem at hand is *much* more general than just building a classifier, but here we are trying to do so. In order to do so though we've got to pull out some nifty modeling tricks. Because we are modelling the problem using distributions, the math breaks a bit if we observe a point that can't be generated from the underlying distribution. 




Our basic classifer will try and classify the setosa from the virginica irises from the Iris dataset, the distribution plot below shows this should be an easy task:

