# Basic Classifier using Infer.Net

Inspired by the `one-feature` email classifier dicussed in <a href='https://mbmlbook.com/EmailClassifier_A_model_for_classification.html'>MBML</a> we'd like to build a simple classifier using infer.net.

The factor graph dicussed in MBML looks like:

<img src='assets/email-classifier.jpg' width='40%'>

with very objectified code <a href='https://github.com/dotnet/mbmlbook/blob/master/src/4.%20Uncluttering%20Your%20Inbox/Models/OneFeatureModel.cs'>here</a>.


Our basic classifer will try and classify the setosa from the virginica irises from the Iris dataset, the distribution plot below shows this should be an easy task:

<img src='./notebooks/sepal-length-dist.jpg' width='70%'>