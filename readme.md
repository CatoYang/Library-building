# Purpose
In light of previous coding failures and being unable to understand the syntax of pandas simply because i dont have sufficient experience in it. i have decided that it would be better to create a library to for use in data processing. I found it largely odd that i have to code the preprocessing of data twice.
- Once during the EDA
- Second during the pipeline

Considering functions can be imported like a module it would be easier and less error prone if i used the same code for both of these steps.

The problem would be the fact it is not easily readable or verifyable for other readers.
This is why after every transformative step we should verify with basic examination commands like df.head() in the EDA. I also included a unit test to compare the transformations between a pipeline and the EDA

This would also be more important during scenarios where i have to examine a database quickly and not get stuck on syntax. This also allows for automated processing where i can load certain commonly done transformations and skip much of the cleaning and go straight into feature engineering.
This is evident in coding challenges where theres a time limit, using the library should BUY TIME