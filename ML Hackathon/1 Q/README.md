# MODELING ABSENTEEISM AT WORK DATASET USING SUPERVISED LEARNING
Machine Learning Models for Absenteeism at Work Dataset

## ABOUT DATASET
### ABSENTEEISM AT WORK DATASET
#### ABSTRACT
The database was created with records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil. Absenteeism at work An UCI dataset on Kaggle link data set : https://www.kaggle.com/loganalive/absenteeism-at-work-an-uci-dataset


    Data Set Characteristics  : Multivariate, Time-Series                Number of Instances   : 740 
    Attribute Characteristics : Integer, Real                            Number of Attributes  : 21
    Associated Tasks          : Classification, Clustering               Missing Values?       : N/A

#### METADATA
    Usage Information             License                     CC0: Public Domain (https://creativecommons.org/publicdomain/zero/1.0//)
                                  Visibility                  visibility Public
    
    Provenance                    Sources                     (https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work)

    Maintainers                   Dataset owner               rahul bhaskaran (https://www.kaggle.com/loganalive)

    Updates                       Expected update frequency   Never
                                  Last updated                2018-04-27                 
                                  Date created                2018-04-27                  
                                  Current version             Version 1

#### DATA SET INFORMATION
The data set allows for several new combinations of attributes and attribute exclusions, or the modification of the attribute type (categorical, integer, or real) depending on the purpose of the research.The data set (Absenteeism at work - Part I) was used in academic research at the Universidade Nove de Julho - Postgraduate Program in Informatics and Knowledge Management.

#### ATTRIBUTE INFORMATION
  1. Individual identification (ID)
  2. Reason for absence (ICD).
    
    Absences attested by the International Code of Diseases (ICD) stratified into 21 categories (I to XXI) as follows:
    
    I Certain infectious and parasitic diseases
    II Neoplasms
    III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
    IV Endocrine, nutritional and metabolic diseases
    V Mental and behavioural disorders
    VI Diseases of the nervous system
    VII Diseases of the eye and adnexa
    VIII Diseases of the ear and mastoid process
    IX Diseases of the circulatory system
    X Diseases of the respiratory system
    XI Diseases of the digestive system
    XII Diseases of the skin and subcutaneous tissue
    XIII Diseases of the musculoskeletal system and connective tissue
    XIV Diseases of the genitourinary system
    XV Pregnancy, childbirth and the puerperium
    XVI Certain conditions originating in the perinatal period
    XVII Congenital malformations, deformations and chromosomal abnormalities
    XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
    XIX Injury, poisoning and certain other consequences of external causes
    XX External causes of morbidity and mortality
    XXI Factors influencing health status and contact with health services.

    And 7 categories without (CID) 
    patient follow-up (22), 
    medical consultation (23), 
    blood donation (24), 
    laboratory examination (25), 
    unjustified absence (26), 
    physiotherapy (27), 
    dental consultation (28).
        
  3. Month of absence
  4. Day of the week (Monday (2), Tuesday (3), Wednesday (4), Thursday (5), Friday (6))
  5. Seasons (summer (1), autumn (2), winter (3), spring (4))
  6. Transportation expense
  7. Distance from Residence to Work (kilometers)
  8. Service time
  9. Age
  10. Work load Average/day
  11. Hit target
  12. Disciplinary failure (yes=1; no=0)
  13. Education (high school (1), graduate (2), postgraduate (3), master and doctor (4))
  14. Son (number of children)
  15. Social drinker (yes=1; no=0)
  16. Social smoker (yes=1; no=0)
  17. Pet (number of pet)
  18. Weight
  19. Height
  20. Body mass index
  21. Absenteeism time in hours (target)

#### RELEVANT PAPERS
Martiniano, A., Ferreira, R. P., Sassi, R. J., & Affonso, C. (2012). Application of a neuro fuzzy network in prediction of absenteeism at work. In Information Systems and Technologies (CISTI), 7th Iberian Conference on (pp. 1-4). IEEE.

#### CITATION REQUEST
Martiniano, A., Ferreira, R. P., Sassi, R. J., & Affonso, C. (2012). Application of a neuro fuzzy network in prediction of absenteeism at work. In Information Systems and Technologies (CISTI), 7th Iberian Conference on (pp. 1-4). IEEE.

#### ACKNOWLEDGEMENTS
- Professor Gary Johns for contributing to the selection of relevant research attributes.
- Professor Emeritus of Management
- Honorary Concordia University Research Chair in Management
- John Molson School of Business
- Concordia University
- Montreal, Quebec, Canada
- Adjunct Professor, OB/HR Division
- Sauder School of Business,
- University of British Columbia
- Vancouver, British Columbia, Canada

## PROBLEM FRAMING
Which one is the best model from machine learning algorithms for classifying and predicting the number of employees absent in the workplace workdays? 

Formulations can improvise by evaluating each model using a set of training data.

## MACHINE LEARNING MODEL
In basic terms, Machine Learning (ML) is the process of training a piece of software, called a model, to make useful predictions using a data set. This predictive model can then serve up predictions about previously unseen data. We use these predictions to take action in a product; for example, the system predicts that a user will like a certain video, so the system recommends that video to the user.

Often, people talk about ML as having two paradigms, supervised, and unsupervised learning. For this repo, the paradigms of supervised learning use to modeling the Absenteeism at work Kaggle dataset.

### SUPERVISED LEARNING
A supervised learning algorithm takes labeled data and creates a model that can make predictions given new data.

These can be either a classification problem or a regression problem. In a classification problem, there might be test data consisting of photos of animals, each one labeled with its corresponding name. The model would be trained on this test data and then the model would be used to classify unlabeled animal photos with the correct name. In a regression problem, there is a relationship trying to be determined among many different variables. Usually, this takes place in the form of historical data being used to predict future quantities. An example of this would be predicting the future price of a stock based on past prices movements.

Following are some of the supervised modeling algorithms used in this repository.

#### DECISION TREE (CATEGORICAL TARGET VARIABLE)
A decision tree is a support tool with a tree-like structure that models probable outcomes, cost of resources, utilities, and possible consequences. Decision trees provide a way to present algorithms with conditional control statements. They include branches that represent decision-making steps that can lead to a favorable result.

A categorical variable decision tree includes categorical target variables that are divide into categories. For example, the categoric of data can be yes or no. That means that every stage of the decision process falls into one of them, and there are no in-betweens.

#### K-NEAREST NEIGHBORS (CATEGORICAL TARGET VARIABLE)
k-Nearest Neighbors (k-NN) is an algorithm that is useful for making classifications/predictions when there are potential non-linear boundaries separating classes or values of interest. Conceptually, k-NN examines the classes/values of the points around it (i.e., its neighbors). To determine the value of the significantly. The majority or average value will assign to interesting that point.

k-NN classification uses when predicting categorical outcomes, and k-NN regression when predicting continuous data.

#### RANDOM FOREST (CATEGORICAL TARGET VARIABLE)
Random Forest is a robust machine learning algorithms uses for a variety of tasks includes regression and classification. It is an ensemble method, meaning that a random forest model can improve the number of small decision trees, called estimators, which each produce their predictions. The random forest model combines the result of the estimators to pursue more accurately.

Random forests are very good for classification problems but are slightly less good at regression problems. In contrast to linear regression, a random forest regressor can't find the result of predict outside the range of its training data.

Random forests are also black boxes: in contrast to some more traditional machine learning algorithms, it is difficult to look inside a random forest classifier and understand what reason behind its decisions. Besides, they can be slow to train and run the models and produce large file sizes.

#### SUPPORT VECTOR MACHINE
Support Vector Machines (SVM) is a state-of-the-art algorithm with strong theoretical foundations based on the Vapnik-Chervonenkis theory. SVM has strong regularization properties. Regularization refers to the generalization of the model to new data.

##### SVM Classification 
SVM classification is base on the concept of decision planes that define decision boundaries. A decision plane is one that separates between a set of objects having different class memberships. SVM finds vectors ("support vectors") specifying a separator that provides a wide range of class separations.

SVM classification supports both binary and multiclass targets.

##### SVM Regression
SVM uses an epsilon-insensitive loss function to solve regression problems.

SVM regression tries to find a continuous function such that the maximum number of data points lie within the epsilon-wide insensitivity tube. Predictions falling within epsilon distance of the right target value not interpret as errors.

References 
- https://developers.google.com/machine-learning/problem-framing/cases
- https://medium.com/datadriveninvestor/what-is-machine-learning-55028d8bdd53
- https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning#introduction
- https://corporatefinanceinstitute.com/resources/knowledge/other/decision-tree/#:~:text=Categorical%20variable%20decision%20tree,there%20are%20no%20in%2Dbetweens.
- https://quantdev.ssri.psu.edu/sites/qdev/files/kNN_tutorial.html
- https://deepai.org/machine-learning-glossary-and-terms/random-forest
- https://docs.oracle.com/cd/E11882_01/datamine.112/e16808/algo_svm.htm#DMCON027

## RESULTS
After modeling with various parameters and models, it can conclude that the model to achieve the best prediction is to use the Random Forest model with categorical variables. Not recommended required for continuous data.

The results obtained from modeling using supervised learning
    
#### RANDOM FOREST (CATEGORICAL TARGET VARIABLE)
Training Data Accuracies

    Random Forest                     98.99%
    Random Forest Parameter Tunning   95.73%
    Random Forest Feature Selection   94.73%
    Random Forest - PCA               95.02%
    Random Forest - LDA               95.16% 

Test Data Accuracies

    Random Forest                     93.82%
    Random Forest Parameter Tunning   89.19%
    Random Forest Feature Selection   94.73%
    Random Forest - PCA               94.73%
    Random Forest - LDA               94.86% 
    
#### SUPPORT VECTOR MACHINE
##### SVM Classification 
Training Data Accuracies
    
    SVM-LINEAR      94.054%
    SVM-POLYNOMIAL  94.054%
    
Test Data Accuracies

    SVM-LINEAR      93.784% 
    SVM-POLYNOMIAL  94.054%
    
##### SVM Regression (Not recommended)
Training Data Accuracies
    
    SVM-LINEAR      44.865% 
    SVM-POLYNOMIAL  44.595%
    
Test Data Accuracies

    SVM-LINEAR      44.865%  
    SVM-POLYNOMIAL  44.595% 
    
#### DECISION TREE (CATEGORICAL TARGET VARIABLE)
Training Data Accuracies

    Decision tree - variable                     90.27%
    Training Decision tree - features selection  90.99%  
    Training Decision tree - scaling             90.54%
    Decision tree - PCA                          87.84%  

Test Data Accuracies

    Decision tree - variable                     5.94%
    Training Decision tree - features selection  6.08% 
    Training Decision tree - scaling             5.95%
    Decision tree - PCA                          5.95%  

#### K-NEAREST NEIGHBORS (CATEGORICAL TARGET VARIABLE)
Training Data Accuracies
    
    K Neighbors Categorical Target Variable      90.54%
  
Test Data Accuracies

    K Neighbors Categorical Target Variable       5.94%
    
