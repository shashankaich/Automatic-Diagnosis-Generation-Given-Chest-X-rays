# Automatic-Diagnosis-Generation-Given-Chest-X-rays

## Business Problem
Generating Automatic Diagnosis from X-Ray's is a problem that many researchers are trying to solve. 
This problem if solved, will help the entire Medical fraternity.

In this case study we will give a humble try to solve this problem.

## Understanding the Data:
In the Case Study, we have the Chest X Rays and Reports. Our Goal is to generate the final diagnosis or Impressions.
The Reports for each patient contain two sections
1. The findings section: the Radiologist lists the radiology observations regarding each area of the body examined in the imaging study.
2. The impression section: the Radiologist/Doctor provides a diagnosis.
Our task is to generate the impression given X-Ray Images.
The Data is Collected from Indiana University - Chest X-Rays
Images: http://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d
Reports: http://academictorrents.com/details/66450ba52ba3f83 f82ef9c91f2bde0e845aba9
The Data set has two parts: Images and Reports.
The Reports have all the Information related to a patient.
In this case study we will try to generate the Final Diagnosis/ Impressions given the X-Rays

## Converting Real World Problem to ML Problem:
Now we will try to pose a Machine Learning Problem from the given Data.
Our objective is to generate the Impression from the X-Rays.
The Input to the Model will be 2 Images (Lateral and Front X-Rays) and the Model should generate the Impression.
Hence in this case we will have to use a Generative Model.
The Model should not take very long tome to generate the Impression but taking a few seconds if perfectly OK.
Since we have a Generative Model we will use BLEU Score as the KPI.

## Conclusion:
In this Case Study we tried to generate the Impression/Diagnosis from the Chest X-Rays.
We cleaned the Impression/Diagnosis from the medical reports
Next, we trained a deep learning Model for the Impression Generation
The BLEU Score that we got from the Model is around 0.57
Finally we created a pipeline for Generating the Impressions

## The Colab link for the notebooks are:

EDA_Notebook: https://colab.research.google.com/drive/1w1-0VLdMWVFXVxrYbti3XB3-485b-SZ0#scrollTo=p4YQpFzhx4tC
Data_Processing_Notebook: https://colab.research.google.com/drive/1FiOKOJ3rhOpx5IYu3ZdQ5_Lg9pgFDJ-e#scrollTo=LltFb99VTB0o
Tensorflow_Model_Notebook: https://colab.research.google.com/drive/11rjzlyFuW5PCeslzzbg6nVZJ40Xe9ypT#scrollTo=8gifbJhdkZRc
Final_Notebook: https://colab.research.google.com/drive/12PpclSeLJHjjEfKnbOIG4RKa8TWjq_Je#scrollTo=Bb9L6b0lpXXo
Pipeline: https://colab.research.google.com/drive/1mb-LjLsVnhlrhzkwQTxwWdXOV3Y_Kb78#scrollTo=yv5gXBDu3ePq
