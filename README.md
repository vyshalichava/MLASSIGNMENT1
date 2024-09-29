In this Project, I Implementated various machine learning models (both regression and classification) to compare their performance across different datasets and feature sets. 
Specifically, the task involves:

**Regression Models:**
Implementing multiple linear regression models.
Comparing models with and without regularization.
Evaluating each model's performance using the Mean Squared Error (MSE).

**Classification Models:**
Implementing multiple classification algorithms like Logistic Regression, Decision Trees, and Support Vector Machines (SVM).
Evaluating the models based on their classification accuracy.

**Model Evaluation:**
Using separate evaluation scripts (such as eval_classifiers1.py) to test the performance of different classifiers.
Comparing and analyzing results for better understanding of each model’s capabilities.

So, The following files describe their part in the project:

**1. Regression Models**

**File: LinearRegression.py**
This file sets up and trains a linear regression model on a dataset.
It evaluates the model’s performance by calculating the Mean Squared Error (MSE) on the test set.
The key metrics (such as MSE) provide insights into the accuracy of predictions by the regression models.

**File: train_regression1.py, train_regression2.py, train_regression3.py, train_regression4.py**
These scripts are designed to train the corresponding regression models.
After training, it saves the trained models for further evaluation or use.

**File: linearwithreg.py**
This script compares the performance of regularized versus non-regularized linear regression models.
Regularization helps in avoiding overfitting by penalizing large model weights, thereby improving generalization.

**File: MutipleLinearRegression.py**
This file extends the regression task to handle multiple output variables.
The script trains a model to predict more than one target variable simultaneously, providing a more complex and comprehensive analysis of the data.

**File: eval_regression1.py, eval_regression2.py, eval_regression3.py, eval_regression4.py**
These scripts evaluate the performance of the corresponding regression model on the test set.
It outputs the MSE to assess how well the model generalizes to unseen data.

**2. Classification Models**
   
**File: classification.py**
This script trains and evaluates different classifiers (Logistic Regression, Decision Trees, and SVM).
Each classifier is trained on the dataset, and its accuracy is measured, allowing you to compare how well each model performs in classification tasks.

**File: logisticRegression.py**
This script implements the Logistic Regression model, a popular classification algorithm.
It trains the model on specific features (like petal or sepal dimensions in the Iris dataset) and calculates the classification accuracy on the test set.

**File: linearDiscriminantAnalysis.py**
This file implements Linear Discriminant Analysis (LDA), a classification technique.
LDA is useful when the data points are linearly separable, and it’s known to perform well on small datasets with clear distinctions between classes.

**File: eval_classifiers1.py, eval_classifiers2.py, eval_classifiers3.py, eval_classifiers4.py**
These scripts evaluate the performance of different classification models.
They run the classifiers (trained earlier in classification.py) on the test set and outputs accuracy metrics to compare their effectiveness.
Since there are multiple evaluation scripts (e.g., eval_classifiers1.py, eval_classifiers2.py, etc.), each of them tests different classifiers or feature sets.

**Key Points of My Solution:**
1. Model Variety: I implemented a wide range of models, including linear regression, logistic regression, decision trees, SVM, and LDA, to effectively compare their performance on both regression and classification tasks.

2. Regularization Expertise: I applied regularization techniques to the regression models, ensuring reduced overfitting and improved generalization for better performance on unseen data.

3. Modular Model Comparison: By splitting the evaluation scripts (e.g., eval_classifiers1.py), I modularized the testing process, allowing for focused and clear comparisons between different models and feature sets.

4. Comprehensive Evaluation: My approach not only involved training the models but also thoroughly evaluating them using metrics like MSE for regression and accuracy for classification, ensuring robust testing and validation of all models.

**Outputs:**
Run LinearRegression.py file to begin. Use the following command:
               **python LinearRegression.py**
It should compile without any errors. There are seperate training scripts for each model named train_regression1.py for model1, train_regression2.py for model2 etc. This will create and save models and also generates an image named training_loss.png.
Output:
Model 1 - Test Mean Squared Error: 0.1548 
Model 2 - Test Mean Squared Error: 0.0480 
Model 3 - Test Mean Squared Error: 0.0688 
Model 4 - Test Mean Squared Error: 0.0493

Training:
Below are the Training loss plots generated for each model.

<img width="230" alt="image" src="https://github.com/user-attachments/assets/058e4d0a-7b91-484b-b73e-8b6ba6242f89">

<img width="230" alt="image" src="https://github.com/user-attachments/assets/08056077-5b3e-4800-99e8-365be4388602">

<img width="230" alt="image" src="https://github.com/user-attachments/assets/5f6afd25-8c8a-41da-9fae-5f62fcabf8eb">

<img width="230" alt="image" src="https://github.com/user-attachments/assets/df906285-ffa2-4ef5-aeab-abb597548cbc">


Next, run the following command to see the difference between regularized and non-regularized models.
              **python linearwithreg.py**
Testing:
Model1: uses input features[0,1] which is sepal length and sepal width.
Model2: uses input features[2,3] which is petal length and petal width.
Model3: uses input features[0,1] which is sepal length, sepal width and petal length. 
Model4: uses input features[0,1] which is sepal length, sepal width, petal length, petal width.

<img width="418" alt="image" src="https://github.com/user-attachments/assets/6ac3d416-4a4d-4df5-9904-ea726ebe11ce">

From the above output we can see that model4 has least mse. So, using all input features is most predictive of their corresponding output features.

**Regression with multiple outputs:**
Run MultipleLinearRegression.py file using the following command:
              **python MultipleLinearRegression.py**
This generates a (multiple_outputs_model_model) file along with a training loss graph(multiple_output_model_training_loss).

Output:
multiple_outputs_model - Mean Squared Error on Test Set: 0.7029

Graph:

<img width="230" alt="image" src="https://github.com/user-attachments/assets/f4084628-6790-488f-9157-27cc28a6d819">

**2. Classification:**
Run the following command:
             **python classification.py**
             
Outputs Accuracy along with the visualizations for different classifiers:
Accuracy with Logistic Regression features using Logistic Regression: 1.00 
Accuracy with Decision Tree features using Decision Tree: 1.00
Accuracy with SVM features using SVM: 1.00
Accuracy with Logistic Regression features using Logistic Regression: 0.82 
Accuracy with Decision Tree features using Decision Tree: 0.67
Accuracy with SVM features using SVM: 0.80

<img width="571" alt="image" src="https://github.com/user-attachments/assets/976c77f9-18a5-4049-b74f-10b99cb4f281">

Accuracy with Logistic Regression features using Logistic Regression: 1.00 
Accuracy with Decision Tree features using Decision Tree: 1.00
Accuracy with SVM features using SVM: 1.00

**Logistic Regression**
Run the following command:
            **python logisticregression.py**
It gives the following output:
Logistic regression Petal - Classification accuracy on testset: 0.96

**Linear Discriminant Analysis**
Run the following command
            **python linearDiscriminantAnalysis.py**
It gives the following output:
Linear Discriminant Analysis - Classification acc on Test Set: 1.00

**Testing:**
Testing for the above classifiers can be done using the eval_classifiers1.py, eval_classifiers2.py, eval_classifiers3.py scripts.
Example to run eval_classifiers1.py:
           **python eval_classifiers1.py**
           
Output:
Linear Discriminant Analysis - Classification acc on Test Set: 1.00
Logistic regression Petal - Classification accuracy on testset: 0.96
Logistic Regression (Petal Length/Width) - Classification accuracy on testset: 0.96
Linear Discriminant Analysis (Petal Length/Width) - Classification accuracy on testset: 0.87

The last two values are main, which gives information about the first variant for both logistic regression and linear discriminant analysis.

**Conclusion:**
In this project, I addressed the problem of comparing the performance of various machine learning models across different datasets and feature sets, focusing on both regression and classification tasks. I implemented multiple linear regression models, with and without regularization, to predict outcomes and evaluated their effectiveness using Mean Squared Error (MSE). Additionally, I applied several classification algorithms, including Logistic Regression, Decision Trees, SVM, and LDA, assessing their performance based on classification accuracy. My approach is efficient because it not only utilizes a diverse set of models tailored to different problem types but also incorporates regularization to improve generalization. The modular structure of the evaluation scripts allows for focused comparison of model performance, making it easier to identify the best model for specific tasks while ensuring robustness and scalability.

