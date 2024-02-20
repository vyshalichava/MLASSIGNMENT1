
 Run LinearRegression.py file to begin. Use the following command:
Python LinearRegression.py
It should compile without any errors. There are seperate training scripts for each model named train_regression1.py for model1, train_regression2.py for model2 etc. This will create and save models.
Output:
Model 1 - Test Mean Squared Error: 0.1548
Model 2 - Test Mean Squared Error: 0.0480
Model 3 - Test Mean Squared Error: 0.0688 
Model 4 - Test Mean Squared Error: 0.0493

 This generates an image named training_loss.png.
  
  Next, run the following command to see the difference between regularized and non-regularized models.
python linearwithreg.py

Model1: uses input features[0,1] which is sepal length and sepal width.
Model2: uses input features[2,3] which is petal length and petal width.
Model3: uses input features[0,1] which is sepal length, sepal width and petal length. Model4: uses input features[0,1] which is sepal length, sepal width, petal length, petal width.

  From the above output we can see that model4 has least mse. So, using all input features is most predictive of their corresponding output features.
1.7 Regression with multiple outputs:
Run MultipleLinearRegression.py file using the following command:
python MultipleLinearRegression.py
This generates a (multiple_outputs_model_model) file along with a training loss graph(multiple_output_model_training_loss).
Output:
multiple_outputs_model - Mean Squared Error on Test Set: 0.7029
Graph:

  2. Classification:
Run the following command:
python classification.py
Outputs Accuracy along with the visualizations for different classifiers. Accuracy with Logistic Regression features using Logistic Regression: 1.00 Accuracy with Decision Tree features using Decision Tree: 1.00
Accuracy with SVM features using SVM: 1.00
Accuracy with Logistic Regression features using Logistic Regression: 0.82 Accuracy with Decision Tree features using Decision Tree: 0.67
Accuracy with SVM features using SVM: 0.80
 (click close)

 Accuracy with Logistic Regression features using Logistic Regression: 1.00 Accuracy with Decision Tree features using Decision Tree: 1.00
Accuracy with SVM features using SVM: 1.00
2.1 Logistic Regression
Run the following command:
python logisticregression.py
It gives the following output:
Logistic regression Petal - Classification accuracy on testset: 0.96
2.2 Linear Discriminant Analysis
Run the following command
python linearDiscriminantAnalysis.py
It gives the following output:
Linear Discriminant Analysis - Classification acc on Test Set: 1.00
2.3 Testing:
Testing for the above classifiers can be done using the eval_classifiers1.py, eval_classifiers2.py, eval_classifiers3.py scripts.
Example to run eval_classifiers1.py:
python eval_classifiers1.py
Output:
Linear Discriminant Analysis - Classification acc on Test Set: 1.00
Logistic regression Petal - Classification accuracy on testset: 0.96
Logistic Regression (Petal Length/Width) - Classification accuracy on testset: 0.96
Linear Discriminant Analysis (Petal Length/Width) - Classification accuracy on testset: 0.87
The last two values are main, which gives information about the first variant for both logistic regression and linear discriminant analysis.
