There are two python files...Network_locum.py and nl_xgb.py

In Network_locum.py:
The 3 data sets (sessions, ccgs and practices are read as pandas dataframes
The first part of the code merges datasets using ccg_id and practice_id
Calculates average hourly rate
Calculates fill_rate
Remove NA rows
Ansers the question: "does the fill rate go up as the average hourly rate goes up?"
Shows the correlation or lack of between ccg_id and the fill rate
A plot of ccg_id versus fill rate shows the variation of fill rate for all ccg_id
Calculates perason's r coefficient which shows a weak positive correlation between average hourly rate and fill rate

In nl_xgb.py is the machine learning code for predictions
I have focused on probability of a job being completed but it can be modified to predict probability of a job not being completed
The code computes the classification report for the predictions. 
Precision measures the number of identified coompleted that are actually completed
Recall measures the number of real completed that have been identified by the prediction
Computes Matthews correlation coefficient which indicates the strength of the predictions
Finally, the code shows the importance of the features used to the effectiveness of the predictions

All codes show what libraries are needed to run the code
You can pip install these libraries in your command prompt 
Remember to cd to the python directory

