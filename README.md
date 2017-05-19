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

In nl_xgb.py:

