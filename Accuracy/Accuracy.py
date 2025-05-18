import json
import pandas as pd 
import time 
import matplotlib.pyplot as plt
import numpy as np


#Gets the Accuracy of the model for each activity label
def Get_Variability(CorrectAnswer_Json, TotalQuestion_Json,Accuracy_Dict):
  MiscCorrect = 0
  MiscTotal = 0
  with open(CorrectAnswer_Json, 'r') as file:
    CorrectAnswer = json.load(file)
  with open(TotalQuestion_Json, 'r') as file:
    TotalQuestions = json.load(file) 

 
  #Counts the total number of examples for each activity label
  for index,key in enumerate(CorrectAnswer):
    #If the number of examples is less than 50, add the numbers to the Misc category
    if(TotalQuestions[key] < 100):
      MiscCorrect += int(CorrectAnswer[key])
      MiscTotal += int(TotalQuestions[key])
    else:
      #Gets the accuracy of the model for each activity label > 50
      Accuracy_Dict[key] = int(CorrectAnswer[key])/int(TotalQuestions[key])
  Accuracy_Dict['Misc'] = MiscCorrect/MiscTotal
  

def Get_TotalAccuracy(CorrectAnswer_Json):
    with open(CorrectAnswer_Json, 'r') as file:
      CorrectAnswer = json.load(file)

    TotalCorrect = sum(CorrectAnswer.values())
    print("Total Correct: ", TotalCorrect)
    TotalAccurary = TotalCorrect/10000
    print("Total Accuracy: ", TotalAccurary)
    return TotalAccurary

  
#Plot the average accuracy of the model for each activity label
def PlotAverageAccuracy(Accuracy1b, Accuracy4b, Accuracy12b):
  # Create a DataFrame from the data
  data = {
      'Model': ['1b', '4b', '12b'],
      'Accuracy': [Accuracy1b, Accuracy4b, Accuracy12b]
  }
  df = pd.DataFrame(data)

  # Plot the data
  plt.figure(figsize=(10, 6))
  plt.bar(df['Model'], df['Accuracy'], color=['blue', 'orange', 'green'])
  plt.xlabel('Model')
  plt.ylabel('Accuracy')
  plt.title('Average Accuracy of Models')
  plt.ylim(0, 1)
  plt.grid(axis='y')
  # Show the plot
  plt.savefig("png/Averageaccuracy_plot.png")


def PlotVaribility(Dict1,Dict2,Dict3):
  data ={
    'Questions': Dict1.keys(),
    'Test1': Dict1.values(),
    'Test2':Dict2.values(),
    'Test3':Dict3.values()
  }
  df = pd.DataFrame(data)
  x = np.arange(len(df))  # the label locations
  width = 0.25  # the width of the bars

  # Plotting
  plt.figure(figsize=(18, 8))
  plt.bar(x - width, df['Test1'], width, label='1b', color='blue')
  plt.bar(x, df['Test2'], width, label='4b', color='orange')
  plt.bar(x + width, df['Test3'], width, label='12b', color='green')

  # Labels and Title
  plt.xlabel('Question')
  plt.ylabel('Accuracy')
  plt.title('Accuracy per Question by Test and Model')
  plt.xticks(x, df['Questions'], rotation=90)
  plt.ylim(0, 1)
  plt.legend()
  plt.grid(axis='y')
  plt.tight_layout()
  print(df)

  plt.savefig("png/model_accuracy_questions.png")
if __name__ == "__main__":
  CorrectFiles= [
    '12b_Test1_CorrectAnswer.json',
    '12b_Test2_CorrectAnswer.json',
    '12b_Test3_CorrectAnswer.json',
    '1b_Test1_CorrectAnswer.json',
    '1b_Test2_CorrectAnswer.json',
    '1b_Test3_CorrectAnswer.json',
    '4b_Test1_CorrectAnswer.json',
    '4b_Test2_CorrectAnswer.json',
    '4b_Test3_CorrectAnswer.json'
    ]
  TotalFiles= [
    '12b_Test1_TotalAnswer.json',
    '12b_Test2_TotalAnswer.json',
    '12b_Test3_TotalAnswer.json',
    '1b_Test1_TotalAnswer.json',
    '1b_Test2_TotalAnswer.json',
    '1b_Test3_TotalAnswer.json',
    '4b_Test1_TotalAnswer.json',
    '4b_Test2_TotalAnswer.json',
    '4b_Test3_TotalAnswer.json'
    ] 

  Accuracy_Dict1 = {}
  Accuracy_Dict2 = {}
  Accuracy_Dict3 = {}
  
  #Plots The Average Accuracy 
  #+=======================================================================#
  """  
  Accuracy12b = 0
  Accuracy1b = 0
  Accuracy4b = 0
  for i in CorrectFiles:
    if '1b' in i:
      Accuracy1b += Get_TotalAccuracy(i)
    elif '4b' in i:
      Accuracy4b += Get_TotalAccuracy(i)
    elif '12b' in i:
      Accuracy12b += Get_TotalAccuracy(i)
  PlotAverageAccuracy(Accuracy1b/3, Accuracy4b/3, Accuracy12b/3) """
  #=========================================================================#
  '''
    # Load the JSON files
  Get_Variability('1b_Test1_CorrectAnswer.json', '1b_Test1_TotalAnswer.json', Accuracy_Dict1)
  Get_Variability('4b_Test2_CorrectAnswer.json', '4b_Test2_TotalAnswer.json', Accuracy_Dict2)
  Get_Variability('12b_Test3_CorrectAnswer.json', '12b_Test3_TotalAnswer.json', Accuracy_Dict3)
  PlotVaribility(Accuracy_Dict1,Accuracy_Dict2,Accuracy_Dict3)
  '''

  


  #CheckConsistency(Accuracy_Dict1,Accuracy_Dict2,Accuracy_Dict3)
  
