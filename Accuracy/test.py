import json
import pandas as pd 
import time 
import matplotlib.pyplot as plt
import numpy as np


def Get_TotalAccuracy(CorrectAnswer_Json):
    with open(CorrectAnswer_Json, 'r') as file:
      CorrectAnswer = json.load(file)

    TotalCorrect = sum(CorrectAnswer.values())
    print("Total Correct: ", TotalCorrect)
    TotalAccurary = TotalCorrect/10000
    print("Total Accuracy: ", TotalAccurary)
    return TotalAccurary


if __name__ == "__main__":
 

  # Manually compute total accuracies
  acc_1b = [
      Get_TotalAccuracy('1b_Test1_CorrectAnswer.json'),
      Get_TotalAccuracy('1b_Test2_CorrectAnswer.json'),
      Get_TotalAccuracy('1b_Test3_CorrectAnswer.json')
  ]
  acc_4b = [
      Get_TotalAccuracy('4b_Test1_CorrectAnswer.json'),
      Get_TotalAccuracy('4b_Test2_CorrectAnswer.json'),
      Get_TotalAccuracy('4b_Test3_CorrectAnswer.json')
  ]
  acc_12b = [
      Get_TotalAccuracy('12b_Test1_CorrectAnswer.json'),
      Get_TotalAccuracy('12b_Test2_CorrectAnswer.json'),
      Get_TotalAccuracy('12b_Test3_CorrectAnswer.json')
  ]

  # Data to plot
  labels = ['1b', '4b', '12b']
  x = np.arange(len(labels))  # [0, 1, 2]
  width = 0.25

  # Plot
  plt.figure(figsize=(10, 6))
  plt.bar(x - width, [acc_1b[0], acc_4b[0], acc_12b[0]], width, label='1b')
  plt.bar(x,         [acc_1b[1], acc_4b[1], acc_12b[1]], width, label='4b')
  plt.bar(x + width, [acc_1b[2], acc_4b[2], acc_12b[2]], width, label='12b')

  # Customize axes
  plt.xticks(x, labels)
  plt.xlabel('Test Cases')
  plt.ylabel('Accuracy')
  plt.title('Accuracy per Test Case for Each Model')
  plt.ylim(0, 1)
  plt.legend()
  plt.grid(axis='y')
  plt.tight_layout()

  # Save and show
  plt.savefig("png/ConsistencyTest.png")
  plt.show()