
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
  labels = ['Test1', 'Test2', 'Test3']
  x = np.arange(len(labels))  # [0, 1, 2]
  width = 0.25

  # Plot
  plt.figure(figsize=(10, 6))
  plt.bar(x - width, acc_1b, width, label='1b', color='blue')
  plt.bar(x, acc_4b, width, label='4b', color='orange')
  plt.bar(x + width, acc_12b, width, label='12b', color='green')

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
  plt.savefig("png/GroupedAccuracyByTest.png")
  plt.show()