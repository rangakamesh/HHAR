import matplotlib.pyplot as plt

def plotModel(train_acc,test_acc,loss,time,model):
  plt.figure(figsize=(20,5))
  
  plt.subplot(131)
  plt.plot(train_acc,label="Training")
  plt.plot(test_acc,label="Testing")
  plt.xlabel("Epochs ->")
  plt.ylabel("Accuracy ->")
  plt.title("Train/Test Accuracy")
  plt.legend()
  
  plt.subplot(132)
  plt.plot(loss)
  plt.xlabel("Epochs ->")
  plt.ylabel("Loss ->")
  plt.title("Loss")
  
  plt.subplot(133)
  plt.plot(time)
  plt.xlabel("Epochs ->")
  plt.ylabel("Time(sec) ->")
  plt.title("Epoch Time")
  
  if(model=='FCNN'):
    plt.suptitle('Fully Connected Neural Network \n')
  else:
    plt.suptitle('Recurrent Neural Network \n')

  plt.show()