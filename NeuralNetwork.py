

import numpy 
import scipy.special

class neuralNetwork:
  
  def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
    self.inodes = inputnodes
    self.hnodes = hiddennodes
    self.onodes = outputnodes
    # коффициент обучения
    self.lr = learningrate
    # Матрицы весовых коэффициентов связей wih (между входным и скрытым слоями)и who (между скрытыми и выходными слоями). 
    # Весовые коэффициенты связей между узлом i и j следующего слоя обозначены как w_i_j:
    # w11 w21
    # w12 w22 и т.д.
    self.wih = numpy.random.normal(0.0, pow(self.hnodes, - 0.5), (self.hnodes, self.inodes))
    self.who = numpy.random.normal(0.0, pow(self.onodes, - 0.5), (self.onodes, self.hnodes))
    
    #Функсия активации (сигмойда)
    self.activation_function = lambda x: scipy.special.expit(x)
    pass

  def train(self, inputs_list, targets_list):
    inputs = numpy.array(inputs_list, ndmin=2).T
    targets = numpy.array(targets_list, ndmin=2).T

    hidden_inputs = numpy.dot(self.wih, inputs)
    hidden_outputs = self.activation_function(hidden_inputs)

    final_inputs =numpy.dot(self.who, hidden_outputs)
    final_outputs = self.activation_function(final_inputs)

    output_errors = targets - final_outputs
    hidden_errors = numpy.dot(self.who.T, output_errors)

    self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transponse(hidden_inputs))
    
    self.wih = self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transponse(inputs))

    pass
  
  def query(self, inputs_list):
    
    #Преобразование входных данных в двухмерный массив
    inputs = numpy.array(inputs_list, ndmin=2).T
    
    hidden_inputs = numpy.dot(self.wih, inputs)
    #Применение к сигналам сглаженым весовыми коэффициентами функции активации.
    hidden_outputs = self.activation_function(hidden_inputs)
    
    final_inputs = numpy.dot(self.who, hidden_outputs)

    final_outputs = self.activation_function(final_inputs)
    
    return final_outputs
    
    pass

# количество входных, скрытых и выходных данных
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
# коэффициент обучения равен 0.3
learning_rate = 0.3
# создать экземпляр нейронной сети
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print(n.query([0.8543, 12.355, 4.20123]))

print(numpy.random.normal(0.0, pow(hidden_nodes, - 0.5), 
(hidden_nodes, input_nodes))) #Ok