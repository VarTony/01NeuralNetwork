class NeuralNetwork:
  
  def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
    self.inodes = inputnodes
    self.hnodes = hiddennodes
    self.onodes = outputnodes
    # коффициент обучения
    self.lr = learningrate
    pass

  def train():
    pass
  
  def query():
    pass

# количество входных, скрытых и выходных данных
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
# коэффициент обучения равен 0.3
learning_rate = 0.3
# создать экземпляр нейронной сети
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)