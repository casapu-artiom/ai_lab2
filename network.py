from math import exp, sqrt
from random import random

__author__ = 'Artiom.Casapu'

#learning rate
learning_rate = 0.005

#function to be tested
f = lambda x: 0.5 * x + 4

min_x = 1
max_x = 1000
min_y = 1
max_y = 1000

class NeuralNetwork:

    class Neuron:

        def __init__(self, ninput=0, weights=None):
            self.ninput = ninput
            self.weights = weights
            if (weights is None):
                self.weights = []
                for i in range(ninput):
                    self.weights.append(random())
            self.output = None

        def calc_output(self, inputs):

            if (self.weights is None):
                self.output = 0
            else:
                self.output = self.activation(sum(map(lambda x, y: x * y, inputs, self.weights)))

        def get_output(self):
            return self.output

        def set_weights(self, weights):
            self.weights = weights

        def activation(self, n):
            return self.sigma(n)

        def set_error(self, error):
            self.error = error

        def get_error(self):
            return self.error

        def sigma(self, x):
            return x
            #return 1.0 / (1.0 + exp(-x))

        def set_output(self, output):
            self.output = output

        def remove_output(self):
            self.output = None

        def get_weight(self, i):
            return self.weights[i]

        def set_weight(self, i, value):
            self.weights[i] = value

    class Layer:
        def __init__(self, nNeurons, nInputs):
            self.neurons = []
            for i in range(nNeurons):
                self.neurons.append(NeuralNetwork.Neuron(nInputs, None))

        def calc_outputs(self, inputs):
            outputs = []
            for i in range(len(self.neurons)):
                self.neurons[i].calc_output(inputs)
                outputs.append(self.neurons[i].get_output())
            return outputs

    def __init__(self, ninputs, nhiddenlayers, noutputs, nhiddenlayersize):
        self.layers = []

        self.layers.append(NeuralNetwork.Layer(ninputs, 0))

        if (nhiddenlayers == 0):
            self.layers.append(NeuralNetwork.Layer(noutputs, ninputs))

        else:
            self.layers.append(NeuralNetwork.Layer(nhiddenlayersize, ninputs))

            for i in range(nhiddenlayers - 1):
                self.layers.append(NeuralNetwork.Layer(nhiddenlayersize, nhiddenlayersize))

            self.layers.append(NeuralNetwork.Layer(noutputs, nhiddenlayersize))

    def calculate_output(self, input):

        outputs = []

        for i in range(len(input)):
            self.layers[0].neurons[i].set_output(input[i])
            outputs.append(input[i])

        for i in range(len(self.layers) - 1):
            outputs = self.layers[i+1].calc_outputs(outputs)

        return outputs

    """
            learning(double inData[][], double outData[][]){
            stopCondition = false;
            epoch = 0;
             globalErr = [];
             while ((!stopCondition) || (epoch < EPOCH_LIMIT)){
             double globalErr = [];
             //for each training example
             for(d = 0; d < inData.size(); d++){
             activate(inData[d]); //activate all the neurons; propagate forward the
            //signal
             double err = []; //backpropagate the error of neurons from the
            //output layer
             globalErr[d] = errorComputationRegression(outData[d], err);
             errorsBackPropagate(err);
             } //for d
             stopCondition = checkGlobalErr(globalErr);
             epoch++;
             }//while
            } //learning

    """
    def learn(self, input_data, output_data, epoch_limit = 100):
        stop_condition = False
        epoch = 0
        globalErr = [0 for i in range(len(input_data))]
        while epoch < epoch_limit and not stop_condition:
            for d in range(len(input_data)):
                output = self.calculate_output(input_data[d])
                err = [0 for i in range(len(output_data[d]))]
                globalErr[d] = self.calc_global_error(output_data[d], err)
                self.error_backpropagate(err)
                stop_condition = self.checkGlobalErr(globalErr)
            epoch += 1


    """
    double globalErr = [];
     for(d = 0; d < inData.size(); d++){ //for each testing example
     activate(inData[d]); //activate all the neurons; propagate forward the
    //signal
     double err = []; //compute the error of neurons from the output
    //layer
     globalErr[d] = errorComputationRegression(outData[d], err);
     }
    """
    def test(self, input_data, output_data):
        correct, fail = 0, 0
        for d in range(len(input_data)):
            output = self.calculate_output(input_data[d])
            print output, output_data[d]
            if (output[0] < 0.0 and output_data[d][0] < 0.0):
                correct += 1
            elif (output[0] > 0.0 and output_data[d][0] > 0.0):
                correct += 1
            else:
                fail += 1
        return correct / (correct + fail + 0.0)

    """
    errorsBackpropagate(double err[]){
        for(l = noHiddenLayers + 1, l >= 1; l--) {
            i = 0;
            for each neuron n1 of layers[l] {
                if (l == noHiddenLayers + 1)
                    n1.setErr(err[i]);
                else {
                    sumErr = 0.0;
                    for each neuron n2 of layers[l + 1]
                        sumErr += n2.getWeight(i) * n2.getErr();
                    n1.setErr(sumErr);
                }//if
                for(j=0; j < n1.getNoInputs(); j++) {
                    netWeight = n1.getWeight(j)+LEAR_RATE*n1.getErr() *
                        layers[l-1].getNeuron(j).getOutput();
                    n1.setWeight(j, netWeight);
                }//for j
                i++;
            } //for neuron n1
        } //for l
    } //errorbackpropagate

    """

    def error_backpropagate(self, error):
        global learning_rate

        for i in range(len(error)):
            self.layers[-1].neurons[i].set_error(error[i])

            for j in range(len(self.layers[-1].neurons)):
                net_weight = self.layers[-1].neurons[j].get_weight(j) + learning_rate * self.layers[-1].neurons[j].get_error() * \
                    self.layers[-1].neurons[j].get_output()
                self.layers[-1].neurons[j].set_weight(j, net_weight)

        for l in range(len(self.layers)-1, 1, -1):
            #current -> self.layers[l-1]

            for i in range(len(self.layers[l-1].neurons)):

                sumErr = 0.0
                for j in range(len(self.layers[l].neurons)):
                    sumErr += self.layers[l].neurons[j].get_weight(i) * self.layers[l].neurons[j].get_error()

                self.layers[l-1].neurons[i].set_error(sumErr)

                for j in range(self.layers[l-1].neurons[i].ninput):

                    net_weight = self.layers[l-1].neurons[i].get_weight(j) + \
                                 self.layers[l-2].neurons[j].get_output() * self.layers[l-1].neurons[i].get_error() * learning_rate

                    self.layers[l-1].neurons[i].set_weight(j, net_weight)


    """
    double errorComputationRegression(double target[noOutputs], double err[noOutputs]){
        globalErr = 0.0;
        for(i = 0; i < layers[noHiddenLayers + 1].getNoNeurons(); i++){
            err[i] = target[i] - layers[noHiddenLayers + 1].getNeuron(i).getOutput();
            globalErr += err[i] * err[i];
        }
        return globalErr;
        }
    """
    def calc_global_error(self, target, err):
        globalErr = 0.0
        for i in range(len(self.layers[-1].neurons)):
            err[i] = target[i] - self.layers[-1].neurons[i].get_output()
            globalErr += err[i] * err[i]
        return globalErr
    """
    bool checkGlobalErr(double err[]){
         //regression
        error = 0.0;
         for( i = 0; i < err.size(); i++)
            error += err[i];
            if (fabs(error - 0.1) < 1.0E-8)
            return true;
            else
        return false;
    }
    """
    def checkGlobalErr(self, err):
        return sum(err) < 0.00001

def point_above_line(x, y):
#    position = sign( (Bx-Ax)*(Y-Ay) - (By-Ay)*(X-Ax) )
    result = x * y - x * f(x)

    if (result <= 0):
        return -1

    return 1

def generate_learning_data(size, filename="learning_data.txt"):
    f = open(filename, mode='w')
    for i in range(size):
        x = random() * (max_x - min_x) + min_x
        y = random() * (max_y - min_y) + min_y
        if (point_above_line(x, y)):
            f.write(str(x) + "," + str(y) + "," + str(point_above_line(x, y)) + "\n")
    f.close()

def parse_data(filename="learning_data.txt"):
    f = open(filename)

    input_data = []
    output_data = []
    for line in f.readlines():
        x,y,z = map(float, line.split(","))
        input_data.append([x, y])
        output_data.append([z])

    return input_data, output_data

def normalize_data(input_data, nfeatures):

    for i in range(nfeatures):

        global_sum = 0.0

        for j in range(len(input_data)):
            global_sum += input_data[j][i]

        global_sum /= len(input_data)

        square_sum = 0

        for j in range(len(input_data)):
            square_sum += (input_data[j][i] - global_sum) ** 2

        deviation = sqrt(square_sum / len(input_data))

        for j in range(len(input_data)):
            input_data[j][i] = (input_data[j][i] - global_sum) / deviation


if __name__ == "__main__":
    #generate_learning_data(1000)
    #generate_learning_data(200, "test_data.txt")
    nn = NeuralNetwork(2, 1, 1, 3)
    input_data, output_data = parse_data()
    normalize_data(input_data, len(input_data[0]))
    nn.learn(input_data, output_data)
    test_input, test_output = parse_data("test_data.txt")
    normalize_data(test_input, len(test_input[0]))
    print nn.test(test_input, test_output)
