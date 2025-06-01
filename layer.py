class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # Use : return the op
        pass

    def backward(self, output_gradient, learning_rate):
        # Use : update parameters(weight, bias) and return input_gradient to next layer
        pass
