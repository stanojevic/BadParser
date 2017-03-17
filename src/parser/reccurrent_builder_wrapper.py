
import dynet as dy

class LSTMBuilderWrapper:

    def __init__(self, layers, input_dim, hidden_dim, model=None, builder=None):
        if model is None and builder is None:
            raise Exception("you are doing wrong")
        if builder is None:
            self.builder = dy.LSTMBuilder(layers, input_dim, hidden_dim, model)
        else:
            self.builder = builder
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def initial_state(self):
        return self.builder.initial_state()
