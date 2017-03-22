
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
        self.dropout_d = 0

    def initial_state(self):
        return self.builder.initial_state()

    def set_dropout(self, d):
        self.dropout_d = d
        self.enable_dropout()

    def disable_dropout(self):
        self.builder.disable_dropout()

    def enable_dropout(self):
        self.builder.set_dropout(self.dropout_d)
