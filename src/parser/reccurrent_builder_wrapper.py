
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


class NGramBuilderState:

    def __init__(self, n, entries):
        self.entries = entries
        self.n = n

    def add_input(self, input):
        if len(self.entries) >= self.n:
            new_entries = self.entries[1:]
        else:
            new_entries = self.entries
        new_entries.append(input)
        return NGramBuilderState(self.n, new_entries)

    def output(self):
        if self.n == 0:
            return None
        elif len(self.entries) >= self.n:
            return dy.concatenate(self.entries)
        else:
            new_entries = []
            for i in range(self.n-len(self.entries)):
                new_entries.append(self.entries[0])
            new_entries.extend(self.entries)
            return dy.concatenate(new_entries)

class NGramBuilderNetwork:

    def __init__(self, n, entry_size):
        self.entry_size = entry_size
        self.n = n
        self.hidden_dim = n*entry_size
        self.input_dim = entry_size

    def initial_state(self):
        return NGramBuilderState(self.n, [])

    def set_dropout(self, d):
        pass

    def disable_dropout(self):
        pass

    def enable_dropout(self):
        pass


