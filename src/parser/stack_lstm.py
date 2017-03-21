import dynet as dy
import numpy as np

class StackLSTM:

    def __init__(self, head, tail, rnn_state):
        self._head = head
        self._tail = tail
        self._rnn_state = rnn_state
        self.vector = rnn_state.output()
        if head is None:
            if tail is not None:
                raise Exception("something is wrong")
            self.size = 0
        else:
            self.size = tail.size + 1

    def push(self, element):
        return StackLSTM(element, self, self._rnn_state.add_input(element.vector))

    def pop(self):
        if self.size == 0:
            raise Exception("popping from empty stack")
        else:
            return self._tail

    def top(self):
        if self.size == 0:
            raise Exception("topping from empty stack")
        else:
            return self._head

    def second_top(self):
        if self.size < 1:
            raise Exception("popping from empty stack")
        else:
            return self._tail._head

    def _recurr_repr(self):
        if self.size == 0:
            return " "
        elif self.size == 1:
            return self._head.__repr__()
        else:
            return self._head.__repr__() + ", " + self._tail._recurr_repr()


    def __repr__(self):
        return "["+self._recurr_repr()+"]"

    @staticmethod
    def construct_empty_stack(rnn_builder):
        init_rnn_state = rnn_builder.initial_state()
        dummy_rnn_input = dy.inputVector(np.zeros(rnn_builder.input_dim))
        init_rnn_state = init_rnn_state.add_input(dummy_rnn_input)
        return StackLSTM(None, None, init_rnn_state)

