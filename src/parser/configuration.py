from parser.stack_lstm import StackLSTM
from parser.string2int_mapper import String2IntegerMapper
from data_formats.tree_representation import TreeNode
import dynet as dy
import numpy as np

class Configuration:

    def __init__(self, stack, buffer, actions, params, action_storage, log_prob, promote_count, prev_conf_lstm_state, prev_conf, last_action):
        self.stack = stack
        self.buffer = buffer
        self.actions = actions
        self.action_storage = action_storage
        self.params = params
        self.composition_function = params['composition_function']
        self.log_prob = log_prob
        self.promote_count = promote_count
        self.prev_conf = prev_conf
        self.last_action = last_action

        conf_input_vec_rep_raw = []
        if stack.vector is not None:
            conf_input_vec_rep_raw.append(stack.vector)
        if buffer.vector is not None:
            conf_input_vec_rep_raw.append(buffer.vector)
        if actions.vector is not None:
            conf_input_vec_rep_raw.append(actions.vector)
        conf_input_vec_rep = dy.concatenate(conf_input_vec_rep_raw)

        w = dy.parameter(params['w'])
        W = dy.parameter(params['W'])
        conf_raw_repr = W*conf_input_vec_rep + w
        if prev_conf_lstm_state is None:
            self.conf_lstm_state = None
            self.vector = conf_raw_repr
        else:
            self.conf_lstm_state = prev_conf_lstm_state.add_input(conf_raw_repr)
            self.vector = self.conf_lstm_state.output()
        self._memo_action_log_probabilities = None
        self._memo_allowed_transitions = None

    def is_final_configuration(self):
        if self.buffer.size == 0 and self.stack.size == 1:
            return True
        else:
            return False

    def action_log_probabilities(self):
        if self._memo_action_log_probabilities is None:
            G = dy.parameter(self.params['G'])
            g = dy.parameter(self.params['g'])
            preactivation = G*self.vector+g
            allowed_transitions = self.allowed_transitions()
            allowed_transitions_list = []
            for i in range(len(allowed_transitions)):
                if allowed_transitions[i]:
                    allowed_transitions_list.append(i)
            log_probs = dy.log_softmax(preactivation, restrict=allowed_transitions_list)
            self._memo_action_log_probabilities = log_probs

        return self._memo_action_log_probabilities

    def allowed_transitions(self):
        if self._memo_allowed_transitions is None:
            allowed = np.ones(self.action_storage.size)
            if self.buffer.size == 0:
                allowed[self.action_storage.SHIFT] = 0
            if self.stack.size < 2:
                allowed[self.action_storage.SWAP] = 0
                allowed[self.action_storage.ADJ_LEFT] = 0
                allowed[self.action_storage.ADJ_RIGHT] = 0
            if self.stack.size == 0 \
                or self.promote_count >= 3 \
                or self.stack.top().label.lower() == "root":  # maximally consecutive3 promotes
                new_allowed = np.zeros(self.action_storage.size)
                for non_pro_transition in [
                    self.action_storage.SWAP,
                    self.action_storage.SHIFT,
                    self.action_storage.ADJ_LEFT,
                    self.action_storage.ADJ_RIGHT]:
                    new_allowed[non_pro_transition] = allowed[non_pro_transition]
                allowed = new_allowed
            if allowed[self.action_storage.ADJ_LEFT] and self.stack.top().is_terminal():
                allowed[self.action_storage.ADJ_LEFT] = 0
            if allowed[self.action_storage.ADJ_RIGHT] and self.stack.second_top().is_terminal():
                allowed[self.action_storage.ADJ_RIGHT] = 0
            if allowed[self.action_storage.SWAP] and \
                self.stack.top().leftmost_word_position < self.stack.second_top().leftmost_word_position:
                allowed[self.action_storage.SWAP] = 0


            self._memo_allowed_transitions = allowed

        return self._memo_allowed_transitions

    def transition(self, transition_id):
        if transition_id == self.action_storage.SHIFT:
            return self._tr_shift(transition_id)
        elif transition_id == self.action_storage.SWAP:
            return self._tr_swap(transition_id)
        elif transition_id == self.action_storage.ADJ_LEFT:
            return self._tr_adj_left(transition_id)
        elif transition_id == self.action_storage.ADJ_RIGHT:
            return self._tr_adj_right(transition_id)
        else:
            return self._tr_promote(transition_id)

    def _tr_shift(self, transition_id):
        transition_log_prob = self.action_log_probabilities()[transition_id]
        action_object = self.action_storage.get_action_object(transition_id)
        actions = self.actions.push(action_object)
        stack = self.stack.push(self.buffer.top())
        buffer = self.buffer.pop()
        return Configuration(stack, buffer, actions, self.params, self.action_storage, self.log_prob + transition_log_prob, 0, self.conf_lstm_state, self, "shift")

    def _tr_swap(self, transition_id):
        transition_log_prob = self.action_log_probabilities()[transition_id]
        action_object = self.action_storage.get_action_object(transition_id)
        actions = self.actions.push(action_object)
        buffer = self.buffer.push(self.stack.second_top())
        stack = self.stack.pop().pop().push(self.stack.top())
        return Configuration(stack, buffer, actions, self.params, self.action_storage, self.log_prob + transition_log_prob, 0, self.conf_lstm_state, self, "swap")

    def _tr_adj_left(self, transition_id):
        transition_log_prob = self.action_log_probabilities()[transition_id]
        action_object = self.action_storage.get_action_object(transition_id)
        actions = self.actions.push(action_object)
        buffer = self.buffer
        comp_const = self.stack.second_top() # COMPLEMENT
        head_const = self.stack.top()        # HEAD
        children = head_const.children.copy()
        children.append(comp_const) #order will be sorted out later

        new_constituent = TreeNode(head_const.label, children, head_const.attributes)
        if self.params['composition_function_head_ordered']:
            new_memory_cell, new_vector = self.composition_function.compose(head_const.memory_cell, head_const.vector, comp_const.memory_cell, comp_const.vector, True)
        else:
            new_memory_cell, new_vector = self.composition_function.compose(comp_const.memory_cell, comp_const.vector, head_const.memory_cell, head_const.vector, True)
        new_constituent.vector = new_vector
        new_constituent.memory_cell = new_memory_cell

        stack = self.stack.pop().pop().push(new_constituent)
        return Configuration(stack, buffer, actions, self.params, self.action_storage, self.log_prob + transition_log_prob, 0, self.conf_lstm_state, self, "adj_left")

    def _tr_adj_right(self, transition_id):
        transition_log_prob = self.action_log_probabilities()[transition_id]
        action_object = self.action_storage.get_action_object(transition_id)
        actions = self.actions.push(action_object)
        buffer = self.buffer
        head_const = self.stack.second_top()  # HEAD
        comp_const = self.stack.top()         # COMPLEMENT
        children = head_const.children.copy()
        children.append(comp_const) #order will be sorted out later

        new_constituent = TreeNode(head_const.label, children, head_const.attributes)
        new_memory_cell, new_vector = self.composition_function.compose(head_const.memory_cell, head_const.vector, comp_const.memory_cell, comp_const.vector, False)
        new_constituent.vector = new_vector
        new_constituent.memory_cell = new_memory_cell

        stack = self.stack.pop().pop().push(new_constituent)
        return Configuration(stack, buffer, actions, self.params, self.action_storage, self.log_prob + transition_log_prob, 0, self.conf_lstm_state, self, "adj_right")

    def _tr_promote(self, transition_id):
        transition_log_prob = self.action_log_probabilities()[transition_id]
        action_object = self.action_storage.get_action_object(transition_id)
        actions = self.actions.push(action_object)
        buffer = self.buffer

        child_const = self.stack.top()
        label = self.action_storage.pro_labels_string[transition_id]
        new_constituent = TreeNode(label, [child_const], {})
        nonterm_id = self.action_storage.pro_labels_int[transition_id]
        nonterm_emb = self.params['E_n'][nonterm_id]

        new_memory_cell, new_vector = self.composition_function.promote(child_const.memory_cell, child_const.vector, nonterm_emb)

        new_constituent.vector = new_vector
        new_constituent.memory_cell = new_memory_cell

        stack = self.stack.pop().push(new_constituent)
        return Configuration(stack, buffer, actions, self.params, self.action_storage, self.log_prob + transition_log_prob, self.promote_count+1, self.conf_lstm_state, self, "pro")

    def __repr__(self):
        return "("+self.stack.__repr__()+" ||| "+self.buffer.__repr__()+")"

    @staticmethod
    def construct_init_configuration(words, pos_seq, params, action_storage, all_s2i, dropout_rate_embeddings=0.0):
        leaf_nodes = Configuration._convert_sentence_to_list_of_tree_nodes(words, pos_seq, params, all_s2i, dropout_rate_embeddings)
        if 'BiLSTM' in params:
            leaf_vectors = [leaf_node.vector for leaf_node in leaf_nodes]
            new_leaf_vectors = params['BiLSTM'].transduce(leaf_vectors)
            for leaf_node, new_leaf_vector in zip(leaf_nodes, new_leaf_vectors):
                leaf_node.vector = new_leaf_vector
        init_buffer = Configuration._construct_init_buffer(leaf_nodes, params)
        init_stack = StackLSTM.construct_empty_stack(params['Stack_LSTM'])
        init_actions = StackLSTM.construct_empty_stack(params['Action_LSTM'])

        init_log_prob = dy.scalarInput(0)

        if 'ConfigurationLSTM' in params:
            conf_lstm_state = params['ConfigurationLSTM'].initial_state()
        else:
            conf_lstm_state = None

        return Configuration(init_stack, init_buffer, init_actions, params, action_storage, init_log_prob, 0, conf_lstm_state, None, None)

    @staticmethod
    def _construct_init_buffer(elements, params):
        buffer = StackLSTM.construct_empty_stack(params['Buffer_LSTM'])
        for element in reversed(elements):
            buffer = buffer.push(element)
        return buffer

    @staticmethod
    def _convert_sentence_to_list_of_tree_nodes(words, pos_seq, params, all_s2i, dropout_rate_embeddings):
        nodes = []
        for word, pos, word_position in zip(words, pos_seq, range(len(words))):
            node = Configuration._convert_word_to_tree_node(word, pos, word_position, params, all_s2i, dropout_rate_embeddings)
            nodes.append(node)
        return nodes

    @staticmethod
    def _convert_word_to_tree_node(word, pos, word_position, params, all_s2i, dropout_rate_embeddings):
        node = TreeNode(word, [], {'tag': pos, 'word_position': word_position}, word_position=word_position)
        all_embeddings = []
        if 'E_w' in params:
            w_emb = params['E_w'][all_s2i.w2i[word]]
            all_embeddings.append(w_emb)
        p_emb = params['E_p'][all_s2i.p2i[pos]]
        all_embeddings.append(p_emb)
        #if all_s2i.c2i is not None:
        if 'E_c' in params:
            c_emb_for_word = Configuration._compute_char_emb_for_word(word, params, all_s2i)
            all_embeddings.append(c_emb_for_word)
        #if all_s2i.ext_w2i is not None:
        if 'E_pretrained' in params:
            ext_embedding = dy.lookup(params['E_pretrained'], all_s2i.ext_w2i[word], update=params['E_pretrained_update'])
            all_embeddings.append(ext_embedding)
        V = dy.parameter(params['V'])
        v = dy.parameter(params['v'])
        input_vec = dy.concatenate(all_embeddings)
        input_vec = dy.dropout(input_vec, dropout_rate_embeddings)
        node.vector = dy.rectify(V*input_vec+v)
        node.memory_cell = dy.inputVector(np.zeros(node.vector.npvalue().size))
        return node

    @staticmethod
    def _compute_char_emb_for_word(word, params, all_s2i):
        fw_lstm = params['Char_LSTM_Forward'].initial_state()
        bw_lstm = params['Char_LSTM_Backward'].initial_state()
        c2i = all_s2i.c2i

        cs = [c2i[c] for c in word]

        cembs = [params['E_c'][c] for c in cs]

        fw_exps = fw_lstm.transduce(cembs)
        bw_exps = bw_lstm.transduce(reversed(cembs))

        return dy.concatenate([fw_exps[-1], bw_exps[-1]])


