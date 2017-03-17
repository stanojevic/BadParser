class Action:

    def __init__(self, name, vector):
        self.name = name
        self.vector = vector

    def __repr__(self):
        return self.name

class ActionStorage:

    def __init__(self, n2i, action_embeddings):
        self.pro_labels_string = [None, None, None, None]
        self.pro_labels_int = [None, None, None, None]
        self.action_names = ['shift', 'swap', 'adj_left', 'adj_right']
        for nonterm in sorted(n2i.s2i.keys()):
            self.action_names.append("pro-%s"%nonterm)
            self.pro_labels_string.append(nonterm)
            self.pro_labels_int.append(n2i[nonterm])
        self.SHIFT = 0
        self.SWAP = 1
        self.ADJ_LEFT = 2
        self.ADJ_RIGHT =3
        self.size = len(self.action_names)
        self.action_embeddings = action_embeddings

    def get_pro_index_for_string_label(self, label):
        for i in range(4, len(self.action_names)):
            if self.pro_labels_string[i] == label:
                return i

    def get_action_object(self, action_id):
        return Action(self.action_names[action_id], self.action_embeddings[action_id])
