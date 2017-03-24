class TreeNode:

    def __init__(self, label, children, attributes, word_position=None): # word_id only for terminals
        self.label = label

        if word_position is not None:
            self._memo_covered_indices = set([word_position])
        else:
            self._memo_covered_indices = None

        if len(children) == 0:
            self.leftmost_word_position = word_position
        else:
            self.leftmost_word_position = 1000000
            for child in children:
                self.leftmost_word_position = min(self.leftmost_word_position, child.leftmost_word_position)

        self.children = sorted(children, key=lambda x: x.leftmost_word_position)

        self.attributes = attributes

        self.vector = None
        self.memory_cell = None

    def is_equal_to(self, node):
        if self.covered_indices == node.covered_indices and self.label == node.label:
            return True
        else:
            return False


    @property
    def covered_indices(self):
        if self._memo_covered_indices is None:
            self._memo_covered_indices = set()
            for child in self.children:
                self._memo_covered_indices |= child.covered_indices
        return self._memo_covered_indices

    def __repr__(self):
        if self.is_terminal():
            return self.label
        else:
            children_rep = " ".join(map(str, self.children))
            return "{%s %s}" % (self.label, children_rep)

    def is_terminal(self):
        return len(self.children) == 0

    def to_tags(self):
        leaves = self.give_me_terminal_nodes()
        tags = list(map(lambda x: x.attributes['tag'], leaves))
        return tags

    def to_sentence(self):
        leaves = self.give_me_terminal_nodes()
        words = list(map(lambda x: x.label, leaves))
        return words

    def _annotate_nodes_with_indices(self, curr_free_index):
        if self.is_terminal():
            return curr_free_index
        else:
            for child in self.children:
                curr_free_index = child._annotate_nodes_with_indices(curr_free_index)
            self.attributes['index'] = curr_free_index
            return curr_free_index+1

    def _to_export_lines(self, parent_index):
        if self.is_terminal():
            line = "\t".join([
                self.label,
                self.attributes['tag'],
                self.attributes.get('morph_tag', "-"),
                self.attributes.get('edge', '-'),
                str(parent_index)
            ])
            return [(self.attributes['word_position'], line)]
        else:
            lines = []
            for child in self.children:
                lines.extend(child._to_export_lines(self.attributes['index']))

            line = "\t".join([
                '#'+str(self.attributes['index']),
                self.label,
                self.attributes.get('morph_tag', '-'),
                self.attributes.get('edge', '-'),
                str(parent_index)])

            lines.append((self.attributes['index'], line))

            return lines

    def to_export(self, sent_id=None):

        lines = []

        self._annotate_nodes_with_indices(500)
        self.attributes['index'] = 0
        for child in self.children:
            lines.extend(child._to_export_lines(0))

        lines = map(lambda x: x[1], sorted(lines, key=lambda x:x[0]))

        if sent_id is None:
            sent_id = self.attributes['sent_id']

        first_line = "#BOS %d 1 985275570 1"%sent_id
        last_line = "#EOS %d"%sent_id

        lines = [first_line] + list(lines) + [last_line]

        return "\n".join(lines)

    def give_me_nonterminal_nodes(self):
        if self.is_terminal():
            return []
        else:
            children_nonterms = [item for child in self.children for item in child.give_me_nonterminal_nodes()]
            children_nonterms.append(self)
            return children_nonterms

    def give_me_terminal_nodes(self):
        if self.is_terminal():
            return [self]
        else:
            leaves = [item for child in self.children for item in child.give_me_terminal_nodes()]
            leaves = sorted(leaves, key=lambda x: x.leftmost_word_position)
            return leaves

