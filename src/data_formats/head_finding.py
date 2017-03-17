
import re

class HeadFinder:

    def __init__(self, filename):
        self.rules = self._load_rules(filename)

    @staticmethod
    def _load_rules(filename):
        rules = dict()
        with open(filename) as fh:
            for line in fh:
                if not line.startswith("%") and not re.match("^\s*$", line):
                    fields = line.rstrip().split(" ")
                    const_type = fields[0]
                    direction = fields[1]
                    potential_heads = fields[2:]
                    if const_type not in rules:
                        rules[const_type] = []

                    rule = (direction, potential_heads)
                    rules[const_type].append(rule)

        return rules

    def mark_head(self, node):
        if len(node.children) == 0:
            return
        elif len(node.children) == 1:
            self.mark_head(node.children[0])
            node.attributes['head_child'] = 0
            return
        else:
            for child in node.children:
                self.mark_head(child)

            rules = self.rules[node.label.lower()]
            for direction, nonterms in rules:
                if direction == "left-to-right":
                    numeration = list(enumerate(node.children))
                elif direction == "right-to-left":
                    numeration = reversed(list(enumerate(node.children)))
                else:
                    raise Exception("unknown head direction")

                for i, child in numeration:
                    for nonterm in nonterms:
                        if child.label.lower() == nonterm or child.attributes.get('tag', '').lower() == nonterm:
                            node.attributes['head_child'] = i
                            return

            # since we are here we didn't find any head
            if rules[0][0] == "left-to-right":
                node.attributes['head_child'] = 0
                return
            else:
                node.attributes['head_child'] = len(node.children)-1
                return


