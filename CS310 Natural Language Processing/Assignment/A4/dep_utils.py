import sys
from collections import defaultdict


class DependencyEdge(object):
    """
    Represent a single dependency edge:
    """
    def __init__(self, word_id, word, pos, head, deprel):
        self.id = word_id
        self.word = word
        self.pos = pos
        self.head = head
        self.deprel = deprel
    
    def __str__(self) -> str:
        return "{d.id}\t{d.word}\t_\t_\t{d.pos}\t_\t{d.head}\t{d.deprel}\t_\t_".format(d=self)


def parse_conll_relation(s):
    fields = s.split("\t")
    word_id_str, word, lemma, upos, pos, feats, head_str, deprel, deps, misc = fields
    word_id = int(word_id_str)
    head = int(head_str)
    return DependencyEdge(word_id, word, pos, head, deprel)


class DependencyTree(object):
    def __init__(self):
        self.deprels = {}
        self.root = None
        self.parent_to_children = defaultdict(list)

    def add_deprel(self, deprel):
        self.deprels[deprel.id] = deprel
        self.parent_to_children[deprel.head].append(deprel.id)
        if deprel.head == 0:
            self.root = deprel.id

    def __str__(self):
        deprels = [v for (k, v) in sorted(self.deprels.items())]
        return "\n".join(str(deprel) for deprel in deprels)
    
    def print_tree(self, parent=None):
        if not parent:
            return self.print_tree(parent=self.root)

        if self.deprels[parent].head == parent:
            return self.deprels[parent].word

        children = [self.print_tree(child) for child in self.parent_to_children[parent]]
        child_str = " ".join(children)
        return "({} {})".format(self.deprels[parent].word, child_str)

    def words(self):
        return [None] + [x.word for (i, x) in self.deprels.items()]

    def pos(self):
        return [None] + [x.pos for (i, x) in self.deprels.items()]
    
    def from_string(s):
        dtree = DependencyTree()
        for line in s.split("\n"):
            if line:
                dtree.add_deprel(parse_conll_relation(line))
        return dtree


def conll_reader(input_file):
    current_deps = DependencyTree()
    while True:
        line = input_file.readline().strip()
        if not line and current_deps:
            yield current_deps
            current_deps = DependencyTree()
            line = input_file.readline().strip()
            if not line:
                break
        current_deps.add_deprel(parse_conll_relation(line))


if __name__ == "__main__":
    with open(sys.argv[1], "r") as in_file:
        relations = set()
        for deps in conll_reader(in_file):
            for deprel in deps.deprels.values():
                relations.add(deprel.deprel)
            print(deps.print_tree())
