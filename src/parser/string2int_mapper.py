class String2IntegerMapper:

    UNK = "<UNK>"

    def __init__(self):
        self.s2i = dict()
        self.i2s = dict()
        self.free_int = 0

    def _add_string_with_index(self, s, i):
        if s in self.s2i or i in self.i2s:
            raise Exception("%s or %d already in there"%(s, i))
        self.s2i[s] = i
        self.i2s[i] = s
        if self.free_int == i:
            self.free_int += 1
        if self.free_int in self.i2s:
            raise Exception("something is terribly wrong")

    def size(self):
        return len(self.s2i)

    def get_or_store(self, s):
        self.add_string_if_not_alredy_in_there(s)
        return self.s2i[s]

    def add_string(self, s):
        if s not in self.s2i:
            i = self.free_int
            self.free_int += 1
            self.s2i[s] = i
            self.i2s[i] = s

    def __getitem__(self, key):
        if type(key) is int:
            return self.i2s[key]
        else:
            if key not in self.s2i:
                return self.s2i[String2IntegerMapper.UNK]
            else:
                return self.s2i[key]

    def save(self, fn):
        with open(fn, "w") as fh:
            for i, s in sorted(self.i2s.items(), key=lambda x: x[0]):
                print("%d %s"%(i,s), file=fh)

    @staticmethod
    def load(fn):
        mapper = String2IntegerMapper()
        with open(fn) as fh:
            for line in fh:
                fields = line.rstrip().split(" ")
                i = int(fields[0])
                s = " ".join(fields[1:])
                mapper._add_string_with_index(s, i)
        return mapper


class ContainerStr2IntMaps:

    def __init__(self, w2i=None, p2i=None, n2i=None, c2i=None, ext_w2i=None):
        self.w2i = w2i
        self.p2i = p2i
        self.n2i = n2i
        self.c2i = c2i
        self.ext_w2i = ext_w2i

