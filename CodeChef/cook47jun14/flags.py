

def paint(N):
    colourings = 0 #possible different colourings
    flag1 = N * (N-1) * (N-1)
    flag2 = flag1
    flag3 = N * (N-1) * (N-2)
    flag4 = N * (N-1) * (N-2) * (N-2)
    flag5 = flag4



class Flag:
    patterns = [ # how many already considered polygons does the j'th one touch 
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 2],
        [0, 1, 2, 2],
        [0, 1, 2, 2],
    ]
    flags = map(Flag, patterns)
    instances = []

    def __new__(cls, *args, **kwargs):
        cls.instances += 1
        return super().__new__(cls)
        
    def __init__(self, *pattern)
        self.pattern = 
    


