import random


class MemoryReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]
        self.memory.append(transition)

    def reset(self):
        self.memory = []

    def sample(self, batch_size):
        if batch_size > self.__len__():
            batch_size = self.__len__()
        batch = random.sample(self.memory, batch_size)
        st = []
        st_n = [] 
        lact = [] 
        lr = []
        nst = []
        nst_n = []
        term = []

        for b in batch:
            st.append(b[0])
            st_n.append(b[1])
            lact.append(b[2])
            lr.append(b[3])
            nst.append(b[4])
            nst_n.append(b[5])
            term.append(b[6])

        return st, st_n, lact, lr, nst, nst_n, term

    def __len__(self):
        return len(self.memory)

