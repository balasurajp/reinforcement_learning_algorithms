import numpy as np
from gym.spaces import Space


class Simplex(Space):
    def __init__(self, n: int) -> None:
        super(Simplex, self).__init__()
        assert n >= 2
        super().__init__(shape=(n, ), dtype=np.float32)
        self.n = n
        self.low = np.zeros(shape=(n, ), dtype=np.float32)
        self.high = np.ones(shape=(n, ), dtype=np.float32)

    def sample(self) -> float:
        return np.random.dirichlet(alpha=[np.random.random()*10.0 + 1.0 for _ in range(self.n)])

    def contains(self, x) -> bool:
        if len(x) != self.n:
            return False
        if sum(x) != 1.0:
            return False
        return True