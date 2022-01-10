import numpy as np
from data_generator import generator


if __name__ == "__main__":
    np.random.seed()

    gen = generator()
    gen.generate_data(5000)






