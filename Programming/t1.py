from Method1 import stochasticModel
import numpy as np
from SamplingMethod import samplingmethod

given_lines = 8
num_sample = 1000
num_period = 20
probab = [0.25, 0.25, 0.25, 0.25]
roll_width = np.ones(given_lines) * 20
I = 4
demand_width_array = np.arange(2, 2+I)
sam = samplingmethod(I, num_sample, num_period, probab)

dw, prop = sam.get_prob()
W = len(dw)

m1 = stochasticModel(roll_width, given_lines, demand_width_array, W, I, prop, dw)

ini_demand, upperbound = m1.solveBenders(eps=1e-4, maxit=20)

print(ini_demand)


class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width


class Square(Rectangle):
    def __init__(self, length):
        super(Square, self).__init__(length, length)

class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height


class RightPyramid(Square, Triangle):
    def __init__(self, base, slant_height):
        self.base = base
        self.slant_height = slant_height
        super().__init__(self.base)

    def area(self):
        base_area = super().area()
        perimeter = super().perimeter()
        return 0.5 * perimeter * self.slant_height + base_area

