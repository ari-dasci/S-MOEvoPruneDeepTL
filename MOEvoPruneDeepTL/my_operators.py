import copy
import random
from typing import List
import numpy as np
from jmetal.core.operator import Crossover
from jmetal.core.solution import BinarySolution
from jmetal.util.ckecking import Check

class UniformCrossover(Crossover[BinarySolution, BinarySolution]):
    def __init__(self):
        super(UniformCrossover, self).__init__(probability=1.0)

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Uniform Crossover"

    def execute(self, parents: List[BinarySolution]) -> List[BinarySolution]:
        Check.that(type(parents[0]) is BinarySolution, "Solution type invalid")
        Check.that(type(parents[1]) is BinarySolution, "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        offspring = copy.deepcopy(parents)

        for i in range(0,parents[0].get_total_number_of_bits()):
            rand = random.random()

            if rand < 0.5:
                swap = offspring[0].variables[0][i]
                offspring[0].variables[0][i] = offspring[1].variables[0][i]
                offspring[1].variables[0][i] = swap

        return offspring
