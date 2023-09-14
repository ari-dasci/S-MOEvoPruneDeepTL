import random
import numpy as np

import platform
import subprocess
import sys
from sys import stderr
import os

from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution

class EvoPruneDeepTL(BinaryProblem):
    def __init__(self, method, cadena, number_of_bits: int = 256):
        super(EvoPruneDeepTL, self).__init__()
        self.number_of_bits = number_of_bits # longitud de la solucion
        self.number_of_objectives = 3 # objetivos-> acc, neuronas
        self.number_of_variables = 1 #EvoDeepTLPruning esto nos dice que tenemos una unica lista con longitud self.number_of_bits
        self.number_of_constraints = 0

        self.method = method
        self.cadena = cadena
        self.memory = dict()
        self.num_evals = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['Accuracy', "# active neurons", "AUROC"]


    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        # llamar a mi fitness
        ind = list(map(int,solution.variables[0]))
        #print(ind)
        code_ind = self.bin2dec(ind)
        solution.objectives[1] = ind.count(1) / self.number_of_bits
        print("wowowowowowow")
        print(solution.objectives[1])
        #print(code_ind)

        if code_ind in self.memory:
            solution.objectives[0] = self.memory[code_ind]
        else:
            #print("Elemento")
            #print(ind)
            #acc, auroc = self.method.fitness(ind)
            #acc = random.random()
            acc, auroc = self.method.fitness(ind)

            solution.objectives[0] = 1-acc
            solution.objectives[2] = 1-auroc


            f = open(self.method.get_problem_name() + self.cadena + ".txt", "a+")

            for c in ind:
                f.write(str(c))

            f.write(",")
            f.write(str(solution.objectives[0]))
            f.write(",")
            f.write(str(solution.objectives[1]))
            f.write(",")
            f.write(str(solution.objectives[2]))
            f.write("\n")
            f.close()

            print("Accuracy para elemento ")
            print(solution.objectives[0])
            print("Neuronas activas")
            print(solution.objectives[1])
            print("AUROC")
            print(solution.objectives[2])

            self.memory[code_ind] = solution.objectives[0]
            self.num_evals += 1


        print("NUM EVALS-> " + str(self.num_evals))
        print(solution.objectives)

        return solution

    def create_solution(self) -> BinarySolution:
        units = [i for i in range(self.number_of_bits)]
        p_1 = random.uniform(0, 1)
        num_ones = round(p_1 * self.number_of_bits)

        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        individual = np.zeros(self.number_of_bits, dtype=np.bool)

        # seleccionamos todos los 1
        indexes = np.random.permutation(units)[:num_ones]
        individual[indexes] = True

        # lista de booleanes true-false indicando si están activas o no
        new_solution.variables[0] = list(map(bool, individual))

        #print(new_solution.variables[0])
        return new_solution

    def get_name(self) -> str:
        return 'MOEvoPruneDeepTL'

    def bin2dec(self, element):
        return int("".join(str(x) for x in element), 2)
