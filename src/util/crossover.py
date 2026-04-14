from __future__ import annotations

import random
from typing import List

import numpy as np
from jmetal.core.operator import Crossover
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import DifferentialEvolutionCrossover


class FastDECrossover(DifferentialEvolutionCrossover):
    """deepcopy を回避し、DE変異をnumpy化した高速版。"""

    def execute(self, parents):
        current = self.current_individual
        child = FloatSolution(
            current.lower_bound,
            current.upper_bound,
            len(current.objectives),
        )

        n = len(current.variables)
        rand_idx = random.randint(0, n - 1)

        current_vars = np.asarray(current.variables)
        new_vars = (
            np.asarray(parents[2].variables)
            + self.F * (np.asarray(parents[0].variables) - np.asarray(parents[1].variables))
        )

        mask = np.random.random(n) < self.CR
        mask[rand_idx] = True

        result = np.where(mask, new_vars, current_vars)
        child.variables = np.clip(result, current.lower_bound[0], current.upper_bound[0])

        return [child]


class FastSBXCrossover(Crossover[FloatSolution, FloatSolution]):
    """SBXCrossover without deepcopy.

    jMetal の SBXCrossover は execute() の冒頭で親個体を deepcopy するが、
    lower_bound / upper_bound（変数次元数の大きなリスト）まで毎回コピーされるため
    大規模グラフで著しく遅くなる。
    本クラスでは子個体を直接生成し、variables のみ ndarray.copy() でコピーする。
    """

    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super().__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        # deepcopy を使わず子個体を直接生成
        # lower_bound / upper_bound は参照共有（問題定義の定数なのでコピー不要）
        child0 = FloatSolution(
            parents[0].lower_bound,
            parents[0].upper_bound,
            parents[0].objectives.__len__(),
        )
        child1 = FloatSolution(
            parents[1].lower_bound,
            parents[1].upper_bound,
            parents[1].objectives.__len__(),
        )
        child0.variables = np.array(parents[0].variables, copy=True)
        child1.variables = np.array(parents[1].variables, copy=True)
        offspring = [child0, child1]

        rand = random.random()
        if rand <= self.probability:
            for i in range(len(parents[0].variables)):
                value_x1 = parents[0].variables[i]
                value_x2 = parents[1].variables[i]

                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        lower_bound = parents[0].lower_bound[i]
                        upper_bound = parents[0].upper_bound[i]

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, 1.0 / (self.distribution_index + 1.0))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))

                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, 1.0 / (self.distribution_index + 1.0))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                        c1 = max(lower_bound, min(c1, upper_bound))
                        c2 = max(lower_bound, min(c2, upper_bound))

                        if random.random() <= 0.5:
                            offspring[0].variables[i] = c2
                            offspring[1].variables[i] = c1
                        else:
                            offspring[0].variables[i] = c1
                            offspring[1].variables[i] = c2
                    else:
                        offspring[0].variables[i] = value_x1
                        offspring[1].variables[i] = value_x2
                else:
                    offspring[0].variables[i] = value_x1
                    offspring[1].variables[i] = value_x2

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Fast SBX crossover"
