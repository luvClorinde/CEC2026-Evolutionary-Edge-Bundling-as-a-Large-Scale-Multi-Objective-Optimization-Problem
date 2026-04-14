import gc

import cupy as cp

import MOEAD
import NSGA2
import NSGA3
import SMPSO
import SPEA2


def _free_memory():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


iter = 4
pc = "sterben"
graphs = [1, "jpair", "usair"]
iterations = 750
population_size = 500

for i in range(iter):
    MOEAD.main(i, graphs, pc, iterations=iterations, population_size=population_size)
    _free_memory()
    NSGA2.main(i, graphs, pc, iterations=iterations, population_size=population_size)
    _free_memory()
    NSGA3.main(i, graphs, pc, iterations=iterations, population_size=population_size)
    _free_memory()
    SMPSO.main(i, graphs, pc, iterations=iterations, population_size=population_size)
    _free_memory()
    SPEA2.main(i, graphs, pc, iterations=iterations, population_size=population_size)
    _free_memory()