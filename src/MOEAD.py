from __future__ import annotations

import cProfile
import csv
import json
import logging
import pstats
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from jmetal.operator.crossover import DifferentialEvolutionCrossover
from util.crossover import FastDECrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.aggregation_function import Tschebycheff
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

import util.function as function
import util.module as module
import util.observer as obs
import util.testgraphs as testgraph

logging.getLogger("jmetal").setLevel(logging.WARNING)

### パラメータ設定


@dataclass(frozen=True)
class MOEADParameters:
    """Hyper-parameters controlling the MOEA/D run."""

    iterations: int = 750  # MOEA/D を実行する総反復回数を指定します。
    population_size: int = 500 # MOEA/Dで扱う個体数を設定します。
    crossover_cr: float = 1.0  # Differential Evolution 交叉のCRを指定します。
    crossover_f: float = 0.5  # Differential Evolution 交叉のスケール係数。
    mutation_probability: float | None = None  # 変異確率（未指定なら変数数の逆数）。
    mutation_distribution_index: float = 20.0  # 多項式変異の分布指数。
    neighborhood_selection_probability: float = 0.9  # 近傍から親を選ぶ確率。
    max_replacements: int = 2  # 1回の評価で置換可能な個体数の上限。
    neighbor_size: int = 20  # 近傍サイズ。
    weight_seed: int = 42  # 重み生成用の乱数シード。

    control_point: int = 3  # コントロールポイントの数を決めます。


EXPERIMENT_DATE = "2026-03-30"  # resultフォルダ名となる
GRAPH_NAMES: Sequence[str] = [1]
OBJECTIVES: Sequence[str] = ("MELD", "MOA", "EDD", "PQ")
VALUE_RANGE: list = [30, 50]  # 制御点の探索範囲を定める上下限を決めます。
RESULT_ROOT = Path("results") / "CEC"
ALGORITHM = "moead"

# Drawing configuration shared with the NSGA-II experiment so that output
# images are consistent between algorithms.
DRAW_ALPHA = 0.6
DRAW_EDGE_WIDTH = 0.8
DRAW_SMOOTH_THRESHOLD = 8.0


### jsonヘルパー

def _json_default(obj):
    """Helper for JSON serialisation of numpy/cupy related values."""

    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


### MOEA/D用重み生成

def _ensure_weight_file(directory: Path, n_obj: int, population_size: int, seed: int) -> Path:
    """Create a MOEA/D-compatible weight file if it does not already exist."""

    directory.mkdir(parents=True, exist_ok=True)
    filename = f"W{n_obj}D_{population_size}.dat"
    file_path = directory / filename

    if file_path.exists():
        return directory

    rng = np.random.default_rng(seed=seed)
    weights = rng.random((population_size, n_obj))
    weights /= weights.sum(axis=1, keepdims=True)

    with file_path.open("w", encoding="utf-8") as handle:
        for row in weights:
            handle.write(" ".join(f"{value:.12f}" for value in row) + "\n")

    return directory


### 問題の定義
class EdgeBundlingMOEAD(FloatProblem):
    def __init__(
        self,
        bound: float,
        module: module.EdgeBundlingModule,
        func: function.MyFunc,
        objectives: Sequence[str],
    ) -> None:
        super().__init__()
        self.module = module
        self.func = func
        self.obj_labels = list(objectives)
        self.n_control = self.module.n_control
        self.lower_bound = [-bound for _ in range(self.number_of_variables())]
        self.upper_bound = [bound for _ in range(self.number_of_variables())]
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MAXIMIZE, self.MINIMIZE]
        self.evaluation = 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        variables = np.asarray(solution.variables, dtype=np.float64)
        pos_np = self.module.move_ver(variables)
        meld = float(self.func.meld(pos_np))
        moa, edd = self.func.moa_edd(pos_np)
        moa_val = moa
        edd_val = float(edd)
        pq = self.func.path_quality(pos_np)

        solution.objectives[0] = meld
        solution.objectives[1] = moa_val
        solution.objectives[2] = -edd_val
        solution.objectives[3] = pq
        self.evaluation += 1
        return solution

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives(),
        )
        values = np.random.uniform(
            low=self.lower_bound[0],
            high=self.upper_bound[0],
            size=self.number_of_variables(),
        )
        new_solution.variables = values
        return new_solution

    def number_of_constraints(self) -> int:
        return 0

    def number_of_objectives(self) -> int:
        return len(self.obj_labels)

    def number_of_variables(self) -> int:
        return self.module.n_edges * self.n_control

    def name(self) -> str:
        return "EdgeBundlingMOEAD"


### 実験定義
class MOEADEdgeBundlingExperiment:
    """Bundle MOEA/D optimisation logic for a single graph."""

    def __init__(
        self,
        graph_name: str,
        objectives: Sequence[str],
        parameters: MOEADParameters,
        results_dir: Path,
        i: int = 0,
    ) -> None:
        self.graph_name = graph_name
        self.parameters = parameters
        self.objectives = list(objectives)
        self.results_dir = results_dir / f"value_range_{VALUE_RANGE[i]}"

        graph, positions, canvas = testgraph.get_graph(graph_name)
        self.module = module.EdgeBundlingModule(
            graph, positions, parameters.control_point, canvas
        )
        self.functions = function.MyFunc(
            graph, positions, parameters.control_point, canvas
        )

        self.dimension = self.module.n_edges * self.module.n_control

        self.problem = EdgeBundlingMOEAD(
            VALUE_RANGE[i],
            self.module,
            self.functions,
            objectives,
        )

        self.max_evaluations = parameters.iterations * parameters.population_size
        mutation_probability = (
            parameters.mutation_probability
            if parameters.mutation_probability is not None
            else 1 / self.problem.number_of_variables()
        )

        weight_dir = self.results_dir / "weights"
        self.weight_dir = _ensure_weight_file(
            weight_dir,
            self.problem.number_of_objectives(),
            parameters.population_size,
            parameters.weight_seed,
        )

        self.algorithm = MOEAD(
            problem=self.problem,
            population_size=parameters.population_size,
            crossover=FastDECrossover(
                CR=parameters.crossover_cr,
                F=parameters.crossover_f,
            ),
            mutation=PolynomialMutation(
                probability=mutation_probability,
                distribution_index=parameters.mutation_distribution_index,
            ),
            neighbourhood_selection_probability=parameters.neighborhood_selection_probability,
            max_number_of_replaced_solutions=parameters.max_replacements,
            neighbor_size=parameters.neighbor_size,
            aggregation_function=Tschebycheff(len(self.objectives)),
            weight_files_path=str(self.weight_dir),
            termination_criterion=StoppingByEvaluations(
                max_evaluations=self.max_evaluations
            ),
        )

        self.value_observer = obs.ValueObserver()
        self.algorithm.observable.register(observer=ProgressBarObserver(max=self.max_evaluations))
        self.algorithm.observable.register(observer=self.value_observer)

        self.profiler = cProfile.Profile()

    def _compute_metrics(self, pos_np: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["MELD"] = float(self.functions.meld(pos_np))
        moa_val, edd_val = self.functions.moa_edd(pos_np)
        metrics["MOA"] = moa_val
        metrics["EDD"] = float(edd_val)
        metrics["PQ"] = self.functions.path_quality(pos_np)
        return metrics

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        if not self.value_observer.iterations:
            return
        history_path = self.results_dir / "history.csv"
        with history_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["evaluations", *self.objectives])
            for iteration, values in zip(
                self.value_observer.iterations, self.value_observer.best_objectives
            ):
                if isinstance(values, list):
                    row = [iteration] + [float(v) for v in values]
                else:
                    row = [iteration, float(values)]
                writer.writerow(row)

    def _save_front(self, front: Sequence[FloatSolution]) -> None:
        if not front:
            return
        var_path = self.results_dir / "var.txt"
        fun_path = self.results_dir / "fun.txt"
        module.save_solutions_to_txt(front, str(var_path), str(fun_path))

    def _save_front_artifacts(self, front: Sequence[FloatSolution]) -> Path | None:
        if not front:
            return None

        image_dir = self.results_dir / "image"
        image_dir.mkdir(parents=True, exist_ok=True)

        for index, solution in enumerate(front, start=1):
            vector = np.asarray(solution.variables, dtype=np.float64)
            pos_np = self.module.move_ver(vector)
            metrics = self._compute_metrics(pos_np)

            edges_path = image_dir / f"straight{index}.png"
            self.module.show_graph_edges_np(
                pos_np,
                edge_width=DRAW_EDGE_WIDTH,
                alpha=DRAW_ALPHA,
                save_path=str(edges_path),
            )

            annotation_values = [metrics[name] for name in self.objectives]
            self.module.annotate_metrics(edges_path, self.objectives, annotation_values)

    def _save_metadata(
        self,
        front_size: int,
        profile_path: Path,
    ) -> None:
        metadata = {
            "graph": self.graph_name,
            "objectives": self.objectives,
            "dimension": self.dimension,
            "parameters": asdict(self.parameters),
            "max_evaluations": self.max_evaluations,
            "evaluations": getattr(self.algorithm, "evaluations", None),
            "front_size": front_size,
            "profile": {
                "file": str(profile_path),
            },
        }
        with (self.results_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2, default=_json_default)

        profile_txt = self.results_dir / "profile.txt"
        with profile_txt.open("w") as f:
            stats = pstats.Stats(str(profile_path), stream=f)
            stats.sort_stats("cumtime").print_stats()

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

        profile_path = self.results_dir / "profile.prof"
        self.profiler.enable()
        self.algorithm.run()
        self.profiler.disable()
        self.profiler.dump_stats(str(profile_path))

        front = get_non_dominated_solutions(self.algorithm.result())
        self._save_history()
        self._save_front(front)
        #self._save_front_artifacts(front)
        self._save_metadata(len(front), profile_path)


def main(i = "0", graphs = None, pc = None, iterations = None, population_size = None) -> None:
    parameters = MOEADParameters()
    overrides = {k: v for k, v in [("iterations", iterations), ("population_size", population_size)] if v is not None}
    if overrides:
        parameters = replace(parameters, **overrides)
    if graphs is None:
        graphs = GRAPH_NAMES
    for r, _value in enumerate(VALUE_RANGE):
        for graph_name in graphs:
            result_dir = (
                RESULT_ROOT
                / pc
                / f"iter_{i}"
                / ALGORITHM
                #/ EXPERIMENT_DATE
                / f"graph_{graph_name}"
                #/ f"objectives_{'_'.join(OBJECTIVES)}"
            ) if pc is not None else (
                RESULT_ROOT
                / f"graph_{graph_name}"
            )
            experiment = MOEADEdgeBundlingExperiment(
                graph_name,
                OBJECTIVES,
                parameters,
                result_dir,
                r,
            )
            experiment.run()


if __name__ == "__main__":
    main()
