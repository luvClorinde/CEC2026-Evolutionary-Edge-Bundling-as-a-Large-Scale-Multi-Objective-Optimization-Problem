# はじめに
最適化されていない(自分の実験の実装環境においてという前提で)ライブラリを、より高速で動くものに進化させようの会

# runner.py
pc名を変える

# 各アルゴリズムファイル
大文字に修正

# density_estimator 
jmetalpy util　主にSPEA2のため

class KNNDE def cde
def compute_density_estimator(self, solutions: List[S]):
    solutions_size = len(solutions)
    if solutions_size <= self.k:
        return

    """# Compute distance matrix 
    高速化のため変更
    self.distance_matrix = numpy.zeros(shape=(solutions_size, solutions_size))
    for i in range(solutions_size):
        for j in range(solutions_size):
            self.distance_matrix[i, j] = self.distance_matrix[j, i] = euclidean(
                solutions[i].objectives, solutions[j].objectives
            )"""

    # Compute distance matrix using vectorized cdist
    from scipy.spatial.distance import cdist
    obj_matrix = numpy.array([s.objectives for s in solutions])
    self.distance_matrix = cdist(obj_matrix, obj_matrix, metric='euclidean')
    

    # Assign knn_density attribute
    for i in range(solutions_size):
        distances = list(self.distance_matrix[i])
        distances.sort()
        solutions[i].attributes["knn_density"] = distances[self.k]

置き換えれば完了