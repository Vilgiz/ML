from typing import List
import numpy as np
import pandas as pd
from typing import NoReturn, Tuple, List

# Task 1


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
        0 --- злокачественной (B).
    """
    data = pd.read_csv(path_to_csv)

    y = data.iloc[:, -1].values
    X = data.iloc[:, :-1].values

    y = data['label'].map({'M': 1, 'B': 0})

    indices = np.random.permutation(len(y))

    X = X[indices]
    y = y[indices]

    return (X, y)


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток,
        1 если сообщение содержит спам, 0 если не содержит.

    """
    data = pd.read_csv(path_to_csv)

    y = data.iloc[:, -1].values
    X = data.iloc[:, :-1].values

    indices = np.random.permutation(len(y))

    X = X[indices]
    y = y[indices]

    return (X, y)


# Task 2


def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    length = int(len(X) * ratio)

    X_train = X[:length]
    y_train = y[:length]
    X_test = X[length:]
    y_test = y[length:]

    return X_train, y_train, X_test, y_test


# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """

    unique_classes = np.unique(y_true)
    Precision = np.zeros(len(unique_classes))
    Recall = np.zeros(len(unique_classes))

    total = len(y_true)

    for i, class_label in enumerate(unique_classes):
        TP = np.sum((y_true == class_label) & (y_pred == class_label))
        TN = np.sum((y_true != class_label) & (y_pred != class_label))
        FN = np.sum((y_true == class_label) & (y_pred != class_label))
        FP = np.sum((y_true != class_label) & (y_pred == class_label))

        Precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        Recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    Accuracy = np.sum(y_true == y_pred)/(len(y_pred))

    return (Precision, Recall, float(Accuracy))


# Task 4


class KDNode:
    def __init__(self, point=None, left=None, right=None, is_leaf=False, points=None):
        self.point = point
        self.points = points
        self.left = left
        self.right = right
        self.is_leaf = is_leaf


class KDTree:
    def __init__(self, X, leaf_size=40):
        self.leaf_size = leaf_size
        self.X = X
        self.root = self.build_kdtree(X)

    def build_kdtree(self, points, depth=0):
        if len(points) == 0:
            return None

        if len(points) <= self.leaf_size:
            return KDNode(points=points, is_leaf=True)

        k = points.shape[1]
        axis = depth % k

        sorted_points = points[np.argsort(points[:, axis])]
        median_idx = len(sorted_points) // 2

        return KDNode(
            point=sorted_points[median_idx],
            left=self.build_kdtree(sorted_points[:median_idx], depth + 1),
            right=self.build_kdtree(sorted_points[median_idx + 1:], depth + 1)
        )

    def query(self, X, k=1):
        return [self.nearest_neighbor(pt, k) for pt in X]

    def nearest_neighbor(self, point, k=1):
        neighbors = []
        max_dist = float('inf')
        self._nearest(self.root, point, depth=0,
                      neighbors=neighbors, k=k, max_dist=max_dist)

        neighbor_indices = [np.where(np.all(self.X == np.array(neighbor[0]), axis=1))[0][0]
                            for neighbor in neighbors[:k]]
        return neighbor_indices

    def _nearest(self, node, point, depth, neighbors, k, max_dist):
        if node is None:
            return

        if node.is_leaf:
            distances = np.linalg.norm(node.points - point, axis=1)
            for pt, dist in zip(node.points, distances):
                if len(neighbors) < k or dist < max_dist:
                    neighbors.append((pt, dist))
                    neighbors.sort(key=lambda x: x[1])
                    if len(neighbors) > k:
                        neighbors.pop()
                    max_dist = neighbors[-1][1] if neighbors else float('inf')
            return

        axis = depth % len(point)

        next_branch = node.left if point[axis] < node.point[axis] else node.right
        opposite_branch = node.right if next_branch is node.left else node.left

        self._nearest(next_branch, point, depth + 1, neighbors, k, max_dist)

        if len(neighbors) < k or abs(point[axis] - node.point[axis]) < max_dist:
            self._nearest(opposite_branch, point, depth +
                          1, neighbors, k, max_dist)

        dist = np.linalg.norm(node.point - point)
        if len(neighbors) < k or dist < max_dist:
            neighbors.append((node.point, dist))
            neighbors.sort(key=lambda x: x[1])
            if len(neighbors) > k:
                neighbors.pop()
            max_dist = neighbors[-1][1] if neighbors else float('inf')


def true_closest(X_train, X_test, k):
    result = []
    for x0 in X_test:
        dists = np.linalg.norm(X_train - x0, axis=1)
        bests = np.argsort(dists)[:k]
        result.append(list(bests))
    return result

# Task 5


class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 10000):
        """
        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.
        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.kd_tree = None
        self.labels = None

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """
        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.
        """
        self.kd_tree = KDTree(X, leaf_size=self.leaf_size)
        self.labels = y

    def predict_proba(self, X: np.array) -> List[np.array]:
        """
        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
        """
        neighbor_indices = self.kd_tree.query(X, k=self.n_neighbors)

        probabilities = []
        for indices in neighbor_indices:
            neighbor_labels = self.labels[indices]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            prob = counts / self.n_neighbors
            prob_array = np.zeros(len(np.unique(self.labels)))
            prob_array[unique] = prob
            probabilities.append(prob_array)

        return probabilities

    def predict(self, X: np.array) -> np.array:
        """
        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.
        """
        return np.argmax(self.predict_proba(X), axis=1)
