def ant_colony():
    print("""
class AntColony:
    def __init__(self, graph, ants, iterations, alpha=1, beta=2, evaporation_rate=0.5, Q=100):
        '''
        Инициализация параметров муравьиной колонии.

        :param graph: Граф NetworkX.
        :param ants: Количество муравьев в колонии.
        :param iterations: Количество итераций алгоритма.
        :param alpha: Влияние уровня феромонов на выбор пути.
        :param beta: Влияние расстояния на выбор пути.
        :param evaporation_rate: Коэффициент испарения феромонов.
        :param Q: Постоянная для обновления феромонов.
        '''
        self.graph = graph
        self.ants = ants
        self.iterations = iterations
        self.alpha = alpha  # Влияние феромона
        self.beta = beta    # Влияние эвристики (1/длина)
        self.evaporation_rate = evaporation_rate
        self.Q = Q  # Общий феромон, выделяемый муравьем

        # Инициализация уровня феромонов на ребрах
        self.pheromone = {}
        for edge in self.graph.edges():
            self.pheromone[edge] = 1.0
            self.pheromone[(edge[1], edge[0])] = 1.0  # Для неориентированного графа

    def run(self, start, end):
        best_path = None
        best_cost = float('inf')

        for iteration in range(self.iterations):
            all_paths = []
            for ant in range(self.ants):
                path = self.construct_solution(start, end)
                if path is None:
                    continue  # Если путь не найден
                cost = self.path_cost(path)
                all_paths.append((path, cost))

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            self.update_pheromones(all_paths)
            print(f"Итерация {iteration+1}/{self.iterations}, лучший путь длиной {best_cost}")

        return best_path, best_cost

    def construct_solution(self, start, end):
        path = [start]
        visited = set()
        visited.add(start)
        current = start

        while current != end:
            neighbors = list(self.graph.neighbors(current))
            probabilities = []

            # Расчет суммы для нормализации вероятностей
            denominator = 0.0
            for neighbor in neighbors:
                if neighbor not in visited:
                    edge = (current, neighbor)
                    pheromone = self.pheromone.get(edge, 1.0) ** self.alpha
                    distance = self.graph[current][neighbor].get('weight', 1.0)
                    heuristic = (1.0 / distance) ** self.beta
                    prob = pheromone * heuristic
                    probabilities.append((neighbor, prob))
                    denominator += prob

            if denominator == 0:
                # Если нет доступных соседей, возвращаемся назад
                if len(path) > 1:
                    visited.remove(current)
                    path.pop()
                    current = path[-1]
                    continue
                else:
                    # Путь не найден
                    return None

            # Нормализация вероятностей
            probabilities = [(node, prob / denominator) for node, prob in probabilities]

            next_node = self.select_next_node(probabilities)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def select_next_node(self, probabilities):
        r = random.random()
        cumulative = 0.0
        for node, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return node
        # В редких случаях может не выбраться узел из-за погрешностей
        return probabilities[-1][0]

    def path_cost(self, path):
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.graph[path[i]][path[i+1]].get('weight', 1.0)
        return cost

    def update_pheromones(self, all_paths):
        # Испарение феромонов
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation_rate)
            if self.pheromone[edge] < 0.0:
                self.pheromone[edge] = 0.0

        # Обновление феромонов на пути
        for path, cost in all_paths:
            contribution = self.Q / cost
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                rev_edge = (path[i+1], path[i])  # Для неориентированного графа
                self.pheromone[edge] += contribution
                self.pheromone[rev_edge] += contribution

G = nx.Graph()
G.add_edge(1, 2, weight=2)
G.add_edge(1, 3, weight=2)
G.add_edge(2, 3, weight=1)
G.add_edge(2, 4, weight=3)
G.add_edge(3, 4, weight=1)
G.add_edge(3, 5, weight=3)
G.add_edge(4, 5, weight=1)
G.add_edge(4, 6, weight=2)
G.add_edge(5, 6, weight=2)

colony = AntColony(graph=G, ants=10, iterations=20, alpha=1, beta=2, evaporation_rate=0.5, Q=100)
best_path, best_cost = colony.run(1, 6)
print(f"\nЛучший найденный путь: {best_path} длиной {best_cost}")

          
import matplotlib.pyplot as plt


pos = nx.spring_layout(G)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

if best_path:
    edgelist = list(zip(best_path, best_path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=2.5, edge_color='r')
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in edgelist}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title(f"Лучший найденный путь: {best_path} длиной {best_cost}")
plt.axis('off')
plt.show()""")
    
def bound_comm():
    print("""
import numpy as np
import networkx as nx
import heapq

class TSPSolver:
    def __init__(self, graph):
        '''
        Инициализация решателя TSP.

        :param graph: Граф NetworkX со взвешенными ребрами.
        '''
        self.graph = graph
        self.num_nodes = self.graph.number_of_nodes()
        self.nodes = list(self.graph.nodes)
        self.best_cost = float('inf')
        self.best_path = []
        self.count = 0  # Счетчик проверенных узлов

    def solve(self, start):
        '''
        Запуск алгоритма решения TSP.

        :param start: Стартовый узел.
        :return: Кортеж (best_cost, best_path).
        '''
        # Создаем начальную матрицу расстояний
        self.adj_matrix = nx.to_numpy_array(self.graph, nodelist=self.nodes)
        # Запускаем алгоритм ветвей и границ
        initial_bound = self.calculate_initial_bound(self.adj_matrix)
        heap = []
        # Используем кучу для хранения узлов дерева решений
        heapq.heappush(heap, (initial_bound, [start], self.adj_matrix.copy(), 0))
        while heap:
            bound, path, reduced_matrix, current_cost = heapq.heappop(heap)
            if bound >= self.best_cost:
                continue  # Отсекаем узел
            if len(path) == self.num_nodes:
                # Добавляем стоимость возврата в начальный узел
                last_to_first = self.adj_matrix[self.nodes.index(path[-1])][self.nodes.index(path[0])]
                total_cost = current_cost + last_to_first
                if total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.best_path = path + [path[0]]
                continue
            else:
                # Ветвление
                last = path[-1]
                idx_last = self.nodes.index(last)
                for i in range(self.num_nodes):
                    if self.nodes[i] not in path:
                        temp_matrix = reduced_matrix.copy()
                        # Устанавливаем бесконечность для пройденных путей
                        temp_matrix[idx_last, :] = np.inf
                        temp_matrix[:, i] = np.inf
                        temp_matrix[i, idx_last] = np.inf  # Исключаем обратный путь
                        temp_cost = current_cost + reduced_matrix[idx_last][i]
                        temp_bound = temp_cost + self.reduce_matrix(temp_matrix)
                        if temp_bound < self.best_cost:
                            heapq.heappush(heap, (temp_bound, path + [self.nodes[i]], temp_matrix, temp_cost))
            self.count += 1
        return self.best_cost, self.best_path

    def calculate_initial_bound(self, matrix):
        '''
        Вычисление начальной нижней границы путем редукции исходной матрицы.

        :param matrix: Матрица расстояний.
        :return: Стоимость нижней границы.
        '''
        reduced_matrix = matrix.copy()
        cost = self.reduce_matrix(reduced_matrix)
        return cost

    def reduce_matrix(self, matrix):
        '''
        Редукция матрицы: вычитание минимальных элементов строк и столбцов.

        :param matrix: Матрица для редукции.
        :return: Стоимость редукции.
        '''
        cost = 0
        # Вычитаем минимальные элементы строк
        for i in range(self.num_nodes):
            row = matrix[i, :]
            min_row = np.min(row)
            if min_row != np.inf and min_row > 0:
                cost += min_row
                matrix[i, :] -= min_row
        # Вычитаем минимальные элементы столбцов
        for j in range(self.num_nodes):
            col = matrix[:, j]
            min_col = np.min(col)
            if min_col != np.inf and min_col > 0:
                cost += min_col
                matrix[:, j] -= min_col
        return cost

# Создаем полный граф с 5 узлами
G = nx.complete_graph(5)
# Задаем случайные веса ребрам
np.random.seed(42)  # Фиксируем зерно для воспроизводимости
for u, v in G.edges():
    G[u][v]['weight'] = np.random.randint(1, 10)

# Выводим ребра с весами
print("Ребра графа с весами:")
for u, v in G.edges():
    print(f"{u} <-> {v}: {G[u][v]['weight']}")

solver = TSPSolver(G)
best_cost, best_path = solver.solve(start=0)
print(f"\nНайденный лучший путь: {best_path}")
print(f"Суммарная стоимость: {best_cost}")
print(f"Количество рассмотренных узлов: {solver.count}")""")

def djicstra():
    print("""
import networkx as nx

def dijkstra_algorithm(graph, start_node):
    '''
    Алгоритм Дейкстры для нахождения кратчайших путей от стартового узла до всех остальных узлов в графе.

    :param graph: Граф NetworkX (взвешенный).
    :param start_node: Стартовый узел.
    :return: Словарь с минимальными расстояниями до каждого узла и предыдущими узлами в пути.
    '''
    # Инициализация словарей расстояний и предыдущих узлов
    distances = {node: float('inf') for node in graph.nodes()}
    previous_nodes = {node: None for node in graph.nodes()}
    distances[start_node] = 0

    # Множество посещенных узлов
    visited = set()

    # Пока есть непосещенные узлы
    while len(visited) < len(graph.nodes()):
        # Выбираем непосещенный узел с минимальным расстоянием
        unvisited_nodes = {node: distances[node] for node in graph.nodes() if node not in visited}
        if not unvisited_nodes:
            break  # Остались недостижимые узлы
        current_node = min(unvisited_nodes, key=unvisited_nodes.get)
        visited.add(current_node)

        # Обновляем расстояния до соседних узлов
        for neighbor in graph.neighbors(current_node):
            edge_weight = graph[current_node][neighbor].get('weight', 1)
            new_distance = distances[current_node] + edge_weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_node

    return distances, previous_nodes

def get_path(previous_nodes, start_node, target_node):
    '''
    Восстанавливает путь от стартового узла до целевого узла на основе информации о предыдущих узлах.

    :param previous_nodes: Словарь предыдущих узлов.
    :param start_node: Стартовый узел.
    :param target_node: Целевой узел.
    :return: Список узлов, представляющий кратчайший путь.
    '''
    path = []
    current_node = target_node
    while current_node != start_node:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
        if current_node is None:
            return None  # Пути нет
    path.insert(0, start_node)
    return path

# Пример использования
if __name__ == "__main__":
    G = nx.Graph()

    # Добавление ребер с весами
    G.add_edge('A', 'B', weight=4)
    G.add_edge('A', 'C', weight=2)
    G.add_edge('B', 'C', weight=1)
    G.add_edge('B', 'D', weight=5)
    G.add_edge('C', 'D', weight=8)
    G.add_edge('C', 'E', weight=10)
    G.add_edge('D', 'E', weight=2)
    G.add_edge('D', 'F', weight=6)
    G.add_edge('E', 'F', weight=2)

    start_node = 'A'
    distances, previous_nodes = dijkstra_algorithm(G, start_node)

    print(f"Кратчайшие расстояния от узла {start_node}:")
    for node, distance in distances.items():
        print(f"До узла {node}: расстояние = {distance}")

    # Пример получения пути от стартового узла до узла 'F'
    target_node = 'F'
    path = get_path(previous_nodes, start_node, target_node)
    print(f"\nКратчайший путь от узла {start_node} до узла {target_node}: {path}")""")

def prima():
    print("""
def prim_mst(graph):
    start_vertex = list(graph.nodes())[0]
    mst_edges = []  # Ребра остовного дерева
    visited = set([start_vertex])  # Посещенные вершины

    # Накапливаем все доступные ребра
    edges = [
        (start_vertex, neighbor, graph[start_vertex][neighbor]['weight'])
        for neighbor in graph.neighbors(start_vertex)
    ]

    # Пока есть непосещенные вершины
    while len(visited) < graph.number_of_nodes():
        # Найдем ребро с минимальным весом
        min_edge = min(edges, key=lambda x: x[2])
        edges.remove(min_edge)

        # Если одна из вершин еще не посещена, добавляем ее в дерево
        vertex = min_edge[1] if min_edge[0] in visited else min_edge[0]
        
        if vertex not in visited:
            visited.add(vertex)
            mst_edges.append(min_edge)
            
            # Добавляем новые доступные ребра из добавленной вершины
            for neighbor in graph.neighbors(vertex):
                if neighbor not in visited:
                    edges.append((vertex, neighbor, graph[vertex][neighbor]['weight']))

    return mst_edges""")

def bellman():
    print("""
import networkx as nx

def bellman_ford_algorithm(graph, source):
    '''
    Алгоритм Беллмана-Форда для нахождения кратчайших путей от источника до всех остальных узлов в графе.

    :param graph: Граф NetworkX (может содержать отрицательные веса ребер).
    :param source: Узел-источник.
    :return: Кортеж из двух элементов:
             - distances: Словарь минимальных расстояний до каждого узла.
             - predecessors: Словарь предшествующих узлов для восстановления путей.
    '''
    # Инициализация
    distances = {node: float('inf') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    distances[source] = 0

    # Основной цикл алгоритма
    for _ in range(len(graph.nodes()) - 1):
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 1)
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u

    # Проверка на наличие отрицательных циклов
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1)
        if distances[u] + weight < distances[v]:
            raise nx.NetworkXUnbounded(
                f"Обнаружен отрицательный цикл, достигаемый из узла {source}"
            )

    return distances, predecessors

def get_path(predecessors, source, target):
    '''
    Восстанавливает путь от источника до целевого узла на основе информации о предшественниках.

    :param predecessors: Словарь предшествующих узлов.
    :param source: Узел-источник.
    :param target: Целевой узел.
    :return: Список узлов, представляющий кратчайший путь.
    '''
    path = []
    current_node = target
    while current_node != source:
        if current_node is None:
            return None  # Путь не существует
        path.insert(0, current_node)
        current_node = predecessors[current_node]
    path.insert(0, source)
    return path

# Пример использования
if __name__ == "__main__":
    G = nx.DiGraph()

    # Добавление ребер с весами, включая отрицательные
    G.add_edge('A', 'B', weight=4)
    G.add_edge('A', 'C', weight=2)
    G.add_edge('B', 'C', weight=-1)
    G.add_edge('B', 'D', weight=2)
    G.add_edge('C', 'D', weight=3)
    G.add_edge('C', 'E', weight=2)
    G.add_edge('D', 'E', weight=-3)
    G.add_edge('E', 'F', weight=2)
    G.add_edge('D', 'F', weight=3)

    source = 'A'

    try:
        distances, predecessors = bellman_ford_algorithm(G, source)
        print(f"Кратчайшие расстояния от узла {source}:")
        for node, distance in distances.items():
            print(f"До узла {node}: расстояние = {distance}")

        # Пример восстановления пути до узла 'F'
        target = 'F'
        path = get_path(predecessors, source, target)
        print(f"\nКратчайший путь от узла {source} до узла {target}: {path}")

    except nx.NetworkXUnbounded as e:
        print(e)""")

def kruskal():
    print("""
def kruskal(G):
    edges = set(sorted(G.edges(data='weight'), key=lambda x: x[2]))
    num_nodes = G.number_of_nodes()
    G_tree = nx.Graph()
    excluded_edges = []
    nodes_in_tree = G_tree.number_of_nodes()
    while nodes_in_tree < num_nodes:
        for edge in edges:
            if edge in excluded_edges:
                continue
            G_tree.add_edge(edge[0], edge[1])
            try:
                nx.find_cycle(G_tree)
                G_tree.remove_edge(edge[0], edge[1])
                excluded_edges.append(edge)
                continue
            except nx.NetworkXNoCycle:
                excluded_edges.append(edge)
                nodes_in_tree = G_tree.number_of_nodes()
            
    return G_tree""")

def floyd_warshall():
    print("""
import networkx as nx

def floyd_warshall_all_pairs_shortest_paths(G):
    '''
    Implements the Floyd-Warshall algorithm to find the shortest paths between all pairs of nodes in a graph G.

    Parameters:
    G (networkx.Graph): An undirected or directed weighted graph.

    Returns:
    distance (dict): A dictionary of shortest path distances between all pairs of nodes.
    predecessor (dict): A dictionary of predecessors for each node in the shortest path.
    '''
    # Initialize distance and predecessor dictionaries
    distance = {u: {v: float('inf') for v in G.nodes()} for u in G.nodes()}
    predecessor = {u: {v: None for v in G.nodes()} for u in G.nodes()}

    # Set the distance from each node to itself to zero
    for node in G.nodes():
        distance[node][node] = 0

    # Initialize distances based on edge weights
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)  # Default weight is 1 if not specified
        distance[u][v] = weight
        predecessor[u][v] = u
        if not G.is_directed():
            distance[v][u] = weight
            predecessor[v][u] = v

    # Floyd-Warshall algorithm
    for k in G.nodes():
        for i in G.nodes():
            for j in G.nodes():
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
                    predecessor[i][j] = predecessor[k][j]

    return distance, predecessor


G = nx.Graph()
G.add_edge('A', 'B', weight=3)
G.add_edge('A', 'C', weight=10)
G.add_edge('B', 'C', weight=1)
G.add_edge('B', 'D', weight=2)
G.add_edge('C', 'D', weight=4)

# Compute shortest paths
distance, predecessor = floyd_warshall_all_pairs_shortest_paths(G)

# Print the shortest path distances
print("Shortest path distances between all pairs of nodes:")
for u in G.nodes():
    for v in G.nodes():
        print(f"Distance from {u} to {v}: {distance[u][v]}")

nx.floyd_warshall_predecessor_and_distance(G)""")

def check_balanced_graph():
    print("""
def is_balanced_signed_graph(graph):
    # Словарь для хранения классов вершин
    color = {}
    
    def bfs_check(start):
        queue = [start]
        color[start] = 0
        
        while queue:
            node = queue.pop(0)
            current_color = color[node]
            
            # Проверяем соседей
            for neighbor, edge_data in graph[node].items():
                if edge_data['weight'] == 1:  # Ребро с плюсом
                    if neighbor in color:
                        if color[neighbor] != current_color:
                            return False
                    else:
                        color[neighbor] = current_color
                        queue.append(neighbor)
                else:  # Ребро с минусом
                    if neighbor in color:
                        if color[neighbor] == current_color:
                            return False
                    else:
                        color[neighbor] = 1 - current_color
                        queue.append(neighbor)
        
        return True

    for node in graph.nodes:
        if node not in color:
            if not bfs_check(node):
                return False

    return True
""")

def ford_fulkerson():
    print("""
def dfs_find_path(G, source, sink, path, visited):
    if source == sink:
        return path
    visited.add(source)
    for neighbor in G[source]:
        capacity = G[source][neighbor]['capacity']
        if capacity > 0 and neighbor not in visited:
            result = dfs_find_path(G, neighbor, sink, path + [(source, neighbor)], visited)
            if result is not None:
                return result
    return None
def ford_fulkerson(G, source, sink):
    residual_graph = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        residual_graph.add_edge(u, v, capacity=data['weight'])
        residual_graph.add_edge(v, u, capacity=0)

    max_flow = 0
    while True:
        visited = set()
        path = dfs_find_path(residual_graph, source, sink, [], visited)

        if path is None:
            break

        flow = min(residual_graph[u][v]['capacity'] for u, v in path)

        for u, v in path:
            residual_graph[u][v]['capacity'] -= flow
            residual_graph[v][u]['capacity'] += flow

        max_flow += flow

    return max_flow
""")

def chineese_postman():
    print("""
import networkx as nx

def chinese_postman_problem(G):
    '''
    Реализует алгоритм для решения задачи о китайском почтальоне.
    Нахождение цикла минимальной длины, проходящего по всем ребрам графа.

    Параметры:
    G (networkx.Graph): Неориентированный взвешенный граф.

    Возвращает:
    circuit (list): Список ребер, представляющий эйлеров цикл.
    '''
    # Проверяем, является ли граф эйлеровым
    if nx.is_eulerian(G):
        # Если граф эйлеров, находим эйлеров цикл
        circuit = list(nx.eulerian_circuit(G))
    else:
        # Создаем копию графа в виде MultiGraph для возможности дублирования ребер
        G_aug = nx.MultiGraph(G.copy())

        # Находим узлы с нечетной степенью
        odd_degree_nodes = [v for v, d in G_aug.degree() if d % 2 == 1]

        # Создаем полный граф на узлах с нечетной степенью с весами кратчайших путей между ними
        import itertools
        odd_graph = nx.Graph()
        for u, v in itertools.combinations(odd_degree_nodes, 2):
            # Находим длину кратчайшего пути между узлами u и v
            length = nx.dijkstra_path_length(G_aug, u, v, weight='weight')
            odd_graph.add_edge(u, v, weight=length)

        # Находим паросочетание минимального веса
        min_weight_matching = nx.min_weight_matching(odd_graph, weight='weight')

        # Добавляем дополнительные ребра из паросочетания в граф G_aug
        for u, v in min_weight_matching:
            # Находим кратчайший путь между u и v в исходном графе G_aug
            path = nx.dijkstra_path(G_aug, u, v, weight='weight')
            # Добавляем ребра пути в G_aug с учетом весов
            for i in range(len(path) - 1):
                u1, v1 = path[i], path[i + 1]
                weight = G_aug[u1][v1][0]['weight']  # Получаем вес ребра
                G_aug.add_edge(u1, v1, weight=weight)  # Добавляем дополнительное ребро

        # Теперь граф G_aug эйлеров, находим эйлеров цикл
        circuit = list(nx.eulerian_circuit(G_aug))

    return circuit


G = nx.Graph()
G.add_edge('A', 'B', weight=3)
G.add_edge('A', 'C', weight=4)
G.add_edge('B', 'C', weight=1)
G.add_edge('B', 'D', weight=2)
G.add_edge('C', 'D', weight=4)
G.add_edge('C', 'E', weight=2)
G.add_edge('D', 'E', weight=3)
G.add_edge('D', 'F', weight=3)
G.add_edge('E', 'F', weight=2)

# Решаем задачу о китайском почтальоне
circuit = chinese_postman_problem(G)

# Выводим эйлеров цикл
print("Маршрут для китайского почтальона (эйлеров цикл):")
for u, v in circuit:
    print(f"{u} -> {v}")""")

def min_cost_flow():
    print("""
from collections import defaultdict, deque

def min_cost_max_flow(graph, capacity, cost, source, sink):
    flow = 0
    flow_cost = 0
    
    while True:
        dist = {i: float('inf') for i in graph}
        dist[source] = 0
        parent = {i: None for i in graph}
        in_queue = {i: False for i in graph}
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            
            for v in graph[u]:
                if capacity[(u, v)] > 0 and dist[v] > dist[u] + cost[(u, v)]:
                    dist[v] = dist[u] + cost[(u, v)]
                    parent[v] = u
                    
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        
        if dist[sink] == float('inf'):
            break
        
        # Finding the maximum flow through the path found.
        flow_through_path = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            flow_through_path = min(flow_through_path, capacity[(u, v)])
            v = u
        
        # Updating residual capacities and flow.
        v = sink
        while v != source:
            u = parent[v]
            capacity[(u, v)] -= flow_through_path
            capacity[(v, u)] += flow_through_path
            flow_cost += flow_through_path * cost[(u, v)]
            v = u
        
        flow += flow_through_path
    
    return flow, flow_cost""")

def dominating_set():
    print("""
import networkx as nx

def greedy_dominating_set(graph):
    '''
    Жадный алгоритм для нахождения доминирующего множества в графе.

    graph - неориентированный граф (NetworkX graph).

    Возвращает доминирующее множество.
    '''
    dominating_set = set()  # Множество для хранения вершин доминирующего множества
    covered = set()  # Множество охваченных вершин

    # Пока не все вершины охвачены
    while len(covered) < len(graph.nodes):
        # Ищем вершину, которая покрывает максимальное количество непокрытых соседей
        max_covering_vertex = None
        max_cover_count = -1

        for vertex in graph.nodes:
            # Пропускаем вершины, которые уже в доминирующем множестве или уже покрыты
            if vertex in dominating_set or vertex in covered:
                continue

            # Считаем, сколько новых непокрытых вершин может покрыть эта вершина
            neighbors = set(graph.neighbors(vertex))
            uncovered_neighbors = neighbors - covered

            if len(uncovered_neighbors) > max_cover_count:
                max_cover_count = len(uncovered_neighbors)
                max_covering_vertex = vertex

        # Добавляем выбранную вершину в доминирующее множество
        dominating_set.add(max_covering_vertex)
        # Обновляем множество охваченных вершин
        covered.add(max_covering_vertex)
        covered.update(graph.neighbors(max_covering_vertex))

    return dominating_set""")

def wattz_strogatz():
    print("""
def watts_strogatz_graph(n, k, p):
    '''
    Создание сети Уоттса-Строгаца.

    :param n: Количество вершин в графе
    :param k: Число соседей для каждой вершины в начальной кольцевой сети
    :param p: Вероятность переподключения каждого ребра
    :return: Граф, сгенерированный по модели Уоттса-Строгаца
    '''
    # Шаг 1: Создание кольцевой сети
    G = nx.Graph()

    for i in range(n):
        for j in range(1, k // 2 + 1):
            G.add_edge(i, (i + j) % n)  # Соединяем с соседями по кольцу

    # Шаг 2: Замена ребер с вероятностью p
    for u, v in list(G.edges()):
        if random.random() < p:
            # Выбираем случайную вершину для переподключения
            new_v = random.choice([x for x in range(n) if x != u and x != v])
            G.add_edge(u, new_v)
            G.remove_edge(u, v)

    return G""")

def barabasi_albert():
    print("""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def barabasi_albert_model(n, m):
    # Создаем начальный полный граф с m узлами
    G = nx.complete_graph(m)
    
    # Добавляем новые узлы
    for i in range(m, n):
        # Получаем список узлов и их степени
        degrees = np.array([G.degree(node) for node in G.nodes()])
        
        # Расчет вероятности подключения
        prob = degrees / degrees.sum()
        
        # Выбор m узлов на основе вероятности
        new_edges = np.random.choice(G.nodes(), size=m, replace=False, p=prob)
        
        # Добавляем новый узел и его связи
        G.add_node(i)
        for edge in new_edges:
            G.add_edge(i, edge)
    
    return G

# Пример использования
n = 100  # Количество узлов
m = 5    # Количество связей, создаваемых новым узлом

graph = barabasi_albert_model(n, m)

# Визуализация графа
plt.figure(figsize=(10, 10))
nx.draw(graph, node_size=30, with_labels=False)
plt.show()
""")

def fleri():
    print("""
import networkx as nx

def is_bridge(graph, u, v):
    '''
    Проверка, является ли ребро (u, v) мостом.
    '''
    # Удаляем ребро
    graph.remove_edge(u, v)
    # Проверяем, связан ли граф
    is_connected = nx.is_connected(graph)
    # Восстанавливаем ребро
    graph.add_edge(u, v)
    return not is_connected

def fleury_algorithm(graph):
    '''
    Реализация алгоритма Флери для построения Эйлерова цикла.
    Возвращает список рёбер, которые составляют Эйлеров цикл.
    '''
    # Проверяем, что все вершины имеют чётную степень
    for node in graph.nodes():
        if graph.degree(node) % 2 != 0:
            raise ValueError("Граф не содержит Эйлерова цикла (не все вершины имеют чётную степень).")

    # Копируем граф, чтобы не модифицировать исходный
    graph_copy = graph.copy()

    # Начинаем с произвольной вершины
    current_node = list(graph_copy.nodes())[0]

    # Список рёбер, который будет содержать Эйлеров цикл
    euler_cycle = []

    # Алгоритм Флери
    while graph_copy.edges():
        # Проверяем соседей текущей вершины
        for neighbor in list(graph_copy.neighbors(current_node)):
            # Если не мост, идем по ребру
            if not is_bridge(graph_copy, current_node, neighbor):
                next_node = neighbor
                break
        else:
            # Если все рёбра - мосты, то используем мост
            next_node = next(graph_copy.neighbors(current_node))

        # Добавляем ребро в Эйлеров цикл
        euler_cycle.append((current_node, next_node))
        # Удаляем ребро из графа
        graph_copy.remove_edge(current_node, next_node)
        # Переходим в следующую вершину
        current_node = next_node

        # Удаляем изолированные вершины
        isolated_nodes = [node for node in graph_copy.nodes() if graph_copy.degree(node) == 0]
        graph_copy.remove_nodes_from(isolated_nodes)

    return euler_cycle""")

def vengerian():
    print("""
import networkx as nx

def dfs(u, match_u, match_v, visited, graph):
    # Рекурсивный поиск для нахождения увеличивающего пути и расширения паросочетания.
    for v in graph[u]:
        if not visited[v]:
            visited[v] = True
            # Если v не связано с каким-либо элементом из множества U или
            # если мы можем найти увеличивающее паросочетание для паросочетания match_v[v]
            if match_v[v] == -1 or dfs(match_v[v], match_u, match_v, visited, graph):
                match_u[u] = v
                match_v[v] = u
                return True
    return False

def hungarian_algorithm(graph):
    # Разделим вершины на два множества U и V
    nodes = list(graph.nodes())
    n = len(nodes)
    mid = n // 2  # для простоты предположим, что граф двудольный

    U = nodes[:mid]
    V = nodes[mid:]

    # Массивы, которые будут хранить текущие паросочетания
    match_u = {u: -1 for u in U}  # match_u[u] - вершина из V, с которой вершина u из U связано
    match_v = {v: -1 for v in V}  # match_v[v] - вершина из U, с которой вершина v из V связано

    result = 0
    for u in U:
        # Массив для пометки посещенных вершин в графе
        visited = {v: False for v in V}
        if dfs(u, match_u, match_v, visited, graph):
            result += 1  # Если нашли увеличивающее паросочетание, увеличиваем результат

    # Формируем список пар паросочетания
    matching_pairs = [(u, match_u[u]) for u in U if match_u[u] != -1]

    return result, matching_pairs""")

def matrices():
    print("""
def adjacency_matrix(G):
    nodes = list(G.nodes())
    n = len(nodes)
    adj_matrix = np.zeros((n, n), dtype=int)
    for i, j in G.edges():
        idx_i = nodes.index(i)
        idx_j = nodes.index(j)
        adj_matrix[idx_i][idx_j] = 1
        if not G.is_directed():
            adj_matrix[idx_j][idx_i] = 1
    return adj_matrix

def incidence_matrix(G):
    m = len(G.edges)
    n = len(G.nodes)
    edge_dict = list(G.edges())
    inc_matrix = np.zeros((n, m), dtype=int)
    nodes = list(G.nodes())
    for idx, (u, v) in enumerate(edge_dict):
        u_idx = nodes.index(u)
        v_idx = nodes.index(v)

        inc_matrix[u_idx][idx] = 1
        if G.is_directed():
            inc_matrix[v_idx][idx] = -1
        else:
            inc_matrix[v_idx][idx] = 1
    return inc_matrix""")

