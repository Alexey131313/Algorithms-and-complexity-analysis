#Copyright Pivovarov Alexey 2023

import matplotlib.pyplot as plt
import numpy as np
import time
import random

def generate_matrix(num_vertices, min_weight, max_weight):
    adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

    for i in range(num_vertices - 1):
        weights = np.random.randint(min_weight, max_weight, size=(num_vertices - i - 1))
        adj_matrix[i, i + 1:] = weights
        adj_matrix[i + 1:, i] = weights
        adj_matrix[i, i + 1:] *= np.random.choice([1, 0], size=(num_vertices - i - 1))

    return adj_matrix

class Node:
    def __init__(self, vertex, weight):
        self.vertex = vertex
        self.weight = weight

class Heap15:
    def __init__(self):
        self.heap = []

    def insert(self, node):
        self.heap.append(node)
        self.__sift_up(len(self.heap) - 1)

    def extract_min(self):
        if not self.heap:
            return None

        root = self.heap[0]
        last_node = self.heap.pop()

        if self.heap:
            self.heap[0] = last_node
            self.__sift_down(0)

        return root

    def __sift_up(self, index):
        parent = (index - 1) // 15

        while parent >= 0 and self.heap[parent].weight > self.heap[index].weight:
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            index, parent = parent, (parent - 1) // 15

    def __sift_down(self, index):
        smallest = index

        while True:
            for i in range(1, 16):
                child = 15 * index + i

                if child < len(self.heap) and self.heap[child].weight < self.heap[smallest].weight:
                    smallest = child

            if smallest != index:
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break

def alg_Dijkstra_with_heap(graph, start_vertex):
    vertex_count = len(graph)
    marks = np.full(vertex_count, float('inf'), dtype=float)
    marks[start_vertex] = 0
    visited = np.zeros(vertex_count, dtype=bool)
    heap = Heap15()

    heap.insert(Node(start_vertex, 0))

    while heap.heap:
        cur_node = heap.extract_min()
        cur_vertex = cur_node.vertex
        visited[cur_vertex] = True

        for v in range(vertex_count):
            if graph[cur_vertex, v] > 0 and not visited[v] and marks[v] > marks[cur_vertex] + graph[cur_vertex, v]:
                marks[v] = marks[cur_vertex] + graph[cur_vertex, v]
                heap.insert(Node(v, marks[v]))

    return marks

def search_min_mark(marks, visited):
    min_mark = float('inf')
    min_index = 0

    for i in range(len(marks)):
        if marks[i] < min_mark and not visited[i]:
            min_mark = marks[i]
            min_index = i

    return min_index

def alg_Dijkstra_with_marks(graph, start_vertex):
    vertex_count = len(graph)
    marks = np.full(vertex_count, float('inf'), dtype=float)
    marks[start_vertex] = 0
    visited = np.zeros(vertex_count, dtype=bool)

    for _ in range(vertex_count - 1):
        cur_vertex = search_min_mark(marks, visited)
        visited[cur_vertex] = True

        for j in range(vertex_count):
            if graph[cur_vertex, j] > 0 and not visited[j] and marks[j] > marks[cur_vertex] + graph[cur_vertex, j]:
                marks[j] = marks[cur_vertex] + graph[cur_vertex, j]

    return marks

# Генерация данных для тестирования
M = 50
x1 = np.arange(1, M * 200 + 1, 200)
matrix_list = []

# Генерация матриц
for num_vertices in x1:
    matrix_list.append(generate_matrix(num_vertices, 30, 40))

# Подготовка данных для графика
y1 = np.zeros(M)
y2 = np.zeros(M)

# Тестирование и измерение времени выполнения
for i, matrix in enumerate(matrix_list):
    t1 = time.time()
    alg_Dijkstra_with_marks(matrix, 0)
    t2 = time.time()
    y1[i] = t2 - t1

    t3 = time.time()
    alg_Dijkstra_with_heap(matrix, 0)
    t4 = time.time()
    y2[i] = t4 - t3

# Фильтрация точек, где времена совпадают
eps = 0.001
points = np.isclose(y1, y2, atol=eps)
x = x1[points]
y = (np.array(y1) + np.array(y2)) / 2
match_y = y[points]


# Построение графика
plt.plot(x1, y1, label='С метками')
plt.plot(x1, y2, label='15 куча')
plt.plot(x, match_y, 'ro', label='Точки совпадения')
plt.xlabel('Число вершин')
plt.ylabel('Время выполнения, с')
plt.title('Сравнение алгоритмов для полных графов')
plt.legend()
plt.show()

