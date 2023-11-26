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

def compare_algorithms_time(time_marks, time_heap15):
    if time_marks < time_heap15:
        faster_algorithm = "Алгоритм с метками"
        time_difference = time_heap15 - time_marks
    elif time_heap15 < time_marks:
        faster_algorithm = "Алгоритм с 15-кучей"
        time_difference = time_marks - time_heap15
    else:
        return "Время выполнения алгоритмов одинаково"

    result_string = f"{faster_algorithm} быстрее на {time_difference:.6f} сек"
    return result_string

# Тест 1: Маленький полный граф с большими весами
num_vertices_small_full = 20
graph_small_full = generate_matrix(num_vertices_small_full, 100, 200)

print("Тест 1: Маленький полный граф с весом ребер от 100 до 200")
print("Исходный граф:")
print(graph_small_full)

t1_small_full = time.time()
heap_small_full = alg_Dijkstra_with_heap(graph_small_full, 0)
t2_small_full = time.time()

t3_small_full = time.time()
marks_small_full = alg_Dijkstra_with_marks(graph_small_full, 0)
t4_small_full = time.time()

print("Время выполнения с 15-кучей:", t2_small_full - t1_small_full, "сек")
print("Время выполнения с метками:", t4_small_full - t3_small_full, "сек")
result = compare_algorithms_time(t4_small_full - t3_small_full, t2_small_full - t1_small_full)
print(result);
print("\n")

# Тест 2: Большой граф с единичными весами
num_vertices_large_sparse = 200
graph_large = generate_matrix(num_vertices_large_sparse, 1, 2)

print("Тест 2: Большой полный граф с 1 весом ребер")
print("Исходный граф:")
print(graph_large)

t1_large = time.time()
heap_large = alg_Dijkstra_with_heap(graph_large, 0)
t2_large = time.time()

t3_large = time.time()
marks_large = alg_Dijkstra_with_marks(graph_large, 0)
t4_large = time.time()

print("Время выполнения с 15-кучей:", t2_large - t1_large, "сек")
print("Время выполнения с метками:", t4_large - t3_large, "сек")
result = compare_algorithms_time(t4_large - t3_large, t2_large - t1_large)
print(result);
print("\n")

# Тест 3: Большой полный граф с большими весами
num_vertices_large_full = 200
graph_large_full = generate_matrix(num_vertices_large_full, 100, 200)

print("Тест 3: Большой полный граф с весом ребер от 100 до 200")
print("Исходный граф:")
print(graph_large_full)

t1_large_full = time.time()
heap_large_full = alg_Dijkstra_with_heap(graph_large_full, 0)
t2_large_full = time.time()

t3_large_full = time.time()
marks_large_full = alg_Dijkstra_with_marks(graph_large_full, 0)
t4_large_full = time.time()

print("Время выполнения с 15-кучей:", t2_large_full - t1_large_full, "сек")
print("Время выполнения с метками:", t4_large_full - t3_large_full, "сек")
result = compare_algorithms_time(t4_large_full - t3_large_full, t2_large_full - t1_large_full)
print(result);