# Problem: Implement the Breadth-First Search (BFS), Depth-First Search (DFS)
# and Greedy Best-First Search (GBFS) algorithms on the graph from Figure 1 in hw1.pdf.


# Instructions:
# 1. Represent the graph from Figure 1 in any format (e.g. adjacency matrix, adjacency list).
# 2. Each function should take in the starting node as a string. Assume the search is being performed on
#    the graph from Figure 1.
#    It should return a list of all node labels (strings) that were expanded in the order they where expanded.
#    If there is a tie for which node is expanded next, expand the one that comes first in the alphabet.
# 3. You should only modify the graph representation and the function body below where indicated.
# 4. Do not modify the function signature or provided test cases. You may add helper functions.
# 5. Upload the completed homework to Gradescope, it must be named 'hw1.py'.

# Examples:
#     The test cases below call each search function on node 'S' and node 'A'
# -----------------------------

from collections import deque
from heapq import heappop, heappush


class Graph:
    """
    Essentially a dictionary where {a: b} == {b: a}
    """
    adjacency_map: dict[str, dict[str, int]]

    def __init__(self, adjacency_list: list[tuple[str, str, int]]) -> None:
        self.adjacency_map = {}
        for entry in adjacency_list:
            self.set(entry[0], entry[1], entry[2])

    def set(self, a: str, b: str, x: int):
        if self.adjacency_map.get(a) is None:
            self.adjacency_map[a] = {}
        if self.adjacency_map.get(b) is None:
            self.adjacency_map[b] = {}
        self.adjacency_map[a][b] = x
        self.adjacency_map[b][a] = x

    def get_edges(self, key: str) -> list[tuple[str, int]]:
        list = [(key, 0)]
        value = self.adjacency_map.get(key, {})
        for key2 in value:
            list.append((key2, value[key2]))

        return list

graph = Graph([
    ('A', 'B', 4),
    ('A', 'E', 1),
    ('B', 'C', 2),
    ('B', 'F', 2),
    ('C', 'H', 4),
    ('C', 'S', 3),
    ('D', 'L', 8),
    ('D', 'S', 2),
    ('E', 'F', 3),
    ('E', 'I', 6),
    ('F', 'J', 6),
    ('F', 'K', 4),
    ('G', 'N', 4),
    ('G', 'M', 4),
    ('G', 'Q', 10),
    ('H', 'K', 3),
    ('H', 'L', 7),
    ('I', 'J', 1),
    ('I', 'M', 5),
    ('J', 'K', 3),
    ('J', 'N', 3),
    ('K', 'L', 9),
    ('K', 'P', 3),
    ('L', 'Q', 10),
    ('N', 'P', 2)
])

# Heuristic
h = {
    'S': 17,
    'A': 17,
    'B': 9,
    'C': 16,
    'D': 21,
    'E': 13,
    'F': 9,
    'G': 0,
    'H': 12,
    'I': 9,
    'J': 5,
    'K': 8,
    'L': 18,
    'M': 3,
    'N': 4,
    'P': 6,
    'Q': 9
}

def BFS(start: str) -> list:
    print(f"BFS from {start}")
    list = []
    visited = {start}
    queue = deque(start)
    while len(queue) > 0:
        current_edge = queue.popleft()
        list.append(current_edge)

        edges = graph.get_edges(current_edge)
        for (edge, _) in edges:
            if edge == 'G':
                list.append(edge)
                print(list)
                return list

            elif edge not in visited:
                print(f'{current_edge} -> {edge}')
                visited.add(edge)
                queue.append(edge)

    print(list)
    return list

# Same as BFS, but with a stack
def DFS(start: str) -> list:
    print(f"DFS from {start}")
    list = []
    visited = {start}
    stack = [start]
    while len(stack) > 0:
        current_edge = stack.pop()
        list.append(current_edge)

        edges = graph.get_edges(current_edge)
        for (edge, _) in edges:
            if edge == 'G':
                list.append(edge)
                print(list)
                return list

            elif edge not in visited:
                print(f'{current_edge} -> {edge}')
                visited.add(edge)
                stack.append(edge)
                break

    print(list)
    return list

# Similar to the previous two, but with a min heap sorted by h(n)
def GBFS(start: str) -> list:
    print(f"GBFS from {start}")
    list = []
    visited = {start}
    min_heap = [(h[start], start)]
    while len(min_heap) > 0:
        current_edge = heappop(min_heap)[1]
        list.append(current_edge)

        edges = graph.get_edges(current_edge)
        for (edge, _) in edges:
            if edge == 'G':
                list.append(edge)
                print(list)
                return list

            elif edge not in visited:
                print(f'{current_edge} -> {edge} ({h[edge]})')
                visited.add(edge)
                heappush(min_heap, (h[edge], edge))

    print(list)
    return list

# test cases - DO NOT MODIFY THESE
def run_tests():
    # Test case 1: BFS starting from node 'A'
    assert BFS('A') == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"

    # Test case 2: BFS starting from node 'S'
    assert BFS('S') == ['S', 'C', 'D', 'B', 'H', 'L', 'A', 'F', 'K', 'Q', 'G'], "Test case 2 failed"

    # Test case 3: DFS starting from node 'A'
    assert DFS('A') == ['A', 'B', 'C', 'H', 'K', 'F', 'E', 'I', 'J', 'N', 'G'], "Test case 3 failed"

    # Test case 4: DFS starting from node 'S'
    assert DFS('S') == ['S', 'C', 'B', 'A', 'E', 'F', 'J', 'I', 'M', 'G'], "Test case 4 failed"

    # Test case 5: GBFS starting from node 'A'
    assert GBFS('A') == ['A', 'B', 'F', 'J', 'N', 'G'], "Test case 5 failed"

    # Test case 6: GBFS starting from node 'S'
    assert GBFS('S') == ['S', 'C', 'B', 'F', 'J', 'N', 'G'], "Test case 6 failed"



    print("All test cases passed!")

if __name__ == '__main__':
    run_tests()
