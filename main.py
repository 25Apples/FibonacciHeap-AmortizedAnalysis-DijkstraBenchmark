import time
from collections import defaultdict

# ============================
# Cấu trúc Fibonacci Heap
# ============================
class FibonacciHeap:
    class Node:
        def __init__(self, key, value=None):
            self.key = key
            self.value = value
            self.degree = 0
            self.mark = False
            self.parent = None
            self.child = None
            self.left = self.right = self

        def __str__(self):
            return f"Node(key={self.key}, value={self.value})"

    def __init__(self):
        self.min_node = None
        self.total_nodes = 0

    def insert(self, key, value=None):
        """INSERT operation - O(1) amortized"""
        node = self.Node(key, value)
        if self.min_node is None:
            self.min_node = node
        else:
            self._add_to_root_list(node)
            if node.key < self.min_node.key:
                self.min_node = node
        self.total_nodes += 1
        return node

    def find_min(self):
        """FIND-MIN operation - O(1)"""
        return self.min_node

    def is_empty(self):
        """Check if heap is empty - O(1)"""
        return self.min_node is None

    def extract_min(self):
        """EXTRACT-MIN operation - O(log n) amortized"""
        z = self.min_node
        if z is not None:
            # Move all children of z to root list
            if z.child is not None:
                children = self._get_all_children(z.child)
                for child in children:
                    self._add_to_root_list(child)
                    child.parent = None

            # Remove z from root list
            self._remove_from_root_list(z)
            
            if z == z.right:  # z was the only node
                self.min_node = None
            else:
                self.min_node = z.right
                self._consolidate()
            
            self.total_nodes -= 1
        return z

    def decrease_key(self, x, k):
        """DECREASE-KEY operation - O(1) amortized"""
        if k > x.key:
            raise ValueError("New key is greater than current key")
        
        x.key = k
        y = x.parent
        
        if y is not None and x.key < y.key:
            self._cut(x, y)
            self._cascading_cut(y)
        
        if x.key < self.min_node.key:
            self.min_node = x

    def _cut(self, x, y):
        """Cut x from its parent y and add to root list"""
        self._remove_from_child_list(y, x)
        y.degree -= 1
        self._add_to_root_list(x)
        x.parent = None
        x.mark = False

    def _cascading_cut(self, y):
        """Perform cascading cut starting from y"""
        z = y.parent
        if z is not None:
            if not y.mark:
                y.mark = True
            else:
                self._cut(y, z)
                self._cascading_cut(z)

    def _consolidate(self):
        """Consolidate the heap to maintain Fibonacci heap properties"""
        max_degree = int(self.total_nodes.bit_length()) + 1
        A = [None] * max_degree
        
        # Get all root nodes first (to avoid modification during iteration)
        root_nodes = self._get_all_roots()
        
        for w in root_nodes:
            x = w
            d = x.degree
            
            # Consolidation loop with safety check
            consolidation_steps = 0
            while consolidation_steps < max_degree and d < len(A) and A[d] is not None:
                y = A[d]
                if x.key > y.key:
                    x, y = y, x
                self._heap_link(y, x)
                A[d] = None
                d += 1
                consolidation_steps += 1
            
            # Expand array if needed
            if d >= len(A):
                A.extend([None] * (d - len(A) + 1))
            A[d] = x

        # Rebuild root list and find new minimum
        self.min_node = None
        for node in A:
            if node is not None:
                if self.min_node is None:
                    self.min_node = node
                    node.left = node.right = node
                else:
                    self._add_to_root_list(node)
                    if node.key < self.min_node.key:
                        self.min_node = node

    def _heap_link(self, y, x):
        """Make y a child of x"""
        # Remove y from root list
        self._remove_from_root_list(y)
        
        # Make y a child of x
        if x.child is None:
            x.child = y
            y.left = y.right = y
        else:
            self._add_to_child_list(x, y)
        
        y.parent = x
        x.degree += 1
        y.mark = False

    def _add_to_root_list(self, node):
        """Add node to root list"""
        if self.min_node is None:
            self.min_node = node
            node.left = node.right = node
        else:
            node.right = self.min_node.right
            node.left = self.min_node
            self.min_node.right.left = node
            self.min_node.right = node

    def _remove_from_root_list(self, node):
        """Remove node from root list"""
        if node.right == node:  # Only node in root list
            return
        node.left.right = node.right
        node.right.left = node.left

    def _add_to_child_list(self, parent, child):
        """Add child to parent's child list"""
        if parent.child is None:
            parent.child = child
            child.left = child.right = child
        else:
            child.right = parent.child.right
            child.left = parent.child
            parent.child.right.left = child
            parent.child.right = child

    def _remove_from_child_list(self, parent, child):
        """Remove child from parent's child list"""
        if parent.child == child:
            if child.right == child:  # Only child
                parent.child = None
            else:
                parent.child = child.right
        
        child.left.right = child.right
        child.right.left = child.left

    def _get_all_children(self, start):
        """Get all children in a circular list starting from start"""
        if start is None:
            return []
        
        children = [start]
        current = start.right
        safety_counter = 0
        max_safety = self.total_nodes + 10  # Safety limit
        
        while current != start and safety_counter < max_safety:
            children.append(current)
            current = current.right
            safety_counter += 1
            
        if safety_counter >= max_safety:
            print("Warning: Potential infinite loop in _get_all_children")
            
        return children

    def _get_all_roots(self):
        """Get all nodes in root list"""
        if self.min_node is None:
            return []
        
        roots = [self.min_node]
        current = self.min_node.right
        safety_counter = 0
        max_safety = self.total_nodes + 10
        
        while current != self.min_node and safety_counter < max_safety:
            roots.append(current)
            current = current.right
            safety_counter += 1
            
        if safety_counter >= max_safety:
            print("Warning: Potential infinite loop in _get_all_roots")
            
        return roots

    def debug_structure(self):
        """Debug function to print heap structure"""
        print(f"🔍 Fibonacci Heap Debug (total_nodes: {self.total_nodes})")
        if self.min_node is None:
            print("   Empty heap")
            return
        
        print(f"   Min node: {self.min_node}")
        roots = self._get_all_roots()
        print(f"   Root list ({len(roots)} nodes):")
        for i, root in enumerate(roots):
            print(f"     [{i}] {root} (degree: {root.degree})")

# ============================
# Dijkstra với Fibonacci Heap
# ============================
class FibDijkstra:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edges(self, edges):
        for u, v, w in edges:
            self.graph[u].append((v, w))
            self.graph[v].append((u, w))

    def dijkstra(self, V, src):
        # Initialize
        dist = [float('inf')] * V
        dist[src] = 0
        visited = [False] * V
        
        # Create Fibonacci heap and node mapping
        heap = FibonacciHeap()
        nodes = {}
        
        for v in range(V):
            key = 0 if v == src else float('inf')
            nodes[v] = heap.insert(key, v)

        processed = 0
        while not heap.is_empty() and processed < V:
            # Extract minimum
            min_node = heap.extract_min()
            if min_node is None:
                break
                
            u = min_node.value
            if visited[u]:
                continue
                
            visited[u] = True
            dist[u] = min_node.key
            processed += 1

            # Update neighbors
            for v, weight in self.graph[u]:
                if not visited[v]:
                    new_dist = dist[u] + weight
                    if new_dist < nodes[v].key:
                        heap.decrease_key(nodes[v], new_dist)

        return dist

# ============================
# Cấu trúc Binary Heap
# ============================
class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, item):
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            return None
        self._swap(0, len(self.heap) - 1)
        min_item = self.heap.pop()
        self._sift_down(0)
        return min_item

    def _sift_up(self, idx):
        parent = (idx - 1) // 2
        while idx > 0 and self.heap[idx][0] < self.heap[parent][0]:
            self._swap(idx, parent)
            idx = parent
            parent = (idx - 1) // 2

    def _sift_down(self, idx):
        n = len(self.heap)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == idx:
                break
            self._swap(idx, smallest)
            idx = smallest

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def empty(self):
        return len(self.heap) == 0

# ============================
# Dijkstra với Binary Heap
# ============================
class BinaryDijkstra:
    def dijkstra(self, V, edges, src):
        # Build graph
        graph = [[] for _ in range(V)]
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))

        # Dijkstra with binary heap
        dist = [float('inf')] * V
        dist[src] = 0
        heap = MinHeap()
        heap.push((0, src))

        while not heap.empty():
            d, u = heap.pop()
            if d > dist[u]:  # Skip outdated entries
                continue
                
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heap.push((dist[v], v))

        return dist

# ============================
# Kiểm thử và so sánh kết quả
# ============================
def test_dijkstra():
    V = 5
    edges = [[0, 1, 4], [0, 2, 8], [1, 4, 6], [2, 3, 2], [3, 4, 10]]
    src = 0

    print("\n" + "="*50)
    print("KIỂM THỬ DIJKSTRA ALGORITHM")
    print("="*50)

    print("\n[Binary Heap] Running...")
    start_time = time.time()
    d_bin = BinaryDijkstra().dijkstra(V, edges, src)
    binary_time = time.time() - start_time
    print(f"[Binary Heap] Kết quả: {d_bin}")
    print(f"[Binary Heap] Thời gian: {binary_time:.6f}s")

    print("\n[Fibonacci Heap] Running...")
    start_time = time.time()
    fib_solver = FibDijkstra()
    fib_solver.add_edges(edges)
    d_fib = fib_solver.dijkstra(V, src)
    fib_time = time.time() - start_time
    print(f"[Fibonacci Heap] Kết quả: {d_fib}")
    print(f"[Fibonacci Heap] Thời gian: {fib_time:.6f}s")

    print("\n" + "="*50)
    print("SO SÁNH KẾT QUẢ")
    print("="*50)
    
    all_correct = True
    for i in range(V):
        if d_bin[i] != d_fib[i]:
            print(f"Sai tại đỉnh {i}: Binary = {d_bin[i]}, Fibonacci = {d_fib[i]}")
            all_correct = False
        else:
            print(f"TRUE: Đỉnh {i}: {d_bin[i]}")
    
    if all_correct:
        print(f"\nTHÀNH CÔNG! Cả hai thuật toán cho kết quả giống nhau!")
        print(f"Tỷ lệ thời gian: Fibonacci/Binary = {fib_time/binary_time:.2f}")
    else:
        print(f"\nCÒN LỖI! Kết quả không khớp.")

    return all_correct

# ============================
# Benchmark với tăng dần số đỉnh
# ============================
def benchmark_range(a, b):
    """
    Kiểm thử hiệu suất với số đỉnh tăng từ a đến b
    Tạo file MATLAB để vẽ biểu đồ
    """
    print("\n" + "="*60)
    print(f"BENCHMARK DIJKSTRA: {a} → {b} NODES")
    print("="*60)
    
    import random
    random.seed(42)  # Đảm bảo kết quả lặp lại
    
    # Danh sách lưu kết quả
    nodes_list = []
    binary_times = []
    fib_times = []
    
    for V in range(a, b + 1):
        print(f"\nTesting với {V} nodes...")
        
        # Tạo đồ thị ngẫu nhiên
        edges = []
        density = min(0.3, 50.0 / V)  # Giảm mật độ với đồ thị lớn
        
        for i in range(V):
            for j in range(i + 1, V):
                if random.random() < density:
                    weight = random.randint(1, 50)
                    edges.append([i, j, weight])
        
        src = 0
        print(f"Đồ thị: {V} đỉnh, {len(edges)} cạnh")
        
        # Test Binary Heap
        start_time = time.time()
        d_bin = BinaryDijkstra().dijkstra(V, edges, src)
        binary_time = time.time() - start_time
        
        # Test Fibonacci Heap  
        start_time = time.time()
        fib_solver = FibDijkstra()
        fib_solver.add_edges(edges)
        d_fib = fib_solver.dijkstra(V, src)
        fib_time = time.time() - start_time
        
        # Kiểm tra tính đúng
        correct = True
        for i in range(V):
            if abs(d_bin[i] - d_fib[i]) > 1e-9:
                correct = False
                break
        
        # Lưu kết quả
        nodes_list.append(V)
        binary_times.append(binary_time)
        fib_times.append(fib_time)
        
        # In kết quả terminal
        status = "TRUE" if correct else "FALSE"
        ratio = fib_time / binary_time if binary_time > 0 else float('inf')
        print(f"{status} | Binary: {binary_time:.6f}s | Fibonacci: {fib_time:.6f}s | Ratio: {ratio:.2f}")
    
    # Tạo file MATLAB
    create_matlab_plot(nodes_list, binary_times, fib_times)
    
    print("\n" + "="*60)
    print("TỔNG KẾT BENCHMARK")
    print("="*60)
    print(f"Đã test {len(nodes_list)} kích thước đồ thị")
    print(f"Binary Heap - Trung bình: {sum(binary_times)/len(binary_times):.6f}s")
    print(f"Fibonacci Heap - Trung bình: {sum(fib_times)/len(fib_times):.6f}s")
    print(f"File MATLAB đã được tạo: dijkstra_benchmark.m")
    
    return nodes_list, binary_times, fib_times


def create_matlab_plot(nodes, binary_times, fib_times):
    """Tạo file MATLAB để vẽ biểu đồ so sánh"""
    
    matlab_code = f"""% MATLAB Script - Dijkstra Algorithm Benchmark
% Generated automatically from Python benchmark

% Data
nodes = {nodes};
binary_times = {binary_times};
fib_times = {fib_times};

% Create figure
figure('Position', [100, 100, 1000, 600]);

% Plot lines
hold on;
plot(nodes, binary_times, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Binary Heap');
plot(nodes, fib_times, 'r-s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Fibonacci Heap');

% Formatting
xlabel('Number of Nodes (V)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Execution Time (seconds)', 'FontSize', 12, 'FontWeight', 'bold');
title('Dijkstra Algorithm Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 11);
grid on;

% Set axis properties
set(gca, 'FontSize', 10);
xlim([min(nodes)-1, max(nodes)+1]);
ylim([0, max(max(binary_times), max(fib_times))*1.1]);

% Add annotations
text(0.7, 0.9, sprintf('Max nodes: %d', max(nodes)), 'Units', 'normalized', ...
     'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
text(0.7, 0.85, sprintf('Total tests: %d', length(nodes)), 'Units', 'normalized', ...
     'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');

% Save figure
saveas(gcf, 'dijkstra_benchmark.png');
saveas(gcf, 'dijkstra_benchmark.fig');

fprintf('\\n=== MATLAB ANALYSIS ===\\n');
fprintf('Binary Heap - Min: %.6f, Max: %.6f, Avg: %.6f\\n', ...
        min(binary_times), max(binary_times), mean(binary_times));
fprintf('Fibonacci Heap - Min: %.6f, Max: %.6f, Avg: %.6f\\n', ...
        min(fib_times), max(fib_times), mean(fib_times));
fprintf('Average Ratio (Fib/Binary): %.2f\\n', mean(fib_times./binary_times));
"""
    
    # Ghi file MATLAB
    with open('dijkstra_benchmark.m', 'w') as f:
        f.write(matlab_code)
    
    print(f"File MATLAB đã được tạo: dijkstra_benchmark.m")
    print(f"Chạy trong MATLAB để xem biểu đồ: >> dijkstra_benchmark")


if __name__ == "__main__":
    # Kiểm thử cơ bản
    test_dijkstra()
    
    # Hỏi người dùng về range benchmark
    print("\n" + " "*20 + "BENCHMARK SETUP" + " "*20 + " ")
    print("="*60)
    
    try:
        user_input = input("Nhập khoảng số nodes (format: a:b): ").strip()
        
        if ':' in user_input:
            a, b = map(int, user_input.split(':'))
            if a >= 2 and b >= a and b <= 100000:  # Giới hạn an toàn
                print(f"Bắt đầu benchmark từ {a} đến {b} nodes...")
                benchmark_range(a, b)
            else:
                print("Range không hợp lệ! (2 ≤ a ≤ b ≤ 100000)")
        else:
            print("Format không đúng! Vui lòng dùng format a:b!")
            
    except (ValueError, KeyboardInterrupt):
        print("Input không hợp lệ hoặc đã hủy benchmark")
        print("Chạy lại với format: python script.py")
        print("Nhập 5:15 để kiểm tra lỗi")
    
    print("\Chương trình kết thúc!")
