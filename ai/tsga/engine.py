"""
Asynchronous spike propagation engine for SSIS.
O(1) event-driven anomaly processing for orbital deviations.
"""

import numpy as np
from collections import deque
from typing import List, Dict, Callable

class AsynchronousSpikeEngine:
    """
    High-performance event scheduler for asynchronous SNNs.
    """
    def __init__(self, buffer_size: int = 2000, dt: float = 0.001):
        self.buffer_size = buffer_size
        self.dt = dt
        self.current_step = 0
        self.bins: List[List[int]] = [[] for _ in range(buffer_size)]
        self.callbacks: Dict[int, Callable] = {}

    def schedule_spike(self, neuron_id: int, delay_steps: int):
        target_step = (self.current_step + delay_steps) % self.buffer_size
        self.bins[target_step].append(neuron_id)

    def register_callback(self, neuron_id: int, callback: Callable):
        self.callbacks[neuron_id] = callback

    def step(self):
        current_idx = self.current_step % self.buffer_size
        firing_now = self.bins[current_idx]
        
        for neuron_id in firing_now:
            if neuron_id in self.callbacks:
                self.callbacks[neuron_id]()
        
        self.bins[current_idx] = []
        self.current_step += 1

class TSGAGraphEngine:
    """
    Engine for propagating spikes across a satellite constellation graph.
    """
    def __init__(self, n_neurons: int, adjacency: np.ndarray, base_delay: int = 5):
        self.n = n_neurons
        self.adj = adjacency
        self.base_delay = base_delay
        self.engine = AsynchronousSpikeEngine(buffer_size=5000)
        
        for i in range(n_neurons):
            self.engine.register_callback(i, lambda nid=i: self._on_spike(nid))

    def _on_spike(self, source_id: int):
        neighbors = np.where(self.adj[source_id] > 0)[0]
        for target_id in neighbors:
            self.engine.schedule_spike(target_id, self.base_delay)

    def trigger_input(self, neuron_ids: List[int]):
        for nid in neuron_ids:
            self.engine.schedule_spike(nid, 0)

    def step(self):
        self.engine.step()

if __name__ == "__main__":
    # Test: Fire a spike and see if it propagates
    adj = np.zeros((3, 3))
    adj[0, 1] = 1
    adj[1, 2] = 1
    
    graph = TSGAGraphEngine(3, adj)
    graph.engine.register_callback(0, lambda: print("Neuron 0 fired"))
    graph.engine.register_callback(1, lambda: print("Neuron 1 fired"))
    graph.engine.register_callback(2, lambda: print("Neuron 2 fired"))
    
    graph.trigger_input([0])
    for i in range(15):
        print(f"Step {i}")
        graph.step()
