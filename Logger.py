import sys
import torch
from Cluster import k_means, pairwise_distance


class IdentityLogger:
    def __init__(self, **kwargs):
        self.log_history_length = kwargs.get('log_history_length') or (sys.maxsize-2)//2
        self.centers = None
        self.pre_history = []
        self.history = []

    def pre_log(self, xywh: Tensor):
        box = xywh[:, 0:2]
        self.pre_history.append(box)

    def get_centers(self, num_iterations=100):
        if self.centers is not None:
            return self.centers
        num_people = round(sum([len(i) for i in self.pre_history]) / len(self.pre_history))
        for init_point in reversed(self.pre_history):
            if len(init_point) == num_people:
                break
        else:  # error
            raise RuntimeError(f"Cannot confirm the number of people {num_people}.")
        init_centers = init_point
        data_points = torch.cat(self.pre_history, dim=0)
        _, self.centers = k_means(init_centers, data_points, num_iterations)
        del self.pre_history[:]
        return self.centers

    def confirm_ids(self, xywh: Tensor):
        data_points = xywh[:, 0:2]
        init_centers = self.centers
        distances = pairwise_distance(data_points.unsqueeze(1), init_centers.unsqueeze(0), p=2)
        cluster_assignments = torch.argmin(distances, dim=1)
        return cluster_assignments

    def log(self, xywh: Tensor, state: Tensor, time: float):
        data_points = xywh[:, 0:2]
        cluster_assignments = self.confirm_ids(xywh=xywh)
        time = torch.zeros_like(state) + time
        self.history.append((cluster_assignments, data_points, state, time))
        if len(self.history) > self.log_history_length * 2:
            self.history = self.history[-self.log_history_length:]

    def get_scores(self, start_time=None, end_time=None):
        ids = torch.cat([i[0] for i in self.history], dim=0)
        states = torch.cat([i[2] for i in self.history], dim=0)
        time = torch.cat([i[3] for i in self.history], dim=0)
        if start_time is None:
            start_time = torch.min(time) - 1.0
        if end_time is None:
            end_time = torch.max(time) + 1.0
        scores = []
        for i in range(len(self.centers)):
            mask = torch.bitwise_and(ids == i, start_time < time < end_time)
            score = torch.masked_select(states, mask).mean().item()
            scores.append(score)
        return scores

    def get_full_history(self):
        return self.history.copy()





