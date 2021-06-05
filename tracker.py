import math

from .track import Track


class SSort:

    def __init__(self, init_step=3, min_hit=5, max_mis=15, cost_threshold=4):
        self.init_step = init_step
        self.min_hit = min_hit
        self.max_mis = max_mis
        self.cost_threshold = cost_threshold
        self.tracks = []
        self.next_track_id = 0

    def forward(self):
        for track in self.tracks:
            track.forward()

    def update(self, detections):
        obverved_tracks = []
        tracks_id = [-1 for _ in range(len(detections))]
        matched_pairs, _, unmatched_detections = self._matching(detections)
        for t, d in matched_pairs:
            self.tracks[t].update(detections[d])
            obverved_tracks.append(self.tracks[t])
            tracks_id[d] = self.tracks[t].id
        # Create new tracks
        for d in unmatched_detections:
            self.tracks.append(Track(self.next_track_id, detections[d]))
            tracks_id[d] = self.next_track_id
            self.next_track_id += 1
        deleted_tracks = []
        next_step_tracks = []
        for track in self.tracks:
            if track.age >= self.init_step:  # confirmed tracks
                if track.n_consecutive_mis > self.max_mis:
                    if track.n_hit >= self.min_hit:
                        deleted_tracks.append(track)
                else:
                    next_step_tracks.append(track)
            elif track.n_consecutive_mis == 0:  # not confirmed tracks
                next_step_tracks.append(track)
        self.tracks = next_step_tracks
        return obverved_tracks, deleted_tracks, tracks_id

    def _matching(self, detections):
        n_track = len(self.tracks)
        n_detection = len(detections)
        matched_pairs = []
        unmatched_tracks = list(range(n_track))
        unmatched_detections = list(range(n_detection))
        for depth in range(self.max_mis+1):
            if len(unmatched_detections) == 0 or len(unmatched_tracks) == 0:
                # completed
                break
            this_level_tracks = [t for t in range(
                n_track) if self.tracks[t].n_consecutive_mis == depth]
            if len(this_level_tracks) == 0:
                # nothing to do at this level
                continue
            # I do not use Hungarian algorithm because it's not simple and fast. I use a simple greedy algorithm
            cost_pairs = []
            for d in unmatched_detections:
                for t in this_level_tracks:
                    cost = SSort._cost_function(self.tracks[t], detections[d])
                    if cost <= self.cost_threshold:
                        cost_pairs.append((t, d, cost))
            cost_pairs = sorted(cost_pairs, key=lambda item: item[2])
            matched_d = set()
            matched_t = set()
            for t, d, cost in cost_pairs:
                if t not in matched_t and d not in matched_d:
                    matched_t.add(t)
                    matched_d.add(d)
                    matched_pairs.append((t, d))
            unmatched_tracks = [
                t for t in unmatched_tracks if t not in matched_t]
            unmatched_detections = [
                d for d in unmatched_detections if d not in matched_d]

        return matched_pairs, unmatched_tracks, unmatched_detections

    @staticmethod
    def _cost_function(track, detection):
        x1, y1, x2, y2 = detection[:4]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        s = (w + h) / 2
        tx, ty, tw, th = track.x, track.y, track.w, track.h
        ts = (tw + th) / 2
        avg_size = (s + ts) / 2
        distance = math.sqrt((x-tx)**2 + (y-ty)**2)
        distance = distance / avg_size
        w_ratio = w / tw - 1 if w > tw else tw / w - 1
        h_ratio = h / th - 1 if h > th else th / h - 1
        return 0.5 * distance + 0.25 * w_ratio + 0.25 * h_ratio
