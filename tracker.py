from track import Track


class SSort:

    def __init__(self, init_step=3, min_hit=5, max_mis=10, min_cost_threshold=0.66, max_cost_threshold=0.99):
        self.init_step = init_step
        self.min_hit = min_hit
        self.max_mis = max_mis
        self.min_cost_threshold = min_cost_threshold
        self.max_cost_threshold = max_cost_threshold
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
        unmatched_detections = list(range(n_detection))
        unmatched_confirmed_tracks = [t for t in range(n_track) if self.tracks[t].age >= self.init_step]
        unmatched_unconfirmed_tracks = [t for t in range(n_track) if self.tracks[t].age < self.init_step]

        # Matching confirmed tracks
        for depth in range(self.max_mis+1):
            cost_threshold = self.min_cost_threshold + depth * (self.max_cost_threshold - self.min_cost_threshold) / self.max_mis
            if len(unmatched_detections) == 0 or len(unmatched_confirmed_tracks) == 0:
                # completed
                break
            this_level_tracks = [t for t in unmatched_confirmed_tracks if self.tracks[t].n_consecutive_mis == depth]
            if len(this_level_tracks) == 0:
                # nothing to do at this level
                continue
            # I don't use Hungarian algorithm because it's not simple and fast. I use a simple greedy algorithm
            cost_pairs = []
            for d in unmatched_detections:
                for t in this_level_tracks:
                    cost = SSort._cost_function(self.tracks[t], detections[d])
                    if cost <= cost_threshold:
                        cost_pairs.append((t, d, cost))
            cost_pairs = sorted(cost_pairs, key=lambda item: item[2])
            matched_d = set()
            matched_t = set()
            for t, d, cost in cost_pairs:
                if t not in matched_t and d not in matched_d:
                    matched_t.add(t)
                    matched_d.add(d)
                    matched_pairs.append((t, d))
            unmatched_confirmed_tracks = [
                t for t in unmatched_confirmed_tracks if t not in matched_t]
            unmatched_detections = [
                d for d in unmatched_detections if d not in matched_d]

        # Matching unconfirmed tracks
        if len(unmatched_detections) > 0 and len(unmatched_unconfirmed_tracks) > 0:
            cost_pairs = []
            for d in unmatched_detections:
                for t in unmatched_unconfirmed_tracks:
                    cost = SSort._cost_function(self.tracks[t], detections[d])
                    if cost <= self.min_cost_threshold:
                        cost_pairs.append((t, d, cost))
            cost_pairs = sorted(cost_pairs, key=lambda item: item[2])
            matched_d = set()
            matched_t = set()
            for t, d, cost in cost_pairs:
                if t not in matched_t and d not in matched_d:
                    matched_t.add(t)
                    matched_d.add(d)
                    matched_pairs.append((t, d))
            unmatched_unconfirmed_tracks = [
                t for t in unmatched_unconfirmed_tracks if t not in matched_t]
            unmatched_detections = [
                d for d in unmatched_detections if d not in matched_d]

        unmatched_tracks = unmatched_confirmed_tracks + unmatched_unconfirmed_tracks
        return matched_pairs, unmatched_tracks, unmatched_detections

    @staticmethod
    def _cost_function(track, detection):
        # We use IOU to calculate cost.
        # You can use class_id as an additional input to calculate the cost or you can come up with your own cost function
        dx1, dy1, dx2, dy2 = detection[:4]
        tx1, ty1, tx2, ty2 = track.get_box()
        x1 = max(dx1, tx1)
        x2 = min(dx2, tx2)
        y1 = max(dy1, ty1)
        y2 = min(dy2, ty2)
        if x1 >= x2 or y1 >= y2:
            return 1
        s_i = (x2 - x1) * (y2 - y1)
        s_u = (dx2 - dx1) * (dy2 - dy1) + (tx2 - tx1) * (ty2 - ty1) - s_i
        iou = s_i / s_u
        return 1 - iou
