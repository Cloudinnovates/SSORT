from track import Track


class SSort:

    def __init__(self,
                 init_step=3,
                 max_mis=10,
                 min_iou=0.1,
                 padding_ratio=0.5):
        self.init_step = init_step
        self.max_mis = max_mis
        self.min_iou = min_iou
        self.padding_ratio = padding_ratio
        self.tracks = []
        self.next_track_id = 0

    def update(self, detections):
        '''Update tracker with new set of detections'''
        self._forward()
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
        # Create tracks in the next frame
        deleted_tracks = []
        next_step_tracks = []
        for track in self.tracks:
            if track.age >= self.init_step:  # confirmed tracks
                if track.n_consecutive_mis > self.max_mis:
                    deleted_tracks.append(track)
                else:
                    next_step_tracks.append(track)
            else:  # unconfirmed track
                if track.n_consecutive_mis > 0:
                    deleted_tracks.append(track)
                else:
                    next_step_tracks.append(track)
        self.tracks = next_step_tracks
        return obverved_tracks, deleted_tracks, tracks_id

    def _forward(self):
        '''Forward internal state of all track to the next frame'''
        for track in self.tracks:
            track.forward()

    def _matching(self, detections):
        '''Matching tracks with detections'''
        # Build all matching pair candidates
        n_track = len(self.tracks)
        n_detection = len(detections)
        matching_candidates = []
        for t in range(n_track):
            for d in range(n_detection):
                score = self._score_function(self.tracks[t], detections[d])
                if score > 0:
                    matching_candidates.append((t, d, score))
        matching_candidates = sorted(matching_candidates, key=lambda item: item[2], reverse=True)
        # Matching base on score
        matched_pairs = []
        matched_d = set()
        matched_t = set()
        for t, d, score in matching_candidates:
            if t in matched_t or d in matched_d:
                continue
            matched_t.add(t)
            matched_d.add(d)
            matched_pairs.append((t, d))
        # Create list of unmatched tracks and unmatched detection
        unmatched_tracks = [t for t in range(n_track) if t not in matched_t]
        unmatched_detections = [d for d in range(n_detection) if d not in matched_d]
        return matched_pairs, unmatched_tracks, unmatched_detections

    def _score_function(self, track, detection):
        '''Matching score function'''
        # We want to match detections with confirmed tracks first
        if track.age < self.init_step:
            confirm_score = 1000000
        else:
            confirm_score = 1000
        # We penalise track that do not match with any detection recently
        mis_detection_penalty = -track.n_consecutive_mis * 10
        # Now we calculate IOU
        dx1, dy1, dx2, dy2 = detection[:4]
        dw = dx2 - dx1
        dh = dy2 - dy1
        # Enlarge box
        dx1 -= dw * self.padding_ratio
        dx2 += dw * self.padding_ratio
        dy1 -= dh * self.padding_ratio
        dy2 += dh * self.padding_ratio
        tx1, ty1, tx2, ty2 = track.get_box()
        tw = tx2 - tx1
        th = ty2 - ty1
        # Enlarge box
        tx1 -= tw * self.padding_ratio
        tx2 += tw * self.padding_ratio
        ty1 -= th * self.padding_ratio
        ty2 += th * self.padding_ratio
        # Overlap area
        x1 = max(dx1, tx1)
        x2 = min(dx2, tx2)
        y1 = max(dy1, ty1)
        y2 = min(dy2, ty2)
        if x1 >= x2 or y1 >= y2:
            iou = 0
        else:
            s_i = (x2 - x1) * (y2 - y1)
            s_u = (dx2 - dx1) * (dy2 - dy1) + (tx2 - tx1) * (ty2 - ty1) - s_i
            iou = s_i / s_u
        if iou < self.min_iou:
            return -1
        else:
            return confirm_score + mis_detection_penalty + iou
