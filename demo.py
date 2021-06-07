import os
import sys

import cv2
import torchvision
import yolov5

from tracker import SSort

DEMO_OUTPUT = 'demo_output'
SAVE_OUTPUT = False


def save_frame(frame):
    if (not os.path.exists(DEMO_OUTPUT)) or (not os.path.isdir(DEMO_OUTPUT)):
        os.makedirs(DEMO_OUTPUT)
    out_file = os.path.join(DEMO_OUTPUT, str(save_frame.frame_count) + '.jpg')
    cv2.imwrite(out_file, frame)
    save_frame.frame_count += 1


save_frame.frame_count = 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python demo.py <path_to_video>')
        exit(-1)

    video_path = sys.argv[1]
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print('Unable to open', video_path)
        exit(-1)

    model = yolov5.load('yolov5s.pt')

    # You must change these parameters to fit your problem
    tracker = SSort(init_step=3, min_hit=5, max_mis=12,
                    min_cost_threshold=0.66, max_cost_threshold=0.99)

    while True:
        ok, frame = video.read()
        if not ok or frame is None:
            break
        detections = model(frame)
        detections = detections.pred[0]
        boxes = detections[:, :4]
        scores = detections[:, 4]
        indices = torchvision.ops.nms(boxes, scores, 0.45)
        detections = detections[indices]
        detections = detections.tolist()

        tracker.forward()
        observed_tracks, deleted_tracks, tracks_id = tracker.update(detections)

        for tid, detection in zip(tracks_id, detections):
            x1, y1, x2, y2 = [int(v) for v in detection[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for track in tracker.tracks:
            x1, y1, x2, y2 = [int(v) for v in track.get_box()]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            if track.age >= tracker.init_step:
                cv2.putText(frame, str(track.id), (x1+2, y1+12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1, cv2.LINE_AA)

        if SAVE_OUTPUT:
            save_frame(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1000 // 30) == 27:
            break
