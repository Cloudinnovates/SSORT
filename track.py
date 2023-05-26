class Track:

    def __init__(self, id, detection):
        '''Initialize new track with a detection. A detection is a tuple (x1, y1, x2, y2, score, class)'''
        self. id = id
        self.first_detection = detection
        self.last_detection = detection
        self.n_hit = 1
        self.n_consecutive_mis = 0
        self.age = 1
        self.last_detection_age = self.age
        self.class_id_count = {detection[5]: detection[4]}
        self.x = (detection[0] + detection[2]) / 2  # center point
        self.y = (detection[1] + detection[3]) / 2  # center point
        self.w = detection[2] - detection[0]
        self.h = detection[3] - detection[1]
        self.vx = None  # x velocity
        self.vy = None  # y velocity

    def forward(self):
        '''Estimate track state in the next frame'''
        self.age += 1
        self.n_consecutive_mis += 1
        if self.vx is not None and self.vy is not None:
            self.x += self.vx
            self.y += self.vy

    def update(self, detection):
        '''Update internal state with a new detection'''
        self.n_consecutive_mis = 0
        self.n_hit += 1

        x = (detection[0] + detection[2]) / 2  # center point
        y = (detection[1] + detection[3]) / 2  # center point
        w = detection[2] - detection[0]
        h = detection[3] - detection[1]

        # You may want to change all these weights to fit your problem
        self.x = 0.9 * x + 0.1 * self.x
        self.y = 0.9 * y + 0.1 * self.y
        self.w = 0.5 * w + 0.5 * self.w
        self.h = 0.5 * h + 0.5 * self.h

        lx = (self.last_detection[0] +
              self.last_detection[2]) / 2  # center point
        ly = (self.last_detection[1] +
              self.last_detection[3]) / 2  # center point
        vx = (x - lx) / (self.age - self.last_detection_age)
        vy = (y - ly) / (self.age - self.last_detection_age)
        if self.vx is not None and self.vy is not None:
            self.vx = 0.5 * vx + 0.5 * self.vx
            self.vy = 0.5 * vy + 0.5 * self.vy
        else:
            self.vx = vx
            self.vy = vy

        self.last_detection = detection
        self.last_detection_age = self.age

    def get_box(self):
        '''Get internal state box'''
        x1 = self.x - self.w / 2
        y1 = self.y - self.h / 2
        x2 = self.x + self.w / 2
        y2 = self.y + self.h / 2
        return [x1, y1, x2, y2]

    def get_dominated_class(self):
        '''Get the most dominated class id'''
        ret_id = -1
        max_count = -1
        for class_id, count in self.class_id_count.items():
            if count > max_count:
                max_count = count
                ret_id = class_id
        return ret_id
