import cv2


class StreamLoader():
    '''
    Loads video streams
    '''

    def __init__(self, path, frame_number, video_segment, use_video_segment, read_all_frames, skip_freq) -> None:
        self.path = path
        self.frame_number = frame_number
        self.video_segment = video_segment
        self.use_video_segment = use_video_segment
        self.read_all_frame = read_all_frames
        self.skip_freq = skip_freq
        self.read_video()

    def read_video(self):
        
        try:
            self.vidcap = cv2.VideoCapture(self.path)
            self.width = self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
            self.total_frames = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

            if self.frame_number:
                self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number-1)

            if not self.vidcap.isOpened():
                raise Exception()
        except Exception:
            print("Could not load video")

    def run(self) -> tuple[list, int, float, float]:
        '''
        Load images from the video and return the images.
        Returns:
            images (list): list of images
            start_frame (int): Start frame number of each video segment
            fps (float): fps of video
        '''

        self.images = []
        self.start_frame = self.frame_number
        self.use_video_segment = True
        while self.read_all_frame:
            ret, frame = self.vidcap.read()
            if ret:
                if self.frame_number%self.skip_freq ==0:
                    self.images.append(frame)
                self.frame_number +=1
            else:
                print("All frames read complete")
                break
        
        while self.use_video_segment:
            ret, frame = self.vidcap.read()
            if ret:
                if self.frame_number%self.skip_freq ==0:
                    self.images.append(frame)
                self.frame_number +=1
                if not (self.frame_number % self.video_segment):
                    self.use_video_segment = False
            else:
                print("All frames read complete!!")
                break
        
        return self.images, self.start_frame, self.fps, self.total_frames

if __name__ == '__main__':

    video_stream = StreamLoader("naist_colonoscopy/processed_videos/2022-11-15 15-54-12.mp4",
                            0, 500, True, False)
    images, start_frame, fps, total_frames = video_stream.run()

    streaming = True
    while streaming:
        print('Length of images: ', len(images))
        print('Start frame number: ',start_frame)
        images, start_frame, fps, total_frames = video_stream.run()
        if not(len(images)):
            streaming = False