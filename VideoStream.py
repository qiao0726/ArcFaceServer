import ffmpeg

args = {"rtsp_transport": "tcp",
        "fflags": "nobuffer",
        "flags": "low_delay"}
class VideoStream:
    def __init__(self, rtsp_url:str) -> None:
        probe = ffmpeg.probe(rtsp_url)
        cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
        
        self.width = cap_info['width']           # 获取视频流的宽度
        self.height = cap_info['height']         # 获取视频流的高度
        self.process1 = (
            ffmpeg
            .input(rtsp_url, **args)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .overwrite_output()
            .run_async(pipe_stdout=True)
        )
    
    def getOneFrame(self):
        # Read one frame
        in_bytes = self.process1.stdout.read(self.width * self.height * 3)
        return in_bytes
    
    def shutdown(self):
        if self.process1 is not None:
            self.process1.kill()