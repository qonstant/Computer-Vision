from IPython.display import HTML
from base64 import b64encode
import os

# yolo predict model=best.pt source=0 imgsz=640 task=detect mode=predict model=best.pt conf=0.25 source='ASL.mp4'

# Input video path
save_path = 'ASL.mp4'

# Compressed video path
compressed_path = "/Users/rakymzhan/Desktop/CV/SignLanguageAlphabetTranslator/result.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
