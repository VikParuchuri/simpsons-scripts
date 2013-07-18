import sys
import os
from config import settings
directory = os.path.abspath(os.path.join(settings.AUDIO_BASE_PATH,".."))
seasons = os.listdir(directory)
for season in seasons:
    season_path = os.path.abspath(os.path.join(directory, season))
    files = os.listdir(season_path)
    video_extensions = [".avi",".mp4", ".mv4"]
    for filename in files:
        if filename[-4:] in video_extensions:
            filepath = os.path.abspath(os.path.join(season_path, filename))
            outfile = filename[:-4] + ".ogg"
            outfilepath = os.path.abspath(os.path.join(directory, "audio", outfile))
            cmd='ffmpeg -i "{0}" -vn -acodec libvorbis "{1}"'.format(filepath,outfilepath)
            os.system(cmd)