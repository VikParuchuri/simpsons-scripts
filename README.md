Simpsons Scripts
====================

Overview
---------------------
This repository enables audio and text analysis of simpsons episodes.  The usage guide below is very rough.  The code is a mixture of R and Python.  Python is used for audio processing, and R is used for sentiment analysis and charting.

This is licensed under the APACHE license, please see LICENSE.txt for details.

Installation
---------------------
You will need to have python 2.7 installed, and ideally a virtualenv installed.

You will also need to follow the installation directions for [percept](http://percept.readthedocs.org/en/latest/installation/overview.html).

Then, you can do the following:

```
git clone git@github.com:VikParuchuri/simpsons-scripts
cd simpsons-scripts
sudo xargs -a apt-packages.txt apt-get install
Activate virtualenv if you are using one
pip install -r requirements.txt
```

Usage
---------------------


### Crawl Sites

You can crawl sites to get subtitles using the scrapy commands in the `crawler` folder.  See the [scrapy docs](http://doc.scrapy.org/en/latest/intro/tutorial.html) for more information.

```
cd simpsons-scripts/crawler
scrapy crawl snpp -t json -o ../data/raw_scripts.json
scrapy crawl sc -t json -o ../data/raw_scripts2.json
scrapy crawl ss -t json -o ../data/transcripts.json
```

These will grab the scripts and transcripts.

### NLP speaker labelling

We can then run our text processing pipeline:

```
cd simpsons-scripts
python manage.py run_flow config/script_save.conf --settings=config.settings --pythonpath=`pwd` -s
```

This will read in and do some processing on the scripts.  It will dump you into a console with the variable tasks.  `tasks[2].data.value` will give you information on the scripts.

We can then label the transcripts with:

```
python manage.py run_flow config/transcript_save.conf --settings=config.settings --pythonpath=`pwd` -s
```

`tasks[3].data.value` will give you more information on the labels.  You can read about percept and the tasks [here](http://percept.readthedocs.org/en/latest/index.html) and [here](http://vikparuchuri.com/blog/predicting-season-records-for-nfl-teams-part-2/).  Essentially, the tasks in `simpsons-scripts/tasks` are being run by `run_flow`, and any class variables that are defined are being saved.

### Audio speaker labelling

In order to process audio, you will first need to find subtitle files and store them in `data/subtitles`.  You will also need to find audio files (ideally in .ogg format) of the simpsons.  The `video_to_audio.py` script will strip audio from video files.  Make sure to set `AUDIO_BASE_PATH` in `config/settings.py`.  Put the audio in the `audio` folder under `AUDIO_BASE_PATH`.  The subtitle files and audio files are matched by season and episode, so make sure that they both have the episode and the string enclosed in square brackets in the filename, like `[4.04]`.

You will also need to label some of the lines in the subtitle files.

To label a .sub file, just add in brackets with the character name inside at the end of the line:

```
{2855}{2941}Uh, medley|of holiday "flavorites."{Skinner}
{2944}{3004}Dashing through the snow
```

To label a .srt file:

```
74
00:05:17,322 --> 00:05:22,055
Have a "D"-lightful summer!
[ Laughing ]{Ms.K}

75
00:05:23,895 --> 00:05:26,455
- Five!
- Four!
```

Once you have subtitles and video files, you can run the audio processing pipeline:

```
cd simpsons-scripts
python manage.py run_flow config/transcript_save.conf --settings=config.settings --pythonpath=`pwd` -s
```

This will take a while!  You can adjust `ONLY_LABELLED_LINES` and `PROCESSED_FILES_LIMIT` in `settings.py` to alter what is read in.  Looking at `tasks[0].data.value` and `tasks[0].res.value` will be useful.

Once you are done looking, do `tasks[0].data.value.to_csv('full_results.csv')`.

### Sentiment Analysis

Once you have a full_results.csv, you can fire up R and open `analyze_sentiment.R` to do the sentiment analysis and charting.

You can also use `generate_charts.R` if you dump out speaker,line dictionaries in json format.  Look at the file for more details.


How to Contribute
-----------------
Contributions are very welcome. The easiest way is to fork this repo, and then
make a pull request from your fork.

Please contact vik dot paruchuri at gmail with any questions or issues.
