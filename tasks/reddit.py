from __future__ import division
from percept.tasks.base import Task
from percept.tasks.train import Train
from percept.fields.base import Complex, List, Dict, Float
from percept.utils.models import RegistryCategories, get_namespace
import logging
from inputs.inputs import SimpsonsFormats
import numpy as np
import calendar
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math
import random
from itertools import chain
from percept.tests.framework import Tester
import os
from percept.conf.base import settings
import praw
import pickle

log = logging.getLogger(__name__)

def read_raw_data_from_cache(filename):
    try:
        raw_data_cache = pickle.load(open(filename, "r"))
    except Exception:
        raw_data_cache = []
    return raw_data_cache

def write_data_to_cache(raw_data, filename, unique_key="message"):
    raw_data_cache = read_raw_data_from_cache(filename)
    raw_data_messages = [r[unique_key] for r in raw_data_cache]
    for r in raw_data:
        if r[unique_key] in raw_data_messages:
            del_index = raw_data_messages.index(r[unique_key])
            del raw_data_cache[del_index]
            del raw_data_messages[del_index]
    raw_data_to_write = [r for r in raw_data if r not in raw_data_cache]
    raw_data_cache += raw_data_to_write

    with open(filename, "w") as openfile:
        pickle.dump(raw_data_cache, openfile)
    return raw_data_cache


def get_single_comment(subreddit_name):
    comment_found = False
    index = 0
    random_comment = None
    while not comment_found:
        r = praw.Reddit(user_agent=settings.BOT_USER_AGENT)
        subreddit = r.get_subreddit(subreddit_name)
        hourly_top = list(subreddit.get_top_from_hour(limit=(index+1)))
        comments = [c for c in hourly_top[index].comments if isinstance(c, praw.objects.Comment)]
        index += 1
        if len(comments)>2:
            rand_int = random.randint(0,len(comments))
            random_comment = comments[rand_int]
            comment_found = True
        if index>10:
            return None
    return random_comment

class PullDownComments(Task):
    data = Complex()

    data_format = SimpsonsFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Pull down comments and store them."

    def train(self, data, **kwargs):
        try:
            items_done = read_raw_data_from_cache(os.path.abspath(os.path.join(settings.DATA_PATH, "items_done.p")))
            comments = [c['comment'] for c in items_done]
            replies = [c['reply'] for c in items_done]
            for subreddit in settings.REPLY_SUBREDDIT_LIST:
                try:
                    comment = get_single_comment(subreddit)
                    print comment
                    if comment is None:
                        log.info("Could not get a comment")
                        continue
                    text = comment.body
                    cid = comment.id
                    reply = test_knn_matcher(knn_matcher, text)
                    if text in comments or (reply in replies and reply is not None):
                        continue
                    data = {'comment' : text, 'reply' : reply, 'comment_id' : cid}
                    items_done.append(data)
                    replies.append(reply)
                    comments.append(text)
                    log.info("Subreddit: {0}".format(subreddit))
                    log.info("Comment: {0} {1}".format(cid, text))
                    log.info("Reply: {0}".format(reply))
                    log.info("-------------------")
                except:
                    log.exception("Cannot get reply for {0}".format(subreddit))
                    continue
                write_data_to_cache(items_done, "items_done.p", "comment_id")
        except Exception:
        log.exception("Could not pull down comment.")
