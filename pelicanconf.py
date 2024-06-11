#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

from datetime import datetime

AUTHOR = 'Anwaar Khalid'
SITENAME = 'Vive Discere'
SITETITLE = 'Vive Discere'
SITEURL=''
RELATIVE_URLS=True

USER_LOGO_URL = SITEURL + 'images/dp.jpg' 
ROUND_USER_LOGO = True

TAGLINE = "Machine Learning is cool"
DISQUS_SITENAME = True

BROWSER_COLOR='#333333'
PYGMENTS_STYLE='monokai'
THEME_COLOR = 'light'
THEME_COLOR_AUTO_DETECT_BROWSER_PREFERENCE = False
THEME_COLOR_ENABLE_USER_OVERRIDE = False

PATH = "content"
OUTPUT_PATH = 'public'

TIMEZONE = 'Asia/Kolkata'
COPYRIGHT_YEAR = datetime.now().year

THEME = './Theme/pelican-svbhack'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Social widget
SOCIAL = (
    ("linkedin", "https://www.linkedin.com/in/anwaar-khalid/"),
    ("github", "https://github.com/hello-fri-end"),
)

LINKS = [
        ('Categories', '/categories.html')
        ]

DEFAULT_PAGINATION = 10
