#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:23:13 2019

@author: azer
"""

import numpy as np


keys = ["a", "b", "c"]
vals = [[1, 2], 3, 4]
dd = dict(zip(keys, vals))

for i in zip(keys, vals):
    print(i)
    