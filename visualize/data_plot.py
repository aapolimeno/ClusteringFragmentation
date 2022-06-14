# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:36:02 2022

@author: Alessandra
"""

#import pandas as pd

#dev = pd.read_csv("../../data/new_split/new_dev.csv", index_col = 0)
#eva = pd.read_csv("../../data/new_split/new_test.csv", index_col = 0)



import plotly.graph_objects as go


labels = ["Human Cloning", "International Space Station", "Ireland Abortion Vote", "US Bird Flu Outbreak", "Facebook Privacy Scandal", "Wikileaks Trials", "Tunesia Protests", "Ivory Coast Army Mutiny", "Equifax Breach", "Brazil Dam Disaster"]

values = [108, 215, 170, 75, 172, 153, 86, 104, 156, 24]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20)
fig.show()

