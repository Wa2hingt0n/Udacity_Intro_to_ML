# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:00:41 2022

@author: Adapted from Stackoverflow
"""

"""\
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py <input> <output>
"""

original = "my_feature_list.pkl"
destination = "my_feature_list_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
  content = infile.read()
with open(destination, 'wb') as output:
  for line in content.splitlines():
    outsize += len(line) + 1
    output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))