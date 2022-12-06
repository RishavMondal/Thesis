path = '__annotator.json'

from pandas.core.common import flatten
import pandas as pd
import copy
import numpy as np
import random
import os
import cv2

import glob
from tqdm import tqdm
from pathlib import Path
import glob
import json
import statistics as st
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay as cd
import matplotlib.pyplot as plt
import seaborn as sn



"""
620 / 5,000
Translation results
"Page border"= 0
   Erasure = 1
   Burr = 2
   "Printed Text" = 3
   "Manuscript Text" = 4
   "Pause (full or almost)" = 5
   "Single note (with at least the head)" = 6
   "Multiple Notes (with at least the head)" = 7
   "Single chord (with at least heads)" = 8
   "Multiple chords (with at least heads)" = 9
   "Accidental(s) (whole or nearly so)" = 10
   "Key(s) (whole(s) or nearly)" = 11
   "Embellishment(s) (whole(s) or nearly)" = 12
   "More categories (with at least one musical score)" = 14
   "More categories (no musical scores)" = 15
   "Other (with at least one musical score)" = 16
   "Other (without musical markings)" = 17
"""


f = open(path)
Json_Data = json.load(f)
print(Json_Data)
print(len(Json_Data))


for i in Json_Data:
    print(i)

    
# Convert Key-Value list Dictionary to Lists of List

res = []
for key, val in Json_Data.items():
    res.append([key] + val)

tenca1 = res[0][1:]
tenca2 = res[1][1:]
majorana_setti1 = res[2][1:]
alessandrini_titolivio1 = res[3][1:]
mariecurie_cernusco1 = res[4][1:]
tenca3 = res[5][1:]
bachelet_abbiategrasso = res[6][1:]

"""
Now i need to find the mode for each list
"""

def Mode(list):
    
    mode_list = []
    for i in list:
        j = st.mode(i)
        mode_list.append(j)
    
    return mode_list


Tenca1 = Mode(tenca1)
Tenca2 = Mode(tenca2)
Majorana_Setti1 = Mode(majorana_setti1)
Alessandrini_Titolivio1 = Mode(alessandrini_titolivio1)
Mariecurie_Cernusco1 = Mode(mariecurie_cernusco1)
Tenca3 = Mode(tenca3)
Bachelet_Abbiategrasso = Mode(bachelet_abbiategrasso)

set = [Tenca1, Tenca2, Majorana_Setti1, Alessandrini_Titolivio1, Mariecurie_Cernusco1, Tenca3, Bachelet_Abbiategrasso]
truth = []

for i in range(0,500):
    track = []
    for j in set:
        track.append(j[i])
    
    truth.append(st.mode(track))
    

classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

num_classes = len(classes)

C_M = pd.DataFrame(np.zeros((num_classes,num_classes), dtype = int), 
                   index = classes, columns = classes)

for i in set:
    for t_label,p_label in zip(truth,i):
        
        C_M.loc[t_label, p_label] += 1
        

print(C_M)


fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()
sn.heatmap(C_M, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(classes, fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(classes, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Refined Confusion Matrix', fontsize=20)

#plt.savefig('ConMat24.png')
plt.show()












