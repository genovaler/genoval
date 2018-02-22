import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objs as go
import plotly.offline as py
import plotly.plotly as pp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test2 = pd.Series([1, 2, 2, 3, 4, 7, 9])
print(test2.describe())

# labels = 'A', 'B', 'C', 'D'
fracs = [15, 30.55, 44.44, 10, 10]
# explode = [0, 0.1, 0, 0]  # 0.1 凸出这部分，
# plt.axes(aspect=1)  # set this , Figure is round, otherwise it is an ellipse
# # autopct ，show percet
# plt.pie(x=fracs, labels=labels, explode=explode, autopct='%3.1f %%',
#         shadow=True, labeldistance=1.1, startangle=90, pctdistance=0.6)
# # plt.show()
#
# print(i for i in [1,2,3,4,5])

# values = [234, 64, 54, 10, 0, 1, 0, 9, 2, 1, 7, 7]
# months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
#           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#
# colors = ['yellowgreen', 'red', 'gold', 'lightskyblue',
#           'white', 'lightcoral', 'blue', 'pink', 'darkgreen',
#           'yellow', 'grey', 'violet', 'magenta', 'cyan']
#
# # plt.pie(values, labels=months, autopct='%1.1f%%', shadow=True,
#         colors=colors, startangle=90, radius=1.2)

# plt.show()

x = np.char.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct', 'Nov','Dec'])
y = np.array([234, 64, 54,10, 0, 1, 0, 9, 2, 1, 7, 7])
colors = ['yellowgreen','red','gold','lightskyblue','white','lightcoral','blue','pink', 'darkgreen','yellow','grey','violet','magenta','cyan']
porcent = 100.*y/y.sum()

patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y), key=lambda x: x[2], reverse=True))

plt.legend(patches, labels, bbox_to_anchor=(-0.1, 1.),fontsize=8)

plt.savefig('piechart.png', bbox_inches='tight')
plt.close()


# fake up some data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 50
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# data = np.concatenate((spread, center, flier_high, flier_low), 0)
#
# # basic plot
# plt.boxplot(data)
#
# # notched plot
# # plt.figure()
# plt.boxplot(data, 1)
#
# # change outlier point symbols
# # plt.figure()
# plt.boxplot(data, 0, 'gD')
#
# # don't show outlier points
# # plt.figure()
# plt.boxplot(data, 0, '')
#
# # horizontal boxes
# # plt.figure()
# plt.boxplot(data, 0, 'rs', 0)
#
# # change whisker length
# # plt.figure()
# plt.boxplot(data, 0, 'rs', 0, 0.75)
#
#
#
# # fake up some more data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 40
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
# data.shape = (-1, 1)
# d2.shape = (-1, 1)
# # data = concatenate( (data, d2), 1 )
# # Making a 2-D array only works if all the columns are the
# # same length.  If they are not, then use a list instead.
# # This is actually more efficient because boxplot converts
# # a 2-D array into a list of vectors internally anyway.
# data = [data, d2, d2[::2, 0]]
# multiple box plots on one figure
# plt.figure()
# plt.boxplot(data)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(2)  #设置随机种子
df = pd.DataFrame(np.random.rand(5,4),columns=['A', 'B', 'C', 'D']) #先生成0-1之间的5*4维度数据，再装入4列DataFrame中
print(df)
df.boxplot() #也可用plot.box()
plt.show()

trace = go.Bar(x=[2, 4, 6], y= [10, 12, 15])
data = [trace]
layout = go.Layout(title='A Simple Plot', width=800, height=640)
fig = go.Figure(data=data, layout=layout)

py.image.save_as(fig, filename='a-simple-plot.png')

from IPython.display import Image
Image('a-simple-plot.png')