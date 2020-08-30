import time
import pandas as pd
from numpy import *
import numpy as np
from collections import defaultdict
import  heapq
from math import ceil


#入节点，出节点，总网页
node_from=[]
node_to=[]
allpages=[]
#状态转移矩阵GM，矫正矩阵D，teleport,迭代矩阵A
GM_matrix=[]
D_matrix=[]
A_matrix=[]
teleport=0.85
max_times=100
min_error= 0.0000001
global time
# #初始pagerank和更新后rank
# pagerank=[]
# lastrank=[]
#出和入两个字典
link_out=defaultdict(list)
link_in=defaultdict(list)
node_dict=defaultdict(list)
out_value=defaultdict(list)
in_value=defaultdict(list)
f=open('WikiData.txt',encoding='UTF-8')
line=f.readline()
while line:
    line=line.replace("\n","")
    temp_node_from,temp_node_to=line.split('\t')
    node_from.append(int(temp_node_from))
    node_to.append(int(temp_node_to))
    link_out[int(temp_node_from)].append(int(temp_node_to))
    link_in[int(temp_node_to)].append(int(temp_node_from))
    if int(temp_node_from) not in allpages:
        allpages.append(int(temp_node_from))
        node_dict[int(temp_node_from)]=0
    if int(temp_node_to) not in allpages:
        allpages.append(int(temp_node_to))
        node_dict[int(temp_node_to)]=0
    line=f.readline()
f.close()

#去除重复元素
count_temp_from=list(set(node_from))
count_temp_to=list(set(node_to))
pagenum=len(allpages)
#统计信息
allpages.sort()
print("网页边数：",len(node_from))
print("网页节点数量：",len(allpages))
print("最小网页号：",min(allpages))
print("最大网页号：",max(allpages))
print("没有出度的网页数量",len(allpages)-len(count_temp_from))
print("没有入度的网页数量",len(allpages)-len(count_temp_to))
#统计入度和出度
for node,array in link_out.items():
    out_value[node]=len(array)
for node,array in link_in.items():
    in_value[node]=len(array)
print("最大出度：",max(out_value.values()))
print("最大入度：",max(in_value.values()))


GM_matrix = np.zeros((pagenum, pagenum))
D_matrix = np.ones((pagenum, pagenum)) * (1 / pagenum)
A_matrix = np.zeros((pagenum, pagenum))
# GM
print("开始初始化稀疏矩阵GM")
for i in node_from:
    temp_array = []
    length = 0
    temp_array = link_out[i]
    length = out_value[i]
    for j in range(0, length):
        node = temp_array[j]
        GM_matrix[allpages.index(node)][allpages.index(i)] = 1 / length
GM=pd.DataFrame(GM_matrix)
GM.to_csv("GM.csv")
print("GMsuccess")


def base_pagerank():
    base_pagerank = np.zeros(pagenum)
    base_lastrank = np.ones(pagenum) * (1 / pagenum)
    print("基础PageRank开始迭代")
    base_time=0
    # 设置最大迭代次数
    for index in range(0, max_times):
        base_time+=1
        # 每一次迭代，计算新的pagerank
        for i in range(0, pagenum):
            sum = 0
            if link_in.__contains__(allpages[i]):
                A=np.array(GM_matrix[i])
                B=np.array(base_lastrank)
                sum+=dot(A,B)
                # for j in range(0, pagenum):
                #     sum += GM_matrix[i][j] * base_lastrank[j]
                base_pagerank[i] = sum
            else:
                base_pagerank[i] = 0
        change = 0  # 判断是否满足条件
        for k in range(0, pagenum):
            change += abs(base_lastrank[k] - base_pagerank[k])

        if change < min_error:
            base_lastrank = base_pagerank
            break
        else:
            base_lastrank = base_pagerank
    print("基础PageRank迭代次数：",base_time)
    # #前100个最大的rank值
    front = heapq.nlargest(100, base_lastrank)
    np.savetxt("base_rank.txt", base_lastrank)
    np.savetxt("base_top.txt", front)
    base_pagerank = base_pagerank.tolist()
    base_lastrank = base_lastrank.tolist()
    front_index = map(base_lastrank.index, heapq.nlargest(100, base_lastrank))
    top_index = list(front_index)
    base_file=open("base_top.txt",'w')
    for i in range(0, 100):
        string=str(allpages[top_index[i]])+'\t'+str(front[i])+'\n'
        base_file.write(string)

    base_file.close()
    return

print("基础PageRank")
base_start = time.time()
print("基础PageRank开始时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
base_pagerank()
base_end = time.time()
print("基础PageRank结束时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("基础PageRank运行时间" + str(round(base_end - base_start, 2)) + "秒")






