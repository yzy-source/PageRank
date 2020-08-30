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

#一次转移迭代
def transform(link_out,link_in,node_dict,teleport):
    new_rank={}
    for row in node_dict.keys():
        t=0
        if link_in.__contains__(row):
            for fromnode in link_in[row]:
                t+=1/len(link_out[fromnode])*node_dict[fromnode]
            new_rank[row]=t*teleport
        else:
            new_rank[row]=0
        new_rank[row]+=(1-teleport)*(1/pagenum)
    return new_rank
#计算是否满足迭代终止条件
def compute_error(init_rank,last_rank,allow_error):
    error=0
    for row in node_dict.keys():
        error+=abs(init_rank[row]-last_rank[row])
    if error<allow_error:
        return True
    else:
        return False
#开始迭代
def Opti_pagerank(link_out,link_in,node_dict,teleport):
    opti_time=0
    #初始化pagerank,初始rank值为1/pagenum
    for key in node_dict.keys():
        node_dict[key]=1/pagenum
    print(node_dict)
    for i in range(0,max_times):
        opti_time+=1
        opti_lastrank=transform(link_out,link_in,node_dict,teleport)
        if compute_error(node_dict,opti_lastrank,min_error):
            node_dict=opti_lastrank
            break
        else:
            node_dict=opti_lastrank
    print("优化稀疏矩阵迭代次数：",opti_time)
    return opti_lastrank

#优化稀疏矩阵
print("优化稀疏矩阵的PageRank")
opti_start=time.time()
print("优化稀疏矩阵开始时间："+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#输出前100
opti_output=Opti_pagerank(link_out,link_in,node_dict,teleport)
sorted_pagerank=sorted(opti_output.items(),key=lambda x:x[1],reverse=True)
opti_file=open('opti_top.txt','w')
for i in range(0,100):
    string=str(sorted_pagerank[i][0])+'\t'+str(sorted_pagerank[i][1])+'\n'
    opti_file.write(string)
opti_file.close()
opti_end=time.time()
print("优化稀疏矩阵结束时间："+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("优化稀疏矩阵运行时间"+str(round(opti_end-opti_start,2))+"秒")



#分块条件下每一次迭代
def block_transform(link_out,link_in,sub_block,teleport):
    new_rank={}
    for row in sub_block.keys():
        t=0
        if link_in.__contains__(row):
            for fromnode in link_in[row]:
                t+=1/len(link_out[fromnode])*node_dict[fromnode]
            new_rank[row]=t*teleport
        else:
            new_rank[row]=0
        new_rank[row]+=(1-teleport)*(1/pagenum)
    return new_rank

#计算子块误差
def block_compute_error(init_rank,last_rank):
    error=0
    for row in init_rank.keys():
        error+=abs(init_rank[row]-last_rank[row])
        return error

def block():
    blocksize=1000
    loop=ceil(pagenum/blocksize)
    print("共"+str(loop)+"块")
    for key in node_dict.keys():
        node_dict[key] = 1 / pagenum
    final_dict= defaultdict(list)
    block_it_time=0
    read_time=0
    for it_time in (0, max_times):
        block_it_time+=1
        it_error=0
        for i in range(0, loop):
            # 每一次加载
            read_time+=0.25
            sub_block = defaultdict(list)
            if i != loop - 1:
                print("第"+str(i)+"个block")
                for j in range(i * blocksize, (i + 1) * blocksize):
                    sub_block[int(allpages[j])] = node_dict[int(allpages[j])]
                opti_lastrank = block_transform(link_out, link_in, sub_block, teleport)
                it_error+=block_compute_error(sub_block,opti_lastrank)
                final_dict.update(opti_lastrank)
            if i == loop - 1:
                print("第" + str(i) + "个block")
                for k in range(i * blocksize, pagenum):
                    sub_block[int(allpages[k])] = node_dict[int(allpages[j])]
                opti_lastrank = block_transform(link_out, link_in, sub_block, teleport)
                it_error += block_compute_error(sub_block, opti_lastrank)
                final_dict.update(opti_lastrank)
        if it_error<min_error:
            break
    print("分块迭代"+str(block_it_time)+"次")
    return final_dict,read_time

print("分块开始运行")
block_start=time.time()
print("分块开始时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
block_output,read=block()
block_sorted_pagerank=sorted(block_output.items(),key=lambda x:x[1],reverse=True)
block_file=open('block_top.txt','w')
for i in range(0,100):
    string=str(block_sorted_pagerank[i][0])+'\t'+str(block_sorted_pagerank[i][1])+'\n'
    block_file.write(string)
block_file.close()
block_end=time.time()
print("分块结束时间："+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("分块运行时间"+str(round(block_end-block_start+read,2))+"秒")



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

#初始化pagerank
def include_pagerank():
    pagerank = np.zeros(pagenum)
    lastrank = np.ones(pagenum) * (1 / pagenum)
    # # 计算A矩阵
    # for i in range(0, pagenum):
    #     for j in range(0, pagenum):
    #         A_matrix[i][j] = GM_matrix[i][j] * teleport + D_matrix[i][j] * (1 - teleport)
    # print("计算转移矩阵结束，开始迭代")
    include_time=0
    # 设置最大迭代次数
    for time in range(0, max_times):
        include_time+=1
        # 每一次迭代，计算新的pagerank
        for i in range(0, pagenum):
            sum = 0
            if link_in.__contains__(allpages[i]):
                for j in range(0, pagenum):
                    sum += GM_matrix[i][j] * lastrank[j]
                pagerank[i] = sum * teleport
            else:
                pagerank[i]=0
            pagerank[i]+=(1-teleport)*(1/pagenum)

        change = 0  # 判断是否满足条件
        for k in range(0, pagenum):
            change += abs(lastrank[k] - pagerank[k])

        if change < min_error:
            lastrank = pagerank
            break
        else:
            lastrank = pagerank

    print("考虑Dead Ends和Spider Traps迭代次数：",include_time)
    # #前100个最大的rank值
    front = heapq.nlargest(100, lastrank)
    np.savetxt("include_rank.txt", lastrank)
    np.savetxt("include_top.txt", front)
    pagerank = pagerank.tolist()
    lastrank = lastrank.tolist()
    front_index = map(lastrank.index, heapq.nlargest(100, lastrank))
    top_index = list(front_index)
    include_file=open("include_top.txt",'w')
    for i in range(0, 100):
        string=str(allpages[top_index[i]])+'\t'+str(front[i])+'\n'
        include_file.write(string)

    include_file.close()
    return

print("考虑Dead Ends and Spider trap")
pagerank_start =time.time()
print("开始时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
include_pagerank()
pagerank_end = time.time()
print("考虑Dead Ends和Spider Traps结束时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("考虑Dead Ends和Spider Traps运行时间" + str(round(pagerank_end - pagerank_start, 2)) + "秒")

