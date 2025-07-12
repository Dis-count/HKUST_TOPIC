# bounded knapsack problem

# unfinished work

def binaryDecomposition(n):  # binaryDecomposition to
     k = 0
     res = []
     while n - 2**(k+1) + 1  > 0:
         res.append(2**k)
         k += 1
     res.append(n - 2 ** (k) + 1)
     return res

# for example  10-> [1,2,4,3] 17-> [1,2,4,8,2]
def trans(weightList, valueList, numList):
    newWeightList = []
    newValueList = []
    for i in range(len(numList)):
        for j in binaryDecomposition(numList[i]):
            newWeightList.append(j * weightList[i])
            newValueList.append(j * valueList[i])
    num = len(newValueList)
    return num,newValueList,newWeightList

num,newValueList,newWeightList = trans(weightList, valueList, demandList)

def zeroOneBag(numList,capacity,weightList,valueList):
    num,newValueList,newWeightList = trans(weightList,valueList,numList)
    valueExcel= [[0 for j in range(capacity + 1)] for i in range(num + 1)]
    for i in range(1,num+1):
        for j in range(1,capacity+1):
            valueExcel[i][j] = valueExcel[i - 1][j]
            if j >= newWeightList[i-1] and valueExcel[i][j] < (valueExcel[i - 1][j - newWeightList[i - 1]] + newValueList[i - 1]):
                valueExcel[i][j] = (valueExcel[i - 1][j - newWeightList[i - 1]] + newValueList[i - 1])
    return valueExcel

capacity = 21*10
demandList = [10, 17, 20, 15]
weightList = [2, 4, 6, 7]
valueList = [i-1 for i in weightList]

valueExcel = zeroOneBag(demandList,capacity,weightList,valueList)

print(f'Total_Value: {valueExcel[-1][-1]}')

def showRes(num,capacity,weightList,valueExcel):
    indexRes = []
    j = capacity
    for i in range(num,0,-1):
        if valueExcel[i][j] != valueExcel[i-1][j]:
            indexRes.append(i)
            j -= weightList[i-1]
    return indexRes

#  Selected items in binary form.
indexNum = showRes(18, capacity, newWeightList, valueExcel)

#  Return the original items and the corresponding number
def transInv(numList, indexNum):
    newList = []
    num_i = [0]*(len(numList)+1)
    for i in range(len(numList)):
        for j in binaryDecomposition(numList[i]):
            newList.append(j)
        num_i[i+1] = len(newList)
    sum_list = [0]*len(numList)

    for j in range(len(numList)-1,-1,-1):
        for i in indexNum:
            if  num_i[j] < i <= num_i[j+1]:
                sum_list[j] += newList[i-1]
# The Complexity need to be revised
    return sum_list

sum_list = transInv(demandList,indexNum)

print(f'The numbers of selected groups are: {sum_list}')
