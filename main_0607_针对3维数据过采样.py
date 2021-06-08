import pandas as pd , re , numpy as np
import datetime
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing
import random
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve


#当数据量过大无法一次读出时调用该方法
def read_bigata(address):
    data = pd.read_table(address,chunksize=119421,sep=',')    #1000000
    return data
#正常读取数据
def read_data(address):
    data = pd.read_csv(address)
    return data

#判断字符串是否为数字
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

#合并数据，1秒内的数据合成一个
def Com_Data(chunk,com_all_data,std_time,temp_1s_data):
    temp_index = 0   #用于标记“temp_1s_data”数组添加新数据位置的
    total_index = 0    #标记“com_all_data”的数量，用来更改最新添加数据的第一个值为日期
    average_1 = np.zeros(shape=(0,111))
    i = 0
    while i < len(chunk):
        # print(i)
        for j in range(1,len(chunk[i])):   #把空值替换为0
            if chunk[i][j] == ' ' or chunk[i][j] == np.nan :
                chunk[i][j] = 0
            elif is_number(chunk[i][j]):
                chunk[i][j] = float(chunk[i][j])
            else:
                chunk[i][j] = 0
        chunk[i][0] = pd.to_datetime(chunk[i][0])
        # print("qiepian:",chunk[i][0],"  stdtime:",std_time)
        if (chunk[i][0]-std_time).seconds < 1:   #不到1s直接添加
            # print("if")
            temp_1s_data = np.append(temp_1s_data,[chunk[i]],axis=0)
            temp_1s_data[temp_index][0] = 0
            temp_index += 1
        else:  #执行求均值，对标准时间加1s的操作
            if temp_1s_data.all():   #如果数组为空数组，表明当前时间范围内在数据表中没有对应的记录
                # print("else if")
                std_time = std_time + datetime.timedelta(seconds=1)
                continue
            # print("elseelse")
            average_1 = np.mean(temp_1s_data,axis=0)    #1s内数据的均值
            # print(average_1)
            temp_1s_data = np.empty(shape=(0,111))    #清空临时变量中的所有元素
            com_all_data.append(average_1)     # np.append(com_all_data,average_1)
            com_all_data[total_index][0] = std_time
            total_index += 1
            # print("融合后的数据：",com_all_data)
            # print("融合后的数据：", std_time)
            std_time = std_time+datetime.timedelta(seconds=1)
            temp_index = 0
            i -= 1
        i += 1
    print("融合后的数据：", com_all_data[4000])
    # print("上一个时间节点：", last_time)
    # print("1s内的临时数据数量：", len(temp_1s_data))
    return com_all_data,std_time,temp_1s_data

#删除自添加的数据(删除的效率太低——已弃用)
def Delete_Data(com_all_data):
    num = 0
    data_len = len(com_all_data)
    for i in range(data_len):
        if (com_all_data[i] == 0).all():    #第i行所有数据均为0
            com_all_date = np.delete(com_all_data,i,0)
            i -= 1
            num += 1
    print("自添加数据的数量：",num)
    return com_all_data

#关联”融合之后的原始数据“和“报错数据”，对所有数据打标签
def Associated_Data_Old(com_all_data_len,error_data,y1):
    # print("融合后所有数据数量：",com_all_data_len)
    head_time = pd.to_datetime('2020-04-01 00:00:00.000000')
    j = 0  # 报错数据的index
    index = 0   #统计“报错”数量
    for i in range(com_all_data_len):
        error_level = 0  # 错误等级0
        if j==len(error_data):    #如果所有报错数据的标签都添加完毕了，那么后续的标签自动标记为无错误
            for n in range(i,com_all_data_len):
                y1.append(error_level)
            break
        tail_time = head_time + datetime.timedelta(seconds=1)
        error_data[j][0] = pd.to_datetime(error_data[j][0])
        if error_data[j][0]>=head_time and error_data[j][0]<=tail_time :   #第二个条件使用“<”会报错？？？？
            index += 1
            # print("错误时间:",error_data[j][0])
            if error_data[j][5] == "Low":   #!!!!!!!!!!!!!判断条件以后根据情况作出调整，是统一报警，还是根据每个舱的情况单独报警
                error_level = 1
            elif error_data[j][5] == "Medium":
                error_level = 2
            else:
                error_level = 3
            j += 1
        y1.append(error_level)  # np.append(y1,error_level)
        head_time = tail_time
    # print("报错数量：",index)

    return y1

#关联”融合之后的原始数据“和“报错数据”，对所有数据打标签
def Associated_Data(com_all_data,error_data,y1):
    re_com_all_data = []     #由于对应方法的限制，会导致 数据长度<=标签长度 所以需要重构一个com_all_data数组
    i = 0   #遍历com_all_data的标签
    j = 0  # 遍历报错数据的标签
    index = 0   #统计“报错”数量
    # for i in range(len(com_all_data)):
    while i < len(com_all_data):
        error_level = 0  # 错误等级0
        if j==len(error_data):    #如果所有报错数据的标签都添加完毕了，那么后续的标签自动标记为无错误
            for n in range(i,len(com_all_data)):
                y1.append(error_level)
                re_com_all_data.append(com_all_data[n][1:111])
            break
        if i==len(com_all_data)-1:
            if error_data[j][5] == "Low":   #!!!!!!!!!!!!!判断条件以后根据情况作出调整，是统一报警，还是根据每个舱的情况单独报警
                error_level = 1
            elif error_data[j][5] == "Medium":
                error_level = 2
            else:
                error_level = 3
            y1.append(error_level)
            re_com_all_data.append(com_all_data[i][1:111])
            break
        com_all_data[i][0] = pd.to_datetime(com_all_data[i][0])
        com_all_data[i+1][0] = pd.to_datetime(com_all_data[i+1][0])
        error_data[j][0] = pd.to_datetime(error_data[j][0])
        # print(error_data[j][0],"    ",com_all_data[i][0],"    ",(error_data[j][0]-com_all_data[i][0]).seconds,"    ",(com_all_data[i+1][0]-com_all_data[i][0]).seconds)
        if error_data[j][0]<com_all_data[i][0]:
            i -= 1
            continue
            #指定在com_all_data某一行之后插入一条数据，如果做不到就放弃该条错误数据
        elif (error_data[j][0]-com_all_data[i][0]).seconds < (com_all_data[i+1][0]-com_all_data[i][0]).seconds :   #当数据时间离报错时间更近
            index += 1
            # print("错误时间:",error_data[j][0])
            if error_data[j][5] == "Low":   #!!!!!!!!!!!!!判断条件以后根据情况作出调整，是统一报警，还是根据每个舱的情况单独报警
                error_level = 1
            elif error_data[j][5] == "Medium":
                error_level = 2
            else:
                error_level = 3
            j += 1
        y1.append(error_level)  # np.append(y1,error_level)
        re_com_all_data.append(com_all_data[i][1:111])
        i += 1
    print("报错数量：",index)
    return re_com_all_data,y1

#重新划分数据集
def Re_Split(com_all_data, y1):
    scaler = StandardScaler()  # 数据标准化
    scaler.fit(com_all_data)  # 训练标准化对象
    com_all_data = scaler.transform(com_all_data)  # 转换数据集
    com_all_data = PCA_Denoise(com_all_data)
    feature_train, feature_test, target_train, target_test = train_test_split(com_all_data, y1, test_size=0.3, random_state=0)
    # print("数据长度：", len(feature_train), "    ", len(feature_train[0]), "    ", feature_train[0])
    return feature_train, feature_test, target_train, target_test

#简单过采样(简单过采样，只复制低频标签，提升数量)
def Simple_Over_Sample(com_all_data,y1,com_all_data_sample,y1_sample,sample_num):
    for i in range(len(y1)):
        if y1[i] == 1:
            for j in range(6):  #每个标签复制N1遍
                com_all_data_sample.append(com_all_data[i])
                y1_sample.append(y1[i])
        elif y1[i] == 2:
            for j in range(28):  #每个标签复制N2遍
                com_all_data_sample.append(com_all_data[i])
                y1_sample.append(y1[i])
        elif y1[i] == 3:
            for j in range(100):  #每个标签复制N2遍
                com_all_data_sample.append(com_all_data[i])
                y1_sample.append(y1[i])
    return com_all_data_sample,y1_sample

#KNN
def Knn_Check(x,y,z):#KNN
    clf = neighbors.KNeighborsClassifier()
    clf.fit(com_all_data, y1)
    temp = np.vstack((x,y,z)).reshape((1,-1))
    # print(temp.shape)
    # print("knn_check的结果：",clf.predict(temp))
    return clf.predict(temp)
#实际过采样操作
def Over_Sample_(com_all_data,y1,com_all_data_sample,y1_sample,label,num_low_medium_high):
    randomNumber = random.randint(0, len(com_all_data) - 1)
    if y1[randomNumber] == label:
        num4 = 0
        while num4 < 1:
            randomNumber2 = random.randint(0, len(com_all_data) - 1)
            if y1[randomNumber2] == label and y1[randomNumber2] != y1[randomNumber]:
                randfloat = random.random()
                xx1 = (com_all_data[randomNumber, 0] - com_all_data[randomNumber2, 0]) * randfloat + com_all_data[randomNumber2, 0]
                yy1 = (com_all_data[randomNumber, 1] - com_all_data[randomNumber2, 1]) * randfloat + com_all_data[randomNumber2, 1]
                zz1 = (com_all_data[randomNumber, 2] - com_all_data[randomNumber2, 2]) * randfloat + com_all_data[randomNumber2, 2]
                # knn检查生成点
                clu = Knn_Check(xx1, yy1, zz1)
                if clu == label:
                    num4 += 1
                    num_low_medium_high += 1
                    temp = np.vstack((xx1, yy1, zz1)).reshape((1, -1))
                    com_all_data_sample.append(temp)
                    y1_sample.append(clu)
    return com_all_data_sample,y1_sample,num_low_medium_high
#过采样 数据降维至3维时使用
def Three_Over_Sample(com_all_data,y1,com_all_data_sample,y1_sample,sample_num):
    num_low = 0   #已生成的低报警数量
    num_medium = 0  # 已生成的中等报警数量
    num_high = 0  # 已生成的高报警数量
    while num_low < (sample_num-1554) :
        com_all_data_sample,y1_sample,num_low = Over_Sample_(com_all_data,y1,com_all_data_sample,y1_sample,1,num_low)
    while num_medium < (sample_num - 322):
        com_all_data_sample,y1_sample, num_medium = Over_Sample_(com_all_data, y1,com_all_data_sample,y1_sample, 2, num_medium)
    return com_all_data_sample,y1_sample

#欠采样
def Under_Sample(com_all_data,y1,com_all_data_sample,y1_sample,sample_num):
    i = 0
    while i<sample_num:
        random_num = np.random.randint(0,len(y1))
        if y1[random_num] == 0:
            com_all_data_sample.append(com_all_data[random_num])
            y1_sample.append(y1[random_num])
            i += 1
    return com_all_data_sample,y1_sample

#主成分分析降噪
def PCA_Denoise(com_all_data):
    sklearn_pca = sklearnPCA(n_components=3)
    com_all_data = sklearn_pca.fit_transform(com_all_data)
    print("PCA处理之后的数据数量：",len(com_all_data),"    每一条的数据的长度",len(com_all_data[0]))
    return com_all_data

#bagging分类器
def Bagging_Classifier(com_all_data,y1):
    scaler = StandardScaler()  # 数据标准化
    scaler.fit(com_all_data)  # 训练标准化对象
    com_all_data = scaler.transform(com_all_data)  # 转换数据集

    feature_train, feature_test, target_train, target_test = train_test_split(com_all_data, y1, test_size=0.3, random_state=0)
    print("数据长度：", len(feature_train), "    ", len(feature_train[0]),"    ",feature_train[0])
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    # n_estimators=500:生成500个决策树
    clf = BaggingClassifier(base_estimator=tree, n_estimators=100, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=1)
    clf.fit(feature_train, target_train)
    predict_results = clf.predict(feature_test)
    print(accuracy_score(predict_results, target_test))
    conf_mat = confusion_matrix(target_test, predict_results)
    print(conf_mat)
    print(classification_report(target_test, predict_results))

#adaBoost分类器
def AdaBoost_classifier(com_all_data,y1):
    com_all_data = PCA_Denoise(com_all_data)

    feature_train, feature_test, target_train, target_test = train_test_split(com_all_data, y1, test_size=0.3, random_state=0)
    AB = AdaBoostClassifier(n_estimators=1000)
    AB.fit(feature_train, target_train)
    predict_results = AB.predict(feature_test)
    print(accuracy_score(predict_results, target_test))
    conf_mat = confusion_matrix(target_test, predict_results)
    print(conf_mat)
    print(classification_report(target_test, predict_results))


if __name__ == '__main__':
    data_address = "E:/深水半潜式/4data/T_801_3633178X111_OlzIVXSBfwJyXScFTA.csv"
    data = read_bigata(data_address)
    com_all_data = []  #该数组用于存储融合之后的数据：：由于不知道一共会生成多少项数据且数据量巨大，所以数据类型定义为list，后面数据全部添加完毕之后将其转为array
    last_time = pd.to_datetime('2020-04-01 00:00:00.000000')  #上一次的时间
    temp_1s_data = np.zeros(shape=(0,111))   #该数组用于存储1s内的数据，并用于求平均值：：虽然无法定义数组长度，但是数组的数据量不会太大，可以定义为array
    for chunk in data:
        chunk = np.array(chunk)
        com_all_data,last_time,temp_1s_data = Com_Data(chunk,com_all_data,last_time,temp_1s_data)
        # print("融合后数据的数量:",len(com_all_data))
        # print("（主方法内）上一时间节点:",last_time)
        # print("1s内的剩余临时数据:",temp_1s_data)

        com_all_data = np.array(com_all_data)
        print("包含自添加数据的数组总长度：",len(com_all_data))



        # com_all_data = Delete_Data(com_all_data)


        error_address = "E:/深水半潜式/报警数据/EventHS/4_data_error_final-2.csv"
        error_data = read_data(error_address)
        error_data = np.array(error_data)
        # print("errordata  len:",len(error_data))
        # for i in range(len(error_data)):
        #     print("cuowu riqi:",error_data[i][0])
        y1 = []   #标签::数据量较大，长度为len(com_all_data)
        # y1 = Associated_Data_Old(len(com_all_data),error_data,y1)   #旧方法，拟造标签，每1s都会生成标签，无论是否有数据
        com_all_data,y1 = Associated_Data(com_all_data, error_data, y1)   #以时间为标准，为离报错数据最近的数据标记错误等级，其他标记正常
        # print("所有数据标签（详细）：",y1)
        y1 = np.array(y1)
        # print("所有数据标签（简略）：",y1)
        print("数据长度：",len(com_all_data),"    ","标签长度：",len(y1))


        #采样前重新划分数据集
        # feature_train, feature_test, target_train, target_test = Re_Split(com_all_data,y1)
        #采样
        sample_num = 9000 #过采样和欠采样的数据数量
        com_all_data_sample = []
        y1_sample = []
        # com_all_data_sample,y1_sample = Simple_Over_Sample(com_all_data,y1,com_all_data_sample,y1_sample,sample_num)
        com_all_data_sample, y1_sample = Three_Over_Sample(com_all_data, y1, com_all_data_sample, y1_sample, sample_num)
        com_all_data_sample,y1_sample = Under_Sample(com_all_data,y1,com_all_data_sample,y1_sample,sample_num)

        Bagging_Classifier(com_all_data_sample,y1_sample)
        # AdaBoost_classifier(com_all_data_sample,y1_sample)



