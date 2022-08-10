import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATConv
# from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.utils.data as Data
from tqdm import tqdm
from tqdm import trange
import math

EPOCH = 101
LR = 0.0001
FOLD = 5
CUDA_AVAILABLE = 1
DRUG = 708
PROTEIN = 1512

#自定义参数
BATCH_SIZE=256

#单分类模型
class One_Class_Classfication(nn.Module):
    def __init__(self,GCN_protein_dim,GCN_node_features_dim,GCN_hidden_size,GCN_output_size,GAT_node_features_dim,GAT_hidden_size,GAT_output_size):
        super(One_Class_Classfication, self).__init__()

        # self.ECFPs_layer = nn.Linear(1024,1024)
        # 定义GCN蛋白质预处理线性层，经过线性层保持与药物的空间一致性
        self.GCN_preProcess = nn.Linear(GCN_protein_dim, 1024)

        # 定义GCN第一层
        self.GCN_layer1 = GCNConv(GCN_node_features_dim, GCN_output_size) #2220*1024->2220*512
        # GCN第二层
        # self.GCN_layer2 = GCNConv(GCN_hidden_size, GCN_output_size) #2220*512->2220*200

        # 定义GAT第一层
        self.GAT_layer1 = GATConv(GAT_node_features_dim, GAT_output_size) #2220*56..->2220*1024
        # GAT第二层
        # self.GAT_layer2 = GATConv(GAT_hidden_size, GAT_output_size) #2220*1024->2220*200

        # 定义CNN第一层
        self.CNN_layer1 = nn.Sequential(#input shape (1,2,400)
            nn.Conv2d(in_channels=1,
                      out_channels=16,# n_filter = 16
                      kernel_size=(3,5), #kernel_size选择？ 6*404->4*400
                      padding=2 #保证con2d出来的图片大小不变 2*400->6*404
                      ),
            nn.LeakyReLU(),
            nn.AvgPool2d(2) #平均池化 [1,1,4,400] -> [1,1,2,200]
        )

        #定义CNN第二层
        self.CNN_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, #input channels
                      out_channels=32, #n_filter
                      kernel_size=(3,5),#kernel size
                      padding=2 #保证con2d出来的图片大小不变 2*400->6*404
                      ), #output shape [1,32,4,200]
            nn.LeakyReLU(),
            nn.MaxPool2d(2) #最大池化 [1,32,4,200] -> [1,32,2,100]
        )

        #定义输出线性层，降维-> batch*2 [关系为0的概率，关系为1的概率]
        self.out = nn.Linear(16*2*200,2) # CNN维度 [1,32,2,100] -> [2]


    def forward(self,GCN_input_drug,GCN_input_protein,GCN_edge,GCN_weight,GAT_input,GAT_edge,idx):
        # print(self)
        #ECFPs线性层修改
        # GCN_input_drug = self.ECFPs_layer(GCN_input_drug)
        #预处理蛋白质数据
        GCN_protein_feature = self.GCN_preProcess(GCN_input_protein)

        # 药物蛋白拼接构成GCN特征矩阵
        GCN_input = torch.cat((GCN_input_drug,GCN_protein_feature),0)

        #替换成了torch.cat()拼接，直接在GPU中完成
        # GCN_input = np.vstack((GCN_input_drug,GCN_protein_feature.numpy()))

        #GCN数据预处理
        GCN_edge = GCN_edge.T

        #GCN是在模型内部处理用的是numpy拼接，所以需要放入cuda进行运算
        # if CUDA_AVAILABLE==1:
        #     GCN_input = GCN_input.cuda()

        #调用GCN层，获得GCN特征：2220*200
        GCN_features = self.GCN_layer1(GCN_input,GCN_edge,GCN_weight)
        GCN_features = GCN_features.to(torch.float32)
        # GCN_features = F.leaky_relu(GCN_features)
        GCN_features = F.dropout(GCN_features,p=0.3,training=self.training)
        # GCN_features = self.GCN_layer2(GCN_features,GCN_edge,GCN_weight)
        GCN_features = F.log_softmax(GCN_features,dim=0)


        #GAT数据预处理
        # GAT_input = torch.from_numpy(GAT_input).float() #在进入前已转换并放入cuda
        GAT_edge = GAT_edge.T #[...,2]->[2,...]

        #调用GAT层获得GAT特征
        GAT_features = self.GAT_layer1(GAT_input,GAT_edge)
        # GAT_features = F.leaky_relu(GAT_features)
        GAT_features = F.dropout(GAT_features,p=0.3,training=self.training)
        # GAT_features = self.GAT_layer2(GAT_features,GAT_edge)
        GAT_features = F.log_softmax(GAT_features,dim=0)
        features = torch.cat((GCN_features, GAT_features), 1)

        #从拼接的药物蛋白特征中抽取当前idx的特征进行降维
        cnn_features = []
        for i in range(idx.shape[0]):
            #药物位置
            r_no = int(idx[i] / PROTEIN)
            #蛋白位置
            p_no = int(idx[i] % PROTEIN)

            #拼接 [2,400]
            cnn_feature = torch.cat((features[r_no, :].unsqueeze(0), features[p_no, :].unsqueeze(0)), 0)

            #加入到数组 [1,2,400]s
            cnn_features.append(cnn_feature.unsqueeze(0))

        #将整个批次做拼接[b,2,400] b->batch
        cnn_features = torch.cat(tuple(cnn_features), 0)
        # print(cnn_features.shape)
        # if CUDA_AVAILABLE ==1:
        #     cnn_features = cnn_features.cuda() #cnn使用torch拼接的，全是GPU中的，不用cuda了torch

        #加入channel维度 [b,1,2,400]
        cnn_features = cnn_features.unsqueeze(1).float()
        # print(cnn_features)

        embedding_cnn = self.CNN_layer1(cnn_features) #[1,1,2,400] -> [1,16,2,200]
        # print(embedding_cnn.shape)

        # embedding_cnn = self.CNN_layer2(embedding_cnn) #[1,16,2,200] -> [1,32,2,100]
        # print(embedding_cnn.shape)
        embedding_cnn = torch.Tensor.tanh(embedding_cnn)

        #获取最终的维度
        b,n_f,h,w = embedding_cnn.shape

        #除了批次维度，其他拉直过线性层降维
        output = self.out(embedding_cnn.view(b, n_f * h * w))
        # print(output.shape)
        return output

#模型训练方法
def model_test(BATCH_SIZE,epo):
    # GCN相关特征：
    protein_vec = np.loadtxt("../data/protein_vector.txt")  # 蛋白质序列的One-Hot编码
    GCN_protein_features = torch.from_numpy(protein_vec / 23).float()  # 蛋白质序列
    GCN_drug_features = np.loadtxt("../data/drug_features.txt")  # 药物ECFPs特征
    GCN_drug_features = torch.from_numpy(GCN_drug_features).float()

    # GCN_input = np.vstack((drug_features,protein_features.detach().numpy()))#由于GCN的蛋白质特征需要先经过线性层处理，因此必须在模型中拼接
    # GCN相关边
    S_Protein = np.loadtxt("../data/Similarity_Matrix_Proteins.txt")  # 蛋白质相似性 1512*1512
    S_Drug = np.loadtxt("../data/Similarity_Matrix_drugs.txt")  # 药物相似性 708*708
    RPI = np.loadtxt("../data/mat_drug_protein_new.txt")  # 药物蛋白相互作用
    #拼接
    drug_adj = np.hstack((S_Drug, RPI))  # 横向拼接708*2220 组成药物特征
    protein_adj = np.hstack((RPI.T, S_Protein))  # 1512*2220
    GCN_adj = np.vstack((drug_adj, protein_adj))  # GCN拼接边 2220*2220 其中相似性带有权重关系

    GCN_edge = []
    GCN_weights = []
    # 构造pyg格式的边关系->二元组边关系，并分离权重
    for i, row in enumerate(GCN_adj):
        for j, num in enumerate(row):
            if num > 0:
                GCN_edge.append([i, j])
                GCN_weights.append(num)

    # GAT相关特征
    RDI = np.loadtxt("../data/mat_drug_disease.txt")  # 药物疾病关联
    PDI = np.loadtxt("../data/mat_protein_disease.txt")  # 蛋白质疾病关联
    GAT_features = np.vstack((RDI, PDI))  # 拼接，可以在外拼接，无需训练特征，节约GPU显存
    # GAT_features = torch.from_numpy(GAT_features).float()

    # GAT边相关
    RRI = np.loadtxt("../data/mat_drug_drug.txt")  # 药物之间相互作用
    PPI = np.loadtxt("../data/mat_protein_protein.txt")  # 蛋白间相互作用
    RPI = np.loadtxt("../data/mat_drug_protein_new.txt")  # 药物蛋白相互作用

    # 拼接->相互作用矩阵拼接
    drug_adj = np.hstack((RRI, RPI))
    protein_adj = np.hstack((RPI.T, PPI))
    GAT_adj = np.vstack((drug_adj, protein_adj))  # 2220*2220
    GAT_edge = []
    # 构造pyg格式的边关系
    for i, row in enumerate(GAT_adj):
        for j, num in enumerate(row):
            if num == 1:
                GAT_edge.append([i, j])
    #list->torch:需要将数据先转变成numpy再torch.fromnumpy()  /  torch.tensor(data,dtype=)
    GAT_edge = np.array(GAT_edge)


    #数据格式转换：numpy->torch
    GCN_edge = torch.tensor(GCN_edge, dtype=torch.long)
    GCN_weights = torch.tensor(GCN_weights, dtype=torch.float32)

    GAT_edge = torch.from_numpy(GAT_edge)
    GAT_features = torch.from_numpy(GAT_features).float()

    #将全部的1的位置读入
    # index_1 = np.loadtxt("result/DTI_index_1.txt")
    # index_0 = np.loadtxt("result/DTI_index_0.txt")
    #进行测试
    for f in range(1):
        #1份1(未训练)，1份0作为测试集
        # fold_index_1 = index_1[4:,:].flatten().tolist()
        # num1 = len(fold_index_1)
        # fold_index_0 = index_0[4:, :].flatten().tolist()

        # 将全部的1的位置读入
        index_1 = np.loadtxt("../pre_process/result/DTI_index_1.txt")
        index_0 = np.loadtxt("../pre_process/result/DTI_index_0.txt")
        # 读入数据

        # 预测测试集中的0
        test_index = index_1[f, :].flatten().tolist() + index_0[f, :].flatten().tolist()

        dataset = Data.TensorDataset(torch.from_numpy(np.array(test_index)))
        dataloader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


        #构造测试集
        # test_index = fold_index_1+ fold_index_0
        model = torch.load("model/epoch_"+str(epo)+".pkl")

        #构造dataset、dataloader
        # dataset = Data.TensorDataset(torch.from_numpy(np.array(test_index)))
        # dataloader = Data.DataLoader(dataset=dataset, batch_size=2*BATCH_SIZE, shuffle=False)

        #CUDA情况下将模型放入CUDA
        if CUDA_AVAILABLE == 1:
            model = model.cuda()
        fold_out = []
        fold_index = []
        for data in tqdm(dataloader):
            batch_idx = data[0]

            # idx预处理
            batch_idx = torch.from_numpy(np.array(batch_idx)).long()

            if CUDA_AVAILABLE == 1:
                batch_idx = batch_idx.cuda()
                GCN_drug_features = GCN_drug_features.cuda()
                GCN_protein_features = GCN_protein_features.cuda()
                GCN_edge = GCN_edge.cuda()
                GCN_weights = GCN_weights.cuda()
                GAT_features = GAT_features.cuda()
                GAT_edge = GAT_edge.cuda()

            # 数据输入模型，得到结果
            batch_out = model(GCN_drug_features, GCN_protein_features, GCN_edge, GCN_weights, GAT_features, GAT_edge,batch_idx)

            if CUDA_AVAILABLE == 1:
                fold_out += batch_out.cpu().detach().numpy().tolist()
                fold_index += batch_idx.cpu().detach().numpy().tolist()

            else:
                fold_out += batch_out.detach().numpy().tolist()
                fold_index += batch_idx.detach().numpy().tolist()

            print('FOLD: ', f, 'Item: ',  math.ceil(len(test_index) / (2 * BATCH_SIZE)))
        np.savetxt('test_result/score_all1_' + str(epo) + '.txt', np.array(fold_out), fmt='%.6f')
        np.savetxt('test_result/idx_all1_' + str(epo) + '.txt', np.array(fold_index), fmt='%.6f')

if __name__ == '__main__':
    for i in range(10,101,10):
        model_test(BATCH_SIZE, i)
        print("epo:",i)
    print("OK")