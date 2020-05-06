import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import argparse
import h5py
import time

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-w","--n_way",type = int, default = 20)                      # way
parser.add_argument("-s","--n_shot",type = int, default = 5)       # support set per class
parser.add_argument("-b","--n_query",type = int, default = 15)       # query set per class
parser.add_argument("-e","--episode",type = int, default= 10000)
#-----------------------------------------------------------------------------------#
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
n_way = args.n_way
n_shot = args.n_shot
n_query = args.n_query
EPISODE = args.episode
#-----------------------------------------------------------------------------------#
LEARNING_RATE = args.learning_rate
GPU = args.gpu

n_examples = 200
im_width, im_height, depth = 17, 17, 100 # cube
Set_iter_routing = 3
num_fea = 32
num_fea_2 = num_fea*2
num_fea_3 = num_fea_2*2
num_fea_4 = num_fea_3*2


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    # Conv3d(in_depth, out_depth, kernel_size, stride=1, padding=0)
    # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))

    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(100, num_fea, kernel_size=1,padding=0),
                        nn.BatchNorm2d(num_fea),
                        nn.ReLU())

        self.res1 = nn.Sequential(
                        nn.Conv2d(num_fea, num_fea, kernel_size=3,padding=1),
                        nn.BatchNorm2d(num_fea),
                        nn.ReLU(),
                        nn.Conv2d(num_fea, num_fea, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_fea),
                        nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_fea_2),
            nn.ReLU())

        self.res2 = nn.Sequential(
                        nn.Conv2d(num_fea_2,num_fea_2,kernel_size=3,padding=1),
                        nn.BatchNorm2d(num_fea_2),
                        nn.ReLU(),
                        nn.Conv2d(num_fea_2, num_fea_2, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_fea_2),
                        nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_fea_2, num_fea_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_fea_3),
            nn.ReLU())

        self.res3 = nn.Sequential(
                        nn.Conv2d(num_fea_3,num_fea_3,kernel_size=3,padding=1),
                        nn.BatchNorm2d(num_fea_3),
                        nn.ReLU(),
                        nn.Conv2d(num_fea_3, num_fea_3, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_fea_3),
                        nn.ReLU())

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.layer4 = nn.Sequential(
            nn.Conv2d(num_fea_3, num_fea_4, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_fea_4),
            nn.ReLU())

    def forward(self,x):

        out = self.layer1(x)
        out1 = self.res1(out) + out
        out1 = self.maxpool(out1)

        out2 = self.layer2(out1)
        out2 = self.res2(out2) + out2
        out2 = self.maxpool(out2)

        out3 = self.layer3(out2)
        out4 = self.res3(out3) + out3
        out4 = self.maxpool(out4)

        out5 = self.layer4(out4)

        #out = out.view(out.size(0),-1)
        #print(list(out5.size())) # [100, 128, 1, 1]
        return out5 # 64


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(num_fea_4*2, 512, kernel_size=1,padding=0),
                        nn.BatchNorm2d(512),
                        nn.ReLU())
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p = 0.5)                                                                              # 测试的时候需要修改....？？？

    def forward(self,x): # [7600, 128, 2, 2]
        out = self.layer1(x)
        #print(list(out.size()))
        #print(list(out.size())) # [6000, 128, 2, 2]
        out = out.view(out.size(0),-1) # flatten
        #print(list(out.size())) # [6000, 512]
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.sigmoid(self.fc2(out))
        #print("ssss", list(out.size())) # [6000, 1]
        return out

def squash(tensor):
    norm = (tensor * tensor).sum(-1)
    scale = norm / (1+norm)
    return scale.unsqueeze(-1) * tensor / torch.sqrt(norm).unsqueeze(-1)

class DynamicRouting(nn.Module):
    def __init__(self, hidden_size):
        super(DynamicRouting, self).__init__()
        self.hidden_size = hidden_size
        self.l_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, encoder_output, iter_routing=Set_iter_routing):
        C, K, H = encoder_output.shape
        b = torch.zeros(C, K).cuda()
        for _ in range(iter_routing):
            d = F.softmax(b, dim=-1).cuda()
            encoder_output_hat = self.l_1(encoder_output)
            c_hat = torch.sum(encoder_output_hat*d.unsqueeze(-1), dim=1)
            c = squash(c_hat)

            b = b + torch.bmm(encoder_output_hat, c.unsqueeze(-1)).squeeze()

        # write.add_embedding(c, metadata=[0, 1, 2, 3, 4],)

        return c

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def train():

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()
    dynamic_routing = DynamicRouting(hidden_size=num_fea_4)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)
    #dynamic_routing.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)
    dynamic_routing.cuda(GPU)

    feature_encoder.train()
    relation_network.train()
    dynamic_routing.train()

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(optimizer=feature_encoder_optim, step_size=1000, gamma=0.5)

    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=1000, gamma=0.5)

    dynamic_routing_optim = torch.optim.Adam(dynamic_routing.parameters(), lr=LEARNING_RATE)
    dynamic_routing_scheduler = StepLR(dynamic_routing_optim, step_size=1000, gamma=0.5)



    f = h5py.File(r'.\data\SA-train-17-17-100.h5', 'r')
    train_dataset = f['data'][:]
    f.close()
    train_dataset = train_dataset.reshape(-1, n_examples, 17, 17, 100)  # 划分成了48类，每类200个样本
    train_dataset = train_dataset.transpose((0, 1, 4, 2, 3))
    print(train_dataset.shape) # (207400, 103, 9, 9)
    n_train_classes = train_dataset.shape[0]

    accuracy_ = []
    loss_ = []

    s_time = time.time()
    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        dynamic_routing_scheduler.step(episode)

        # ##################################################
        # start:##################################################
        epi_classes = np.random.permutation(n_train_classes)[
                      :n_way]
        support = np.zeros([n_way, n_shot, depth, im_height, im_width], dtype=np.float32)  # n_shot = 5
        query = np.zeros([n_way, n_query, depth, im_height, im_width], dtype=np.float32)  # n_query= 15
        # (N,C_in,H_in,W_in)

        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_examples)[:n_shot + n_query]
            support[i] = train_dataset[epi_cls, selected[:n_shot]]
            query[i] = train_dataset[epi_cls, selected[n_shot:]]

        support = support.reshape(n_way * n_shot, depth, im_height, im_width)
        query = query.reshape(n_way * n_query, depth, im_height, im_width)
        labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8).reshape(-1)
        # print(labels)

        support_tensor = torch.from_numpy(support)
        query_tensor = torch.from_numpy(query)
        label_tensor = torch.LongTensor(labels)
        # end:####################################################################################
        # end:####################################################################################


        # calculate features#################################################################################
        sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))
        #print( list(sample_features.size()) ) # [100, 64, 1, 1]
        sample_features = sample_features.view(n_way, n_shot, list(sample_features.size())[-3])  # view函数改变shape
        #print(list(sample_features.size())) # [20, 5, 64]

        # induction
        sample_features = dynamic_routing(sample_features)
        #print(list(sample_features.size())) # [20, 64]

        batch_features = feature_encoder(Variable(query_tensor).cuda(GPU)).squeeze()
        #print(list(batch_features.size())) # [300, 64]

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(n_query * n_way, 1, 1)  # # repeat函数沿着指定的维度重复tensor
        #print(list(sample_features_ext.size())) # [300, 20, 64]
        batch_features_ext = batch_features.unsqueeze(0).repeat(n_way, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        #print(list(batch_features_ext.size())) # [300, 20, 64]

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2)
        relation_pairs = relation_pairs.view(-1,  list(relation_pairs.size())[-1], 1, 1)
        #print(list(relation_pairs.size())) # [6000, 128]

        relations = relation_network(relation_pairs)
        #print(list(relations.size())) # [6000, 1]
        relations = relations.view(-1, n_way)
        #print(list(relations.size())) # [300, 20]

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(
            torch.zeros(n_query * n_way, n_way).scatter_(dim=1, index=label_tensor.view(-1, 1), value=1).cuda(GPU))
        loss = mse(relations, one_hot_labels)

        # training
        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()


        if (episode+1)%50 == 0:
            print("episode:",episode+1,"loss",loss)
            ##################################
            _, predict_label = torch.max(relations.data, 1)
            predict_label = predict_label.cpu().numpy().tolist()
            #print(predict_label)
            #print(labels)
            rewards = [1 if predict_label[j] == labels[j] else 0 for j in range(labels.shape[0])]
            # print(rewards)
            total_rewards = np.sum(rewards)
            # print(total_rewards)

            accuracy = total_rewards*100.0 / labels.shape[0]
            print("accuracy:", accuracy)
            accuracy_.append(accuracy)
            loss_.append(loss.item())

    f = open('./results/SA_meta_train_time.txt', 'w')
    f.write(str(time.time()-s_time) + '\n')

    f = open('./results/SA_loss.txt', 'w')
    for i in range(np.array(loss_).shape[0]):
        f.write(str(loss_[i]) + '\n')
    f = open('./results/SA_accuracy.txt', 'w')
    for i in range(np.array(accuracy_).shape[0]):
        f.write(str(accuracy_[i]) + '\n')



if __name__ == '__main__':
    train()


