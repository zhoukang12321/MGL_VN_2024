import math
import h5py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.nn.parameter import Parameter

#有几个问题
#resnet50直接从图像跑，太慢了
#稀疏矩阵乘法的速度
#target的全连接层是300x300
#g
#不带参数的GAT
torch.cuda.current_device()
torch.cuda._initialized = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ScenePriorsGATModel(nn.Module):
    """Scene Priors implementation"""
    def __init__(
        self,
        action_sz,
        state_sz = 8192,
        target_sz = 300,
        ):
        super(ScenePriorsGATModel, self).__init__()
        target_embed_sz = 300
        self.fc_target = nn.Linear(target_sz, target_embed_sz)
        # Observation layer
        self.fc_state = nn.Linear(state_sz, 512)

        # GAT layer
        self.gat = GAT()

        # Merge word_embedding(300) + observation(512) + gcn(512)
       
        self.navi_net = nn.Linear(
            target_embed_sz+ 1024, 512)#
        self.navi_hid = nn.Linear(512,512)
        #output
        self.actor_linear = nn.Linear(512, action_sz)
        self.critic_linear = nn.Linear(512, 512)

    def forward(self, model_input):

        x = model_input['fc|4'].reshape(-1, 8192)
        y = model_input['glove']
        z = model_input['score']
        x = self.fc_state(x)
        x = F.relu(x, True)
        y = self.fc_target(y)
        y = F.relu(y, True)
        z = self.gat(z)#
        z=z.reshape(1,512)
        #print(x.shape,y.shape,z.shape)
        xyz = torch.cat((x, y, z), dim = 1)
        xyz = self.navi_net(xyz)#(4,512)
        xyz = F.relu(xyz, True)#(4,512)
        xyz = F.relu(self.navi_hid(xyz), True)#(512,512)
        return dict(
            policy=self.actor_linear(xyz),
            value=self.critic_linear(xyz)
            )

class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()

        # Load adj matrix for GAT有向图的邻接矩阵
        #A_raw = torch.load("../thordata/gcn/obj.dat")
        A_raw = torch.load ("../thordata/gcn/obj_direct.dat")
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A).detach())
        #print('self.A',self.A.size())
		#新的有向图，有向图的glove以及有向图的关系权重邻接矩阵
        #objects = open("../thordata/gcn/objects_ori.txt").readlines()
        objects = open ("../thordata/gcn/direction_obj.txt").readlines ()
        objects = [o.strip() for o in objects]
        self.n = len(objects)
        glove = h5py.File ("../thordata/word_embedding/glove_map300d_direct.hdf5", "r", )
        glove.requires_grad=True
        #print("???",self.n,".........................................................单词个数",objects)
        self.register_buffer('all_glove', torch.zeros(self.n, 300).detach())
		#glove需要自己构建新的有向图glove,在原有基础上追加新的目标
        #glove = h5py.File("../thordata/word_embedding/glove_map300d_direct.hdf5","r",)

        self.all_glove.requires_grad=True
        self.all_glove_=self.all_glove.data
        #print(type(self.all_glove))
        #print((self.all_glove).shape)
        #print(type(glove),"1111111111111111111")
        for i in range(self.n):
            #print("here")
            self.all_glove_[i, :]= torch.from_numpy(glove[objects[i]][:])
        #print(self.all_glove.shape,type(self.all_glove),"22222222222222222")
        #self.all_glove=
        #glove.close()
        #print("heeeee")
        self.all_glove=self.all_glove_
        nhid = 512
        # Convert word embedding to input for gat
        self.word_to_gat = nn.Linear(300, 512)

        # Convert resnet feature to input for gat
        self.resnet_to_gat = nn.Linear(1000, 512)
        #self.resnet_to_gat = nn.Linear (1000, 512)

        # Gat net
        self.gat1 = GraphAttentionConvolution( 512+512, nhid,dropout=0.2,alpha=0.02)#512,1024
        self.gat2 = GraphAttentionConvolution(nhid, nhid,dropout=0.2,alpha=0.02)#1024,1024
        self.gat3 = GraphAttentionConvolution(nhid, 512,dropout=0.2,alpha=0.02)#1024,1

        self.mapping = nn.Linear(512,self.n)#92到512
    def gat_embed(self, x,params):
        #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx=",x.shape)
        if params==None:
            #print("xxx",x.shape)
            resnet_embed = self.resnet_to_gat(x)
            #print(self.all_glove.shape,"111")
            word_embedding = self.word_to_gat(self.all_glove.detach())

            n_steps = resnet_embed.shape[0]
            resnet_embed = resnet_embed.repeat(self.n,1,1)
        #print("resnet",resnet_embed.shape,"word_embedding",word_embedding.shape,n_steps)
            output = torch.cat(
                (resnet_embed.permute(1,0,2), word_embedding.repeat(n_steps,1,1)),
                dim=2
                )
        #print('output=',output.size())#1,92,1024
            output=output.squeeze(0)
        else:
            resnet_embedx = self.resnet_to_gat(x)
            #print(self.all_glove.shape,"222")
            word_embeddingx = self.word_to_gat(self.all_glove.detach())
            resnet_embed = F.linear(
                #resnet_embedx,
                x,
                weight=params["resnet_to_gat.weight"],
                bias=params["resnet_to_gat.bias"],
            )
            #self.all_glove.requires_grad=True
            #self.all_glove_=self.all_glove.data
            #self.all_glove=self.all_glove_
            #print("self.all_glove",self.all_glove.shape)
            self.all_glove_=self.all_glove.clone().detach()
            #print(";;;;;")
            word_embedding = F.linear(
                self.all_glove_,
                #word_embeddingx,
                weight=params["word_to_gat.weight"],
                bias=params["word_to_gat.bias"],
            )
            #print("www",word_embedding.shape)
            n_steps = resnet_embed.shape[0]
            resnet_embed = resnet_embed.repeat(self.n,1,1)
            output = torch.cat(
                (resnet_embed.permute(1,0,2), word_embedding.repeat(n_steps,1,1)),
                dim=2
                )
        #print('output',output.shape)#(512,92,1024)

        return output
    def gat2_embed(self, x):

        resnet_embed = self.resnet_to_gat(x)

        word_embedding = self.word_to_gat(self.all_glove.detach()).clone()

        resnet_embed=resnet_embed.repeat(92,1)
        #output = torch.cat ((resnet_embed, word_embedding),dim=0)#2048*92

        n_steps = resnet_embed.shape[0]
        resnet_embed = resnet_embed.repeat(self.n,1,1)
        output = torch.cat((resnet_embed.permute(1,0,2), word_embedding.repeat(n_steps,1,1)),dim=2)
        return output
    def forward(self, x,params=None):

        # x = (current_obs)
        # Convert input to gcn input
        #print('x,params',x.size())#512,1,1,1000
        x = self.gat_embed(x,params)#(512,1000)#
        #print("xxxx",x.shape)#92,1024
        #print(x.shape,"x====????",self.A.shape)#92,1024-->92,92
        fc10=nn.Linear(10,1).cuda()
        if x.shape[0]==10:
            x=fc10(x.view(-1,10)).view(x.shape[1],-1)
        if params==None:
            #print("xxx",x.shape,self.A.shape)
            #92,1024--92,92
            x = F.relu(self.gat1(x, self.A))#512,1024
        #print("xxxx",x.shape)#92,512
        #x=x.unsqueeze(0)
            #print (x.shape, "x====????2222", self.A.shape)  # 92,512-->92,92
            x = F.relu(self.gat2(x, self.A))#1024,1024
        #print("xxx",x.shape)#92,256
        #print (x.shape, "x====????3333", self.A.shape)  # 92,512->92,92
            x = F.relu(self.gat3(x, self.A))#1024,1
        #print (x.shape, "x====????44444", self.A.shape)  # 92,512->92,92
            x.squeeze_(-1)
        #print("xxxx",x.shape)#92,128
            x = self.mapping(x)
        #print (x.shape, "x====????5555", x.shape)#92,92
        else:
            gat_p = [
                dict(
                    weight = params[f'gat{x}.weight'], bias = params[f'gat{x}.bias']
                    )
                for x in [1,2,3]
                ]
            #print("xxx2",x.shape)
            x = F.relu(self.gat1(x.detach().clone(), self.A.detach().clone(), gat_p[0])).detach().clone()
            x = F.relu(self.gat2(x.detach().clone(), self.A.detach().clone(), gat_p[1])).detach().clone()
            x = F.relu(self.gat3(x.detach().clone(), self.A.detach().clone(), gat_p[2])).detach().clone()
            x.squeeze_(-1)
            #print("xxxxxxxxxxx",x.shape)
            x = F.linear(
                    x,
                    weight=params["mapping.weight"],
                    bias=params["mapping.bias"],
                )

        return x
class GraphAttentionConvolution(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__ (self, in_features, out_features, dropout, alpha, concat=True, bias=True):
        super (GraphAttentionConvolution, self).__init__ ()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.weight = nn.Parameter (torch.empty (size=(in_features, out_features)))
        #self.W = nn.Parameter (torch.empty (size=(in_features, out_features)))
        nn.init.xavier_uniform_ (self.weight.data, gain=1.414)
        self.a = nn.Parameter (torch.empty (size=(2 * out_features, 1)))
        nn.init.xavier_uniform_ (self.a.data, gain=1.414)
    
        self.leakyrelu = nn.LeakyReLU (self.alpha)

        # print('w:=',self.weight.size())
        if bias:
            self.bias = Parameter (torch.FloatTensor (out_features))
        else:
            self.register_parameter ('bias', None)

        nn.init.xavier_uniform_ (self.weight.data, gain=1.414)

        # attention初始权重
        self.A = nn.Parameter (torch.zeros (size=(92, 92)))
        nn.init.xavier_uniform_ (self.A.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU (self.alpha)
        self.reset_parameters ()
    # 将上次训练的参数保存到weight和bias中
    def reset_parameters (self):
        stdv = 1. / math.sqrt (self.weight.size (1))
        self.weight.data.uniform_ (-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_ (-stdv, stdv)


    def forward2(self, h, adj,params=None):
        #print("h==",h.shape,adj.shape,self.weight.shape)#,92,1024-->92,92-->512,1024
        h=h.squeeze(0)
        Wh = torch.mm (h, self.weight)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input (Wh)
    
        zero_vec = -9e15 * torch.ones_like (e)
        attention = torch.where (adj > 0, e, zero_vec)
        attention = F.softmax (attention, dim=1)
        attention = F.dropout (attention, self.dropout, training=self.training)
        h_prime = torch.matmul (attention, Wh)
        if params==None:
            if self.bias is not None:
                return h_prime+self.bias
            else:
                return h_prime
        elif params!=None:
            params ['weight']=params['weight'].view(1024,-1)
            support2 = torch.matmul(input, params['weight'])
            
            #print('attention',attention.shape,'support',support.shape)
            support=support2.view(92,-1)
            h_prime = torch.matmul(attention, support)
            if self.bias is not None:
                #print('output',output.shape,'params_bias]',params['bias'].shape)
                h_prime=h_prime.view(-1,512)
                params ['bias']=params['bias'].repeat(1,1)
                
                return h_prime + params['bias']
    
        if self.concat:
            return F.elu (h_prime)
        else:
            return h_prime
    def forward(self, input, adj, params = None):
        #print("111input",input.shape,adj.shape)
        #input=np.squeeze(input)
        #92,1024--92,92

        #print(input.shape,self.weight.shape,"222")
        #92,512--512,256
        #92,1024--92,512
        #if self.weight.shape[1]==512:
            #self.weight=self.weight.view(1024,-1).cuda()
        Wh=torch.matmul(input,self.weight)
        # Wh=torch.matmul(input,self.W.t())#第一轮184，512->1024,512第二轮1，92，512->512，512
        #print('Wh=',Wh.shape)
        a_input2=self._prepare_attentional_mechanism_input(Wh)
        #print('a_input',a_input.shape,self.A.shape)
        a_input=a_input2.view(92,-1).detach()
        an=a_input.size()[1]
        
        #print('an====',an)aa
        aa=self.A.view(an,-1)
        #print ('self.A', a_input.shape,self.A.size (),an)#1024,92*4--92,368
        e=self.leakyrelu(torch.matmul(a_input,aa).squeeze(1))
        #print("eee",e.shape,a_input.shape,self.A.shape,an)
        #92,92---92,368--4096,46--368
        #print('a_input',a_input.shape,'self.A==',self.A.shape,'e==',e.shape)
 
        zero_vec = -9e15 * torch.ones_like (e)
        
        #print("adj",adj.shape,e.shape,zero_vec.shape)
        attention = torch.where (adj > 0, e, zero_vec)#把大于0的都变成0
        attention = F.softmax (attention, dim=1)
        #self.droupout=0.2
        attention = F.dropout (attention,self.dropout,training=self.training)
        #print('attention',attention.shape,'Wh==',Wh.shape)#true 92,92->184，1024
        #92,92--92,46
        att=attention.view(-1,92)
        output = torch.matmul(att, Wh)#Wh
        #print('output=',output.shape)#46,1024
        #fc512=nn.Linear(46,512).cuda()
        #output=fc512(output)
        #output=output.repeat(1,).view(-1,512)
        if params == None:
            #support = torch.matmul(input, self.weight)
            #output = torch.matmul(adj, support)
            if self.bias is not None:
                #print('output',output.shape,'self.bias=',self.bias.shape)#92,512....512
                return output + self.bias
            else:
                #print(output)
                return output
        elif params!=None:
            #print('weight_size',params['weight'].size())
            params ['weight']=params['weight'].view(1024,-1)
            if params['weight'].shape[1]==256:
                params['weight']=params['weight'].view(512,512)
            #print(input.shape,params['weight'].shape,"222")
            support3= torch.matmul(input, params['weight'])
            
            #print('attention',attention.shape,'support',support3.shape)
            support=support3.view(92,-1)
            output = torch.matmul(attention, support)
            if self.bias is not None:
                #print('output',output.shape,'params_bias]',params['bias'].shape)
                fout=nn.Linear(256,1024).cuda()
                if output.size()[1]==256:
                    
                    output=fout(output)
                else:pass
                output=output.view(-1,512)
                
                
                return output + params['bias']
        elif self.concat:
            #print('after_concat',output.shape)
            return F.elu(output)
        else:
            #print('output=',output.shape)
            return output
    def _prepare_attentional_mechanism_input (self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        #print(Wh.shape,"whhh")
        Wh=Wh.view(92,512)
        #fc512=nn.Linear(46,512).cuda()
        #Wh=fc512(Wh)
        #print("wh",Wh.shape,self.a [:self.out_features, :].shape)
        #184,46--512,1
        Wh1 = torch.matmul (Wh, self.a [:self.out_features, :])
        Wh2 = torch.matmul (Wh, self.a [self.out_features:, :])
        # broadcast add
        #print("Wh1",Wh1.shape,Wh2.shape)
        e = Wh1 + Wh2.t()
        #print(e.shape,"eeee")
        return self.leakyrelu (e)

    def __repr__ (self):
        return self.__class__.__name__ + ' (' + str (self.in_features) + ' -> ' + str (self.out_features) + ')'

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

if __name__ == "__main__":
    model = ScenePriorsGATModel(9)
    input1 = torch.randn(4,8192)#resent
    input2 = torch.randn(4,300)#glve
    input3 = torch.randn(1,1,1,1000)
    #input3= torch.randn (512, 1,1,1000)#score
    #out = model.forward({'fc|4':input1, 'glove':input2, 'RGB':input3})
    out = model.forward ({'fc|4': input1, 'glove': input2, 'score': input3})
    print(out['policy'])
    print(out['value'])

