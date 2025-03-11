import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        """
        GCN
        :param in_c: input channels
        :param hid_c:  hidden nodes
        :param out_c:  output channels
        """
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data):
        graph_data = data["graph"][0]  # [N, N]
        graph_data = self.process_graph(graph_data)
        flow_x = data["flow_x"]  # [B, N, H, D]
        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C]
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [N, N], [B, N, Hid_C]
        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C]

        return output_2.unsqueeze(2)

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        graph_data += matrix_i  # A~ [N, N]

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

        degree_matrix = torch.diag(degree_matrix)  # [N, N]

        return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A)




class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c, alpha=0.2):
        """
        graph attention layer
        :param in_c:
        :param out_c:
        :param alpha:
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_c, out_c)))
        nn.init.xavier_normal_(self.W.data)
        self.a = nn.Parameter(torch.empty(size=(2 * out_c, 1)))
        nn.init.xavier_normal_(self.a.data)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, features, adj):
        B, N = features.size(0), features.size(1)
        adj = adj + torch.eye(N, dtype=adj.dtype).cuda()  # A+I
        h = torch.matmul(features, self.W)  # [B,N,out_features]
        # [B, N, N, 2 * out_features]
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_c)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [B,N, N, 1] => [B, N, N]
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [B,N,N]
        attention = F.softmax(attention, dim=2)  # softmax [N, N]
        h_prime = torch.matmul(attention, h)  # [B,N, N]*[N, out_features] => [B,N, out_features]
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads=6):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param out_c: int, number of output channels.
        :param n_heads: how many heads
        """
        super(GAT, self).__init__()
        self.attentions = nn.ModuleList([GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])
        self.conv2 = GraphAttentionLayer(hid_c*n_heads, out_c)
        self.act = nn.ELU()

    def forward(self, data):
        # data prepare
        adj = data["graph"][0]  # [N, N]
        x = data["flow_x"]  # [B, N, H, D]
        B, N = x.size(0), x.size(1)
        x = x.view(B, N, -1)  # [B, N, H*D]

        # forward
        outputs = self.act(torch.cat([attention(x, adj) for attention in self.attentions], dim=-1))
        output_2 = self.act(self.conv2(outputs, adj))

        return output_2.unsqueeze(2)  # [B,1,N,1]

