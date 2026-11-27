# gcn_/models/cgal.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 2


# ------------------ 1. 多视图图编码器 ------------------
class ViewEncoder(nn.Module):
    """单个视图(距离或流动)的GCN编码器"""

    def __init__(self, n_node, in_dim, hid_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hid_dim)
        self.w2 = nn.Linear(hid_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x, adj):
        # x: (B, N, F)  adj: (B, N, N)
        h = torch.relu(self.w1(torch.bmm(adj, x)))  # 聚合邻居
        h = torch.relu(self.w2(torch.bmm(adj, h)))
        return h.mean(dim=1)  # 图级平均 -> (B, out_dim)


# ------------------ 2. Assemble-Disassemble ------------------
class AssembleDisassemble(nn.Module):
    def __init__(self, n_view, view_dim, emb_dim, recon_dim):
        super().__init__()
        # assemble: 拼接后降维
        self.fc_ass = nn.Linear(n_view * view_dim, emb_dim)
        # disassemble: 升维回各视图
        self.fc_dis = nn.Linear(emb_dim, n_view * recon_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_ass.weight)
        nn.init.zeros_(self.fc_ass.bias)
        nn.init.xavier_uniform_(self.fc_dis.weight)
        nn.init.zeros_(self.fc_dis.bias)

    def assemble(self, view_h_list):
        h = torch.cat(view_h_list, dim=-1)  # (B, n_view*view_dim)
        z = self.fc_ass(h)  # (B, emb_dim)
        return z

    def disassemble(self, z):
        h = self.fc_dis(z)  # (B, n_view*recon_dim)
        h = h.view(z.size(0), -1, self.fc_dis.out_features // len(view_h_list))  # (B, n_view, recon_dim)
        return [h[:, i, :] for i in range(h.size(1))]


# ------------------ 3. 双对抗判别器 ------------------
class Discriminator(nn.Module):
    """通用MLP判别器"""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)  # 未sigmoid，用BCEWithLogits


class MatrixDiscriminator(nn.Module):
    """判别嵌入余弦相似度矩阵vs先验矩阵"""

    def __init__(self, n_reg):
        super().__init__()
        self.n = n_reg
        # 把上三角展平后判别
        triu_dim = n_reg * (n_reg - 1) // 2
        self.net = nn.Sequential(
            nn.Linear(triu_dim, triu_dim // 2),
            nn.ReLU(),
            nn.Linear(triu_dim // 2, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, z, R_prior):
        # z: (B, emb_dim)  R_prior: (B, n_reg, n_reg)
        B = z.size(0)
        M = F.cosine_similarity(z.unsqueeze(2), z.unsqueeze(1), dim=-1)  # (B, n_reg, n_reg)
        diff = (M - R_prior).triu(diagonal=1)  # (B, n_reg, n_reg)
        diff = diff.view(B, -1)  # (B, triu_dim)
        return self.net(diff), M


# ------------------ 4. CGAL 主模型 ------------------
class CGAL(nn.Module):
    """
    CGAL (Cross-chain Graph Adversarial Learning) 模型

    CGAL模型来源于以下论文概念：
    - 结合了多视图图学习和对抗学习的思想
    - 通过对抗训练来学习区域间的相似性关系
    - 可以同时处理多种图结构（如距离图、流动图等）

    相关研究基础：
    1. Multi-view graph learning (如MvGNN等)
    2. Adversarial learning in graph representation (如GraphGAN等)
    3. Graph autoencoders (如VGAE等)
    """

    def __init__(self, n_reg, n_view, node_feat_dim,
                 view_hid=64, view_out=32, emb_dim=64,
                 recon_dim=32, prior_R=None):
        """
        Args:
            n_reg: 区域数量（节点数）
            n_view: 视图数量
            node_feat_dim: 节点特征维度
            view_hid: 视图编码器隐藏层维度
            view_out: 视图编码器输出维度
            emb_dim: 区域嵌入维度
            recon_dim: 重构维度
            prior_R: (n_reg, n_reg) 区域间先验相似度矩阵
        """
        super().__init__()
        self.n_reg = n_reg
        self.n_view = n_view

        # 如果没有提供先验矩阵，则初始化为全1矩阵
        if prior_R is None:
            self.prior_R = torch.ones(n_reg, n_reg)
        else:
            self.prior_R = prior_R

        self.prior_R = self.prior_R.to(DEVICE)

        # 每个视图独立编码器
        self.encoders = nn.ModuleList([
            ViewEncoder(n_reg, node_feat_dim, view_hid, view_out)
            for _ in range(n_view)
        ])

        # Assemble-Disassemble
        self.ass_dis = AssembleDisassemble(n_view, view_out, emb_dim, recon_dim)

        # 双判别器
        self.D_z = Discriminator(emb_dim)  # 嵌入分布
        self.D_M = MatrixDiscriminator(n_reg)  # 相似度矩阵

        # 重构解码器（简单线性）
        self.decoders = nn.ModuleList([
            nn.Linear(recon_dim, node_feat_dim) for _ in range(n_view)
        ])

        # 输出层 - 用于跨链性能预测
        self.output_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim // 2, TARGET_DIM)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for decoder in self.decoders:
            nn.init.xavier_uniform_(decoder.weight)
            nn.init.zeros_(decoder.bias)

        # 初始化输出层
        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x_list, adj_list):
        """
        Args:
            x_list: List[(B, n_reg, F)] 各视图节点特征
            adj_list: List[(B, n_reg, n_reg)] 各视图邻接矩阵

        Returns:
            output: (B, 1, TARGET_DIM) 跨链性能预测结果
        """
        # 确保输入列表长度与视图数量一致
        if len(x_list) != self.n_view or len(adj_list) != self.n_view:
            # 如果输入视图数量不匹配，复制第一个视图
            if len(x_list) == 1 and len(adj_list) == 1:
                x_list = [x_list[0] for _ in range(self.n_view)]
                adj_list = [adj_list[0] for _ in range(self.n_view)]
            else:
                raise ValueError(f"输入视图数量({len(x_list)})与模型配置({self.n_view})不匹配")

        # 1. 各视图编码
        h_views = [enc(x, adj) for enc, x, adj in zip(self.encoders, x_list, adj_list)]

        # 2. Assemble -> 区域嵌入
        z = self.ass_dis.assemble(h_views)

        # 3. 通过输出层进行跨链性能预测
        output = self.output_layer(z)  # (B, TARGET_DIM)
        output = output.unsqueeze(1)  # (B, 1, TARGET_DIM)

        return output

    def forward_with_recon(self, x_list, adj_list):
        """
        带重构的前向传播，用于训练阶段
        """
        # 确保输入列表长度与视图数量一致
        if len(x_list) != self.n_view or len(adj_list) != self.n_view:
            # 如果输入视图数量不匹配，复制第一个视图
            if len(x_list) == 1 and len(adj_list) == 1:
                x_list = [x_list[0] for _ in range(self.n_view)]
                adj_list = [adj_list[0] for _ in range(self.n_view)]
            else:
                raise ValueError(f"输入视图数量({len(x_list)})与模型配置({self.n_view})不匹配")

        # 1. 各视图编码
        h_views = [enc(x, adj) for enc, x, adj in zip(self.encoders, x_list, adj_list)]

        # 2. Assemble -> 区域嵌入
        z = self.ass_dis.assemble(h_views)

        # 3. Disassemble -> 重构视图
        h_recon = self.ass_dis.disassemble(z)
        x_hat = [dec(h) for dec, h in zip(self.decoders, h_recon)]

        # 4. 通过输出层进行跨链性能预测
        output = self.output_layer(z)  # (B, TARGET_DIM)
        output = output.unsqueeze(1)  # (B, 1, TARGET_DIM)

        return output, z, x_hat

    def loss_recon(self, x_list, x_hat):
        return sum(F.mse_loss(x, xh) for x, xh in zip(x_list, x_hat))

    def loss_adv_z(self, z):
        batch_size = z.size(0)
        # 先验：标准正态
        z_real = torch.randn_like(z)
        d_real = self.D_z(z_real)
        d_fake = self.D_z(z)
        # 判别器希望真=1，假=0；生成器希望假=1
        loss_d = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                 F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
        loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        return loss_d, loss_g

    def loss_adv_matrix(self, z):
        batch_size = z.size(0)
        R_batch = self.prior_R.unsqueeze(0).expand(batch_size, -1, -1)
        d_out, M = self.D_M(z, R_batch)
        # 判别器：真=0（diff=0），假=1（diff0）
        loss_d = F.binary_cross_entropy_with_logits(d_out, torch.zeros_like(d_out))
        # 生成器：让判别器以为 diff=0
        loss_g = F.binary_cross_entropy_with_logits(d_out, torch.zeros_like(d_out))
        # 额外 Frobenius 约束（论文 Eq.9）
        fro_loss = ((M - R_batch) ** 2).mean()
        return loss_d, loss_g + fro_loss


# ------------------ 5. 简化版CGAL用于跨链预测 ------------------
class CGAL_CrossChain(nn.Module):
    """
    简化版CGAL模型，专门用于跨链性能预测任务
    """

    def __init__(self, input_dim, hidden_dim=64, num_views=2, num_nodes=10, output_dim=TARGET_DIM):
        super(CGAL_CrossChain, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_views = num_views
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        # 特征投影层
        self.feature_projection = nn.Linear(input_dim, hidden_dim)

        # 多视图编码器
        self.view_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            ) for _ in range(num_views)
        ])

        # 视图融合层
        self.view_fusion = nn.Linear((hidden_dim // 2) * num_views, hidden_dim)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, adj=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            adj: (num_nodes, num_nodes) 邻接矩阵

        Returns:
            output: (batch_size, 1, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        # 特征投影
        x = self.feature_projection(x)  # (batch_size, seq_len, hidden_dim)

        # 为适应多视图，将特征复制到所有节点
        node_features = x.unsqueeze(2).repeat(1, 1, self.num_nodes, 1)  # (batch_size, seq_len, num_nodes, hidden_dim)

        # 平均池化获取图级表示
        graph_features = node_features.mean(dim=2)  # (batch_size, seq_len, hidden_dim)

        # 如果没有提供邻接矩阵，创建全连接图
        if adj is None:
            adj = torch.ones(self.num_nodes, self.num_nodes, device=x.device)

        # 构造多视图输入
        views_x = []
        views_adj = []
        for _ in range(self.num_views):
            views_x.append(graph_features)
            views_adj.append(adj.unsqueeze(0).expand(batch_size, -1, -1))

        # 多视图编码
        view_outputs = []
        for i, encoder in enumerate(self.view_encoders):
            # 简化处理：直接对序列特征进行编码
            view_out = encoder(views_x[i])  # (batch_size, seq_len, hidden_dim//2)
            # 序列级池化
            view_pooled = view_out.mean(dim=1)  # (batch_size, hidden_dim//2)
            view_outputs.append(view_pooled)

        # 视图融合
        fused = torch.cat(view_outputs, dim=-1)  # (batch_size, (hidden_dim//2)*num_views)
        fused = self.view_fusion(fused)  # (batch_size, hidden_dim)

        # 输出层
        output = self.output_layer(fused)  # (batch_size, output_dim)
        output = output.unsqueeze(1)  # (batch_size, 1, output_dim)

        return output
