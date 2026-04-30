import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# ==========================================
# 1. 基础组件：残差卷积块
# ==========================================
class ResCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, padding=7):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


# ==========================================
# 辅助模块：VQBridge
# ==========================================
class VQBridge(nn.Module):
    """
    FVQ 论文中的 VQBridge 模块：
    - 输入：原始可学习码本 C ∈ R[K, D]
    - 输出：映射后的有效码本 Ĉ ∈ R[K, D]（用于实际量化）
    """
    def __init__(
        self,
        num_embeddings=16384,    # 码本大小 K
        embedding_dim=256,       # 原始码本维度 D
        latent_dim=256,          # ViT 隐藏层维度 d'
        depth=2,                 # ViT 编码器层数
        **kwargs
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        # 步骤 1: 投影到潜在空间
        self.proj_in = nn.Linear(embedding_dim, latent_dim)
        self.norm_in = nn.LayerNorm(latent_dim)

        # 步骤 2: 使用 ViT 建模全局交互（所有 K 个码字作为序列输入）
        nhead = 4
        
        encoder_layer = TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN 结构，训练更稳定
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)

        # 步骤 3: 投影回原始维度
        self.norm_out = nn.LayerNorm(latent_dim)
        self.proj_out = nn.Linear(latent_dim, embedding_dim)

    def forward(self, codebook):
        K, D = codebook.shape
        assert K == self.num_embeddings and D == self.embedding_dim, \
            f"输入码本形状 {codebook.shape} 与初始化配置不匹配"

        # 1. 投影到潜在空间：[K, D] -> [K, d']
        x = self.proj_in(codebook)
        x = self.norm_in(x)

        # 2. 添加虚拟 batch 维度以适配 Transformer：[K, d'] -> [1, K, d']
        x = x.unsqueeze(0)
        x = self.transformer(x)      # [1, K, d']
        x = x.squeeze(0)             # [K, d']

        # 3. 恢复到原始维度：[K, d'] -> [K, D]
        x = self.norm_out(x)
        mapped_codebook = self.proj_out(x)

        return mapped_codebook


# ==========================================
# 主模块：FVQ Vector Quantizer
# ==========================================
class VectorQuantizer(nn.Module):
    """
    FVQ 向量量化器
    """
    def __init__(
        self,
        num_embeddings=16384,      # 码本大小 K
        embedding_dim=256,         # 码本维度 D
        beta=0.25,                 # commitment loss 权重
        vqbridge_config=None,      # VQBridge 配置字典
        **kwargs
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # 初始化原始可学习码本 C ∈ R[K, D]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0, 1.0)

        # 初始化 VQBridge
        default_config = dict(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            latent_dim=embedding_dim,  # 默认潜在维度等于原始维度
            depth=2
        )
        if vqbridge_config is not None:
            default_config.update(vqbridge_config)
        self.vqbridge = VQBridge(**default_config)

    def forward(self, z):
        input_shape = z.shape

        # 展平为 [N, D]，便于批量计算
        z_flattened = z.reshape(-1, self.embedding_dim)  # [N, D]

        # 获取有效码本 Ĉ ∈ R[K, D]
        codebook = self.vqbridge(self.embedding.weight)  # [K, D]

        # 计算 L2 距离平方：||z - c_i||^2 = ||z||^2 + ||c_i||^2 - 2*z·c_i
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)      # [N, 1]
            + torch.sum(codebook ** 2, dim=1)                     # [K]
            - 2 * torch.matmul(z_flattened, codebook.t())         # [N, K]
        )  # 最终形状: [N, K]

        # 找到最近邻码字的索引
        indices = torch.argmin(distances, dim=1)  # [N]

        # # 先对输入和码本做L2归一化
        # z_norm = F.normalize(z_flattened, p=2, dim=1)      # [N, D]
        # codebook_norm = F.normalize(codebook, p=2, dim=1)  # [K, D]

        # # 计算余弦相似度（等价于归一化后的点积）
        # cos_sim = torch.matmul(z_norm, codebook_norm.t())  # [N, K]

        # # 找到相似度最高的码字（注意：argmax 替代 argmin）
        # indices = torch.argmax(cos_sim, dim=1)  # [N]

        # 使用有效码本 Ĉ 进行查表量化
        z_q = F.embedding(indices, codebook).reshape(input_shape)  # [*, D]

        # 计算 VQ 损失（标准两部分）
        # 1. Commitment loss: 鼓励 encoder 输出靠近码本（冻结码本梯度）
        commitment_loss = F.mse_loss(z_q.detach(), z)
        # 2. Codebook loss: 鼓励码本靠近 encoder 输出（冻结 encoder 梯度）
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = self.beta * commitment_loss + codebook_loss

        # 直通估计器（Straight-Through Estimator）：
        # 前向使用 z_q，反向梯度直接传给 z
        z_q = z + (z_q - z).detach()

        # 计算 Perplexity（衡量码本使用均匀性）
        encodings = F.one_hot(indices, self.num_embeddings).float()  # [N, K]
        avg_probs = torch.mean(encodings, dim=0)                    # [K]
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, loss, perplexity, indices


# ==========================================
# 3. 异构图交互层
# ==========================================
class HeteroGATInteraction(MessagePassing):
    def __init__(self, channels, heads=4, concat=True, dropout=0.2, negative_slope=0.2):
        """
        channels: 输入特征维度
        heads: 注意力头数量
        concat: True->拼接所有头的输出 (维度变 heads*channels); False->取平均 (维度不变)
        """
        # GAT 标准使用 'add' 聚合，因为注意力权重已经在 message 中乘过了
        super().__init__(aggr='add')

        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        # 每个头的输出维度
        self.out_channels_per_head = channels
        # 总输出维度
        self.hidden_channels = channels * heads

        # 1. 线性变换矩阵 W (源节点和目标节点共享或分开均可，这里分开以符合通用实现)
        # 输出形状: [N, heads * channels]
        self.lin_src = nn.Linear(channels, self.hidden_channels, bias=False)
        self.lin_dst = nn.Linear(channels, self.hidden_channels, bias=False)

        # 2. 注意力向量 a
        # 形状: [1, heads, out_channels_per_head]
        # 用于计算 e_ij = LeakyReLU( a^T [W*h_i || W*h_j] )
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.out_channels_per_head))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.out_channels_per_head))

        self.leaky_relu = nn.LeakyReLU(self.negative_slope)
        self.drop = nn.Dropout(dropout)

        if concat:
            self.out_proj = nn.Linear(self.hidden_channels, channels)
        else:
            self.out_proj = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index):
        """
        x: [Total_Nodes, Channels]
        edge_index: [2, Num_Edges]
        """
        N = x.size(0)

        # 1. 线性投影并重塑为多头格式 [N, heads*head_dim]
        x_src = self.lin_src(x)
        x_dst = self.lin_dst(x)

        # 2. 添加自环 (Self-Loops)
        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=N)

        # 3. 消息传播
        # propagate 会调用 message() 和 aggregate()
        # x_dst 对应目标节点 i (在 message 中作为 x_i)
        # x_src 对应源节点 j (在 message 中作为 x_j)
        out = self.propagate(edge_index_with_loop, x_src=x_src, x_dst=x_dst, size=(N, N))

        # 4. 输出处理
        if self.concat:
            out = self.out_proj(out)
        else:
            out = out.view(N, self.heads, self.out_channels_per_head).mean(dim=1)

        return out

    def message(self, x_src_j, x_dst_i, index):

        E = x_src_j.size(0)

        x_src_j = x_src_j.view(E, self.heads, self.out_channels_per_head)
        x_dst_i = x_dst_i.view(E, self.heads, self.out_channels_per_head)

        alpha_src = (x_src_j * self.att_src).sum(dim=-1, keepdim=True)
        alpha_dst = (x_dst_i * self.att_dst).sum(dim=-1, keepdim=True)

        alpha = alpha_src + alpha_dst
        alpha = self.leaky_relu(alpha)

        alpha = softmax(alpha, index)
        alpha = self.drop(alpha)

        out = x_src_j * alpha

        return out.view(E, self.hidden_channels)


# ==========================================
# 4. 主模型架构
# ==========================================
class ECG_VQ_Graph(nn.Module):
    def __init__(self, input_channels=12, seq_len=2500, hidden_dim=256, codebook_size=256):
        super().__init__()

        base_channels = 64
        input_channels = 12
        out_channels = 12

        # --- 编码器分支 1: 全局形态 (大尺度卷积) ---
        # 策略：大核，大步长，捕捉长程波形形状
        self.enc_morph = nn.Sequential(
            ResCNNBlock(input_channels, base_channels, kernel_size=15, stride=1, padding=7),  # 2500 -> 1250
            nn.AvgPool1d(kernel_size=2, stride=2),
            ResCNNBlock(base_channels, base_channels*2, kernel_size=15, stride=1, padding=7),  # 1250 -> 625
            nn.AvgPool1d(kernel_size=2, stride=2),
            ResCNNBlock(base_channels*2, base_channels*4, kernel_size=15, stride=1, padding=7),
            ResCNNBlock(base_channels*4, base_channels*4, kernel_size=15, stride=1, padding=7)
        )

        # --- 编码器分支 2: 局部节律 (小尺度卷积) ---
        # 策略：小核，捕捉高频细节和局部节律变化
        self.enc_rhythm = nn.Sequential(
            ResCNNBlock(input_channels, base_channels, kernel_size=5, stride=1, padding=2),  # 2500 -> 1250
            nn.AvgPool1d(kernel_size=2, stride=2),
            ResCNNBlock(base_channels, base_channels*2, kernel_size=5, stride=1, padding=2), # 1250 -> 625
            nn.AvgPool1d(kernel_size=2, stride=2),
            ResCNNBlock(base_channels*2, base_channels*4, kernel_size=5, stride=1, padding=2),
            ResCNNBlock(base_channels*4, base_channels*4, kernel_size=5, stride=1, padding=2)
        )

        # --- 码本量化 ---
        self.vq_morph = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=hidden_dim)
        self.vq_rhythm = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=hidden_dim)

        # --- 异构图交互模块 ---
        # 输入节点特征维度是 256
        self.graph_interact = HeteroGATInteraction(channels=hidden_dim)

        # --- 输出头 1: 房颤分类器 (基于增强后的形态特征) ---
        # 输入: [Batch, 625, 256] -> Pool -> FC
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)  # 二分类：AF vs Non-AF
        )

        # --- 输出头 2: 导联重构解码器 (基于增强后的节律特征) ---
        # 对称结构：上采样
        self.dec1 = ResCNNBlock(base_channels * 4, base_channels * 2, kernel_size=15, stride=1, padding=7)
        # Stage 3: 1250 -> 2500
        self.dec2 = ResCNNBlock(base_channels * 2, base_channels * 1, kernel_size=15, stride=1, padding=7)
        # Final refinement
        self.final = nn.Sequential(
            ResCNNBlock(base_channels * 1 + 12, base_channels, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(base_channels, out_channels, kernel_size=1)
        )


    def build_hetero_graph(self,feat_morph, feat_rhythm):
        """
        向量化版本：高效构建异构图
        包含:
        1. 模态内部时序连接 (t <-> t+1)
        2. 跨模态错位连接 (Mt <-> Rt-1, Mt <-> Rt+1)
        3. 跨模态同步连接 (Mt <-> Rt)
        """
        B, T, C = feat_morph.shape
        device = feat_morph.device

        # 1. 准备节点特征和 Batch 向量
        x_all = torch.cat([feat_morph.reshape(-1, C), feat_rhythm.reshape(-1, C)], dim=0)
        batch_vec = torch.repeat_interleave(torch.arange(B, device=device), 2 * T)

        edge_list = []

        # 辅助：生成基址偏移量
        # morph_base: [0, T, 2T, ...] (形状 [B, 1])
        morph_base = (torch.arange(B, device=device) * T).view(-1, 1)
        # rhythm_base: [B*T, B*T+T, ...] (形状 [B, 1])
        rhythm_base = morph_base + B * T

        # 时间步模板 [0, 1, ..., T-2] (用于 t -> t+1)
        t_idx_next = torch.arange(T - 1, device=device).view(1, -1)

        # --- 2. 构建边 (全部向量化) ---

        # A. 形态内部边 (M_t <-> M_{t+1})
        src_m = (morph_base + t_idx_next).flatten()
        dst_m = (morph_base + t_idx_next + 1).flatten()
        edge_list.append(torch.stack([src_m, dst_m], dim=0))
        edge_list.append(torch.stack([dst_m, src_m], dim=0))

        # B. 节律内部边 (R_t <-> R_{t+1})
        src_r = (rhythm_base + t_idx_next).flatten()
        dst_r = (rhythm_base + t_idx_next + 1).flatten()
        edge_list.append(torch.stack([src_r, dst_r], dim=0))
        edge_list.append(torch.stack([dst_r, src_r], dim=0))

        # C. 跨模态错位连接 (Mt <-> Rt-1 和 Mt <-> Rt+1)

        # C1. Mt <-> R_{t-1} (t 从 1 到 T-1)
        if T > 1:
            t_cross_prev = torch.arange(1, T, device=device).view(1, -1)
            m_node_prev = (morph_base + t_cross_prev).flatten()
            r_node_prev = (rhythm_base + t_cross_prev - 1).flatten()
            edge_list.append(torch.stack([m_node_prev, r_node_prev], dim=0))
            edge_list.append(torch.stack([r_node_prev, m_node_prev], dim=0))

        # C2. Mt <-> R_{t+1} (t 从 0 到 T-2)
        if T > 1:
            t_cross_next = torch.arange(T - 1, device=device).view(1, -1)
            m_node_next = (morph_base + t_cross_next).flatten()
            r_node_next = (rhythm_base + t_cross_next + 1).flatten()
            edge_list.append(torch.stack([m_node_next, r_node_next], dim=0))
            edge_list.append(torch.stack([r_node_next, m_node_next], dim=0))

        # D. 跨模态同步连接 (Mt <-> Rt)
        # 逻辑：所有 t (0 到 T-1) 都连接
        t_sync = torch.arange(T, device=device).view(1, -1)  # [0, 1, ..., T-1]

        m_node_sync = (morph_base + t_sync).flatten()  # 所有 M_t
        r_node_sync = (rhythm_base + t_sync).flatten()  # 所有 R_t

        # 添加双向边
        edge_list.append(torch.stack([m_node_sync, r_node_sync], dim=0))
        edge_list.append(torch.stack([r_node_sync, m_node_sync], dim=0))

        # --- 3. 合并与输出 ---
        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.cat(edge_list, dim=1)
            # 理论上向量化逻辑不会产生重复，除非 T=1 时某些边界条件重叠，
            # 但 T=1 时 C1/C2 不会执行，只有 D 执行，所以也是安全的。
            # 如果为了绝对保险，可以取消下面这行的注释，但会略微降低速度：
            # edge_index = torch.unique(edge_index, dim=1, sorted=False)

        return Data(x=x_all, edge_index=edge_index, batch=batch_vec)

    def forward(self, x, x_aug):
        """
        x: [B, 12, 2500]
        """
        B = x.size(0)

        # 1. 编码
        # 输出形状: [B, 256, 625] -> 转置为 [B, 625, 256] 方便后续处理
        feat_morph_cont = self.enc_morph(x).transpose(1, 2)
        feat_rhythm_cont = self.enc_rhythm(x).transpose(1, 2)
        
        feat_rhythm_cont_aug = self.enc_rhythm(x_aug).transpose(1, 2)

        # 2. 量化
        z_morph, loss_vq_m, perp_m, indice_m = self.vq_morph(feat_morph_cont)
        z_rhythm, loss_vq_r, perp_r, indice_r = self.vq_rhythm(feat_rhythm_cont)

        # z_morph = feat_morph_cont + z_morph
        # z_rhythm = feat_rhythm_cont + z_rhythm

        # 3. 构建异构图并交互
        graph_data = self.build_hetero_graph(z_morph, z_rhythm)

        # 运行 GNN
        h_all = self.graph_interact(graph_data.x, graph_data.edge_index)

        # 分离回 Morph 和 Rhythm
        total_seq_len = z_morph.shape[1]
        h_morph_enhanced = h_all[:B * total_seq_len].reshape(B, total_seq_len, -1)
        h_rhythm_enhanced = h_all[B * total_seq_len:].reshape(B, total_seq_len, -1)

        # h_morph_enhanced = feat_morph_cont
        # h_rhythm_enhanced = feat_rhythm_cont

        # 4. 输出分支 1: 房颤检测 (基于节律)
        rhythm_pool = h_rhythm_enhanced.mean(dim=1)
        logits_af = self.classifier(rhythm_pool)  # [B, 2]

        # 5. 输出分支 2: 导联重构 (基于形态)
        # 转置回 [B, C, T]
        rec_input = h_morph_enhanced.transpose(1, 2)
        # Decoder
        # Upsample to 1250
        d1 = F.interpolate(rec_input, size=rec_input.shape[-1]*2, mode='linear', align_corners=False)  # [B, 256, 1250]
        d1 = self.dec1(d1)               # [B, 128, 1250]

        # Upsample to 2500
        d2 = F.interpolate(d1, size=d1.shape[-1]*2, mode='linear', align_corners=False)  # [B, 128, 2500]
        d2 = self.dec2(d2)               # [B, 64, 2500]

        x_recon = self.final(torch.cat([x, d2], dim=1))             # [B, 12, 2500]

        return {
            'logits_af': logits_af,
            'x_recon': x_recon,
            'loss_vq': loss_vq_m + loss_vq_r,
            'perplexity': (perp_m + perp_r) / 2,
            'feat_rhythm_cont': feat_rhythm_cont,
            'feat_rhythm_cont_aug': feat_rhythm_cont_aug,
            'indice_m': indice_m,
            'indice_r': indice_r
        }


# ==========================================
# 5. 损失函数计算
# ==========================================
def calculate_loss(outputs, targets_af, x_original,
                   lambda_cls=1.0,
                   lambda_rec=1.0,
                   lambda_local_con=0.5,
                   lambda_global_con=0.5,
                   temperature=0.1,
                   lambda_vq=0.1):
    """
    targets_af: [B] (0 or 1)
    x_original: [B, 12, 2500]
    """

    # 1. 房颤分类损失 (CrossEntropy)
    loss_cls = F.cross_entropy(outputs['logits_af'], targets_af)

    # 2. 重构损失 (MSE)
    loss_rec = F.l1_loss(outputs['x_recon'], x_original)

    # 3. 对比损失
    feat1 = outputs['feat_rhythm_cont']      # [B, T, C]
    feat2 = outputs['feat_rhythm_cont_aug']  # [B, T, C]

    B, T, C = feat1.shape

    # --- 全局对比损失 ---
    global1 = feat1.mean(dim=1)  # [B, C]
    global2 = feat2.mean(dim=1)  # [B, C]
    loss_global_con = info_nce_loss(global1, global2, temperature=temperature)

    # --- 局部对比损失 ---
    # reshape to [B*T, C]
    local1 = feat1.reshape(B * T, C)
    local2 = feat2.reshape(B * T, C)
    loss_local_con = info_nce_loss(local1, local2, temperature=temperature)


    # 4. VQ-VAE 损失
    loss_vq = outputs['loss_vq']

    total_loss = (
        lambda_cls * loss_cls +
        lambda_rec * loss_rec +
        lambda_vq * loss_vq +
        lambda_global_con * loss_global_con +
        lambda_local_con * loss_local_con
    )

    return total_loss, {
        'cls': loss_cls.item(),
        'rec': loss_rec.item(),
        'con_local': loss_local_con.item(),
        'con_global': loss_global_con.item(),
        'vq': loss_vq.item()
    }

def info_nce_loss(z1, z2, temperature=0.1):
    """
    标准对称 InfoNCE 损失（SimCLR 风格）
    z1, z2: [N, D] —— 正样本对 (z1[i] 与 z2[i] 是一对)
    """
    # 1. L2 归一化（使点积 = 余弦相似度）
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    N = z1.size(0)
    
    # 2. 拼接两个视图 → [2N, D]
    z = torch.cat([z1, z2], dim=0)  # [a, b, a', b']
    
    # 3. 计算全连接相似度矩阵 [2N, 2N]
    sim = torch.mm(z, z.t()) / temperature
    
    # 4. 创建标签：每个样本的正样本在另一半
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(z.device)
    # labels = [2, 3, 0, 1] when N=2
    
    # 5. 关键：只 mask 掉 self-similarity（对角线）
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)  # 防止自己和自己匹配
    
    # 6. 计算 InfoNCE loss
    loss = F.cross_entropy(sim, labels)
    return loss
