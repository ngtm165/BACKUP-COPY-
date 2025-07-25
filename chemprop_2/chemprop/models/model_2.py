"""
reaction_mpnn.py
-----------------------------------------------------------
PyG >= 2.5   •  torch >= 2.2  •  RDKit only for data prep (không dùng ở đây)

File này kết hợp:
1.  Kiến trúc GNN phân cấp cho phản ứng hóa học (từ mã gốc của bạn).
2.  Một lớp LightningModule (`ReactionMPNN`) để tích hợp GNN vào một
    quy trình huấn luyện tiêu chuẩn, tương tự như `chemprop`.
3.  Một ví dụ về cách tải dữ liệu giả lập và huấn luyện mô hình.
-----------------------------------------------------------
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, GINConv
from torch_geometric.utils import to_undirected, add_self_loops, k_hop_subgraph

import lightning as pl
from torch import optim, Tensor

# =============================================================================
# PHẦN 1: KIẾN TRÚC GNN PHẢN ỨNG (Lấy từ mã gốc của bạn)
# =============================================================================

# ---------- 0. Misc helpers --------------------------------------------------

def add_k_jump_edges(edge_index, num_nodes, k: int = 3):
    """
    Thêm các cạnh k-jump để giảm đường kính đồ thị.
    """
    if k <= 1:
        return edge_index
    
    new_edges = []
    for hop in range(2, k + 1):
        _, _, n_hop, _ = k_hop_subgraph(
            None, hop, edge_index, num_nodes=num_nodes, relabel_nodes=False)
        new_edges.append(n_hop)
    if len(new_edges):
        edge_index = torch.cat([edge_index] + new_edges, dim=1)
    return to_undirected(edge_index)

# ---------- 1. Two-layer Atom-level DMPNN ------------------------------------

class DMPNNLayer(MessagePassing):
    def __init__(self, in_dim, edge_dim):
        super().__init__(aggr='add')
        self.W_msg = nn.Linear(in_dim + edge_dim, in_dim, bias=False)
        self.gru   = nn.GRUCell(in_dim, in_dim)

    def forward(self, x, edge_index, edge_attr):
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.gru(m, x)
        return x

    def message(self, x_j, edge_attr):
        return self.W_msg(torch.cat([x_j, edge_attr], dim=-1))

class AtomDMPNN(nn.Module):
    """Level-0/1 local encoder: 2 directed-edge DMPNN layers."""
    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.gnn1 = DMPNNLayer(hidden, edge_dim)
        self.gnn2 = DMPNNLayer(hidden, edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.gnn1(x, edge_index, edge_attr)
        x = self.gnn2(x, edge_index, edge_attr)
        return x

# ---------- 2. MixHop (single block)  ----------------------------------------

class MixHopConv(nn.Module):
    """
    Parallel 1-,2-,3-hop GINs whose outputs are concatenated.
    """
    def __init__(self, hidden, hops=(1, 2, 3)):
        super().__init__()
        self.hops = hops
        self.convs = nn.ModuleList([
            GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
            for _ in hops
        ])
        self.out_proj = nn.Linear(hidden * len(hops), hidden)

    def forward(self, x, edge_index):
        outs = []
        for hop, conv in zip(self.hops, self.convs):
            h = x
            for _ in range(hop):
                h = conv(h, edge_index)
            outs.append(h)
        x = self.out_proj(torch.cat(outs, dim=-1))
        return x

# ---------- 3. Gated Skip-edge block  ----------------------------------------

class GatedSkipBlock(nn.Module):
    """
    One hop, gated spectators (non-RC atom -> supernode S).
    """
    def __init__(self, hidden):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.W    = nn.Linear(hidden, hidden, bias=False)
        self.gru  = nn.GRUCell(hidden, hidden)

    def forward(self, h, rc_mask, idx_S):
        N = rc_mask.size(0)
        idx_atoms = torch.arange(N, device=h.device)
        non_rc    = idx_atoms[~rc_mask]

        alpha   = torch.sigmoid(self.gate(h[non_rc]))
        msgs    = alpha * self.W(h[non_rc])
        m_sum   = msgs.sum(0, keepdim=True)

        m_rc    = self.W(h[-2:-1])
        m_total = m_sum + m_rc

        h_S_new = self.gru(m_total, h[idx_S:idx_S+1])
        h[idx_S] = h_S_new
        return h

# ---------- 4. Whole Reaction Encoder  ---------------------------------------

class ReactionEncoder(nn.Module):
    def __init__(self, hidden, edge_dim, k_jump=3):
        super().__init__()
        self.hidden = hidden
        self.k_jump = k_jump
        self.atom_dmpnn = AtomDMPNN(hidden, edge_dim)
        self.mixhop = MixHopConv(hidden)
        self.att_q = nn.Linear(hidden, hidden)
        self.att_k = nn.Linear(hidden, hidden)
        self.att_v = nn.Linear(hidden, hidden)
        self.out_linear = nn.Linear(hidden, hidden)
        self.skip_block = GatedSkipBlock(hidden)
        self.rc_init = nn.Parameter(torch.zeros(1, hidden))
        self.s_init  = nn.Parameter(torch.zeros(1, hidden))
        nn.init.xavier_normal_(self.rc_init)
        nn.init.xavier_normal_(self.s_init)

    def _augment_graph(self, data):
        N_atoms = data.x.size(0)
        edge_index = add_self_loops(data.edge_index, num_nodes=N_atoms)[0]
        edge_index = add_k_jump_edges(edge_index, N_atoms, k=self.k_jump)
        rc_idx = N_atoms
        s_idx  = N_atoms + 1
        rc_atoms = data.rc_mask.nonzero(as_tuple=True)[0]
        rc_edges = torch.stack([torch.cat([rc_atoms, rc_atoms.clone().fill_(rc_idx)]),
                                torch.cat([rc_atoms.clone().fill_(rc_idx), rc_atoms])], dim=0)
        rc_s = torch.tensor([[rc_idx, s_idx], [s_idx, rc_idx]], dtype=torch.long, device=edge_index.device)
        non_rc = (~data.rc_mask).nonzero(as_tuple=True)[0]
        skip_e = torch.stack([torch.cat([non_rc, non_rc.clone().fill_(s_idx)]),
                              torch.cat([non_rc.clone().fill_(s_idx), non_rc])], dim=0)
        edge_index = torch.cat([edge_index, rc_edges, rc_s, skip_e], dim=1)
        edge_index = to_undirected(edge_index)
        return edge_index, rc_idx, s_idx

    def forward(self, data):
        h_atoms_dmpnn = self.atom_dmpnn(data.x, data.edge_index, data.edge_attr)
        h_atoms_mixhop = self.mixhop(h_atoms_dmpnn, data.edge_index)
        
        edge_index, rc_idx, s_idx = self._augment_graph(data)
        h = torch.cat([h_atoms_mixhop, self.rc_init, self.s_init], dim=0)
        h = self.skip_block(h, data.rc_mask, s_idx)
        
        q = self.att_q(h)
        k = self.att_k(h)
        v = self.att_v(h)
        attn = (q @ k.T) / (self.hidden ** 0.5)
        attn = attn.softmax(dim=-1)
        h_attn = h + self.out_linear(attn @ v)
        
        # Mở rộng h_atoms để phù hợp với kích thước của h_attn cho JK
        h_atoms_padded = torch.nn.functional.pad(h_atoms_mixhop, (0, 0, 0, 2))

        h_nodes = torch.stack([h_atoms_padded, h_attn], dim=0)
        h_jk = h_nodes.max(0).values
        
        h_S = h_jk[s_idx]
        return h_S

# ---------- 5. Property head & full model -------------------------------

class ReactionPropertyModel(nn.Module):
    def __init__(self, hidden, edge_dim, k_jump=3, out_dim=1):
        super().__init__()
        self.encoder = ReactionEncoder(hidden, edge_dim, k_jump)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, data):
        h_S = self.encoder(data)
        return self.mlp(h_S)

# =============================================================================
# PHẦN 2: TÍCH HỢP PYTORCH LIGHTNING (THAY THẾ LỚP MPNN CŨ)
# =============================================================================

class ReactionMPNN(pl.LightningModule):
    """
    Một lớp LightningModule để huấn luyện mô hình ReactionPropertyModel.
    Lớp này thay thế lớp MPNN gốc của chemprop, sử dụng kiến trúc GNN
    mới và làm việc với dữ liệu từ PyTorch Geometric.
    """
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        k_jump: int = 3,
        out_dim: int = 1,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        loss_fn: str = "mse",
    ):
        super().__init__()
        # Lưu các siêu tham số để dễ dàng checkpoint và tải lại
        self.save_hyperparameters()

        self.model = ReactionPropertyModel(
            hidden=hidden_dim,
            edge_dim=edge_dim,
            k_jump=k_jump,
            out_dim=out_dim
        )
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Loss function '{loss_fn}' không được hỗ trợ.")

    def forward(self, data: Data | DataLoader.Collater) -> Tensor:
        """
        Thực hiện một lượt dự đoán.
        Input `data` là một batch từ PyG DataLoader.
        """
        return self.model(data)

    def _shared_step(self, batch, batch_idx):
        """Hàm dùng chung cho training, validation, và test."""
        # `batch` ở đây là một đối tượng `torch_geometric.data.Batch`
        # Nhãn (target) được lưu trong `batch.y`
        targets = batch.y
        
        # Dự đoán
        preds = self(batch).squeeze(-1)
        
        # Tính loss
        loss = self.criterion(preds, targets)
        
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        # `prog_bar=True` để hiển thị loss trên thanh tiến trình
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        """
        Thiết lập optimizer và learning rate scheduler.
        Sử dụng lại cấu hình từ mã gốc của chemprop.
        """
        opt = optim.Adam(self.parameters(), self.hparams.init_lr)

        # Cần ước tính số bước cho scheduler
        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches

        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.hparams.warmup_epochs * steps_per_epoch
        
        if self.trainer.max_epochs == -1:
            # Huấn luyện vô hạn
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.hparams.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.max_lr,
            total_steps=warmup_steps + cooldown_steps,
            pct_start=warmup_steps / (warmup_steps + cooldown_steps) if (warmup_steps + cooldown_steps) > 0 else 0,
            div_factor=self.hparams.max_lr / self.hparams.init_lr,
            final_div_factor=self.hparams.init_lr / self.hparams.final_lr,
            anneal_strategy='linear'
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}