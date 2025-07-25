# chemprop/models/dgin.py

import torch
from typing import Iterable

from chemprop.data import BatchMolGraph
from chemprop.nn import Aggregation, ChempropMetric, Predictor
from chemprop.nn.message_passing import MessagePassing
from chemprop.nn.gin import GINEncoder
from chemprop.models.model import MPNN
from torch import Tensor, nn

class DGIN(MPNN):
    """
    Mô hình D-GIN (Directed-Edge Graph Isomorphism Network).
    """
    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        gin_hidden_dim: int,
        gin_n_layers: int,
        **kwargs
    ):
        # Gọi super().__init__ của lớp cha MPNN và truyền ĐẦY ĐỦ các tham số.
        super().__init__(message_passing=message_passing, agg=agg, predictor=predictor, **kwargs)

        # --- PHẦN KHỞI TẠO RIÊNG CỦA KIẾN TRÚC D-GIN ---

        atom_features_dim = self.hparams.message_passing["d_v"]
        
        self.gin_encoder = GINEncoder(
            in_dim=atom_features_dim,
            hidden_dim=gin_hidden_dim,
            n_layers=gin_n_layers
        )

        dmpnn_output_dim = self.message_passing.output_dim
        concatenated_dim = dmpnn_output_dim + gin_hidden_dim

        use_batch_norm = self.hparams.batch_norm
        self.bn = nn.BatchNorm1d(concatenated_dim) if use_batch_norm else nn.Identity()

        predictor_hparams = self.hparams.predictor
        predictor_hparams["input_dim"] = concatenated_dim
        self.predictor = Predictor.from_hparams(predictor_hparams)

        self.hparams.gin_hidden_dim = gin_hidden_dim
        self.hparams.gin_n_layers = gin_n_layers
        self.hparams.predictor = self.predictor.hparams

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """
        Ghi đè phương thức fingerprint để tính toán đặc trưng lai D-GIN.
        """
        dmpnn_atom_features = self.message_passing(bmg, V_d)
        gin_atom_features = self.gin_encoder(bmg)

        concatenated_atom_features = torch.cat([dmpnn_atom_features, gin_atom_features], dim=1)

        H = self.agg(concatenated_atom_features, bmg.batch)
        H = self.bn(H)
        
        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    # ==============================================================================
    # == PHẦN THÊM VÀO ĐỂ SỬA LỖI TRIỆT ĐỂ ==
    # ==============================================================================
    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """
        Định nghĩa TƯỜNG MINH phương thức forward cho DGIN.
        
        Hàm này đảm bảo rằng DGIN sẽ luôn chấp nhận đúng số lượng tham số
        (self, bmg, V_d, X_d) mà trainer truyền vào, loại bỏ mọi sự cố về kế thừa.
        
        Logic của nó đơn giản là gọi predictor trên kết quả của hàm fingerprint mới của chúng ta.
        """
        return self.predictor(self.fingerprint(bmg, V_d, X_d))