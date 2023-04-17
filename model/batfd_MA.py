from typing import Dict, Optional, Union, Sequence, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.lavdf import Metadata
from loss import MaskedFrameLoss, MaskedBMLoss, MaskedContrastLoss
from .audio_encoder import CNNAudioEncoder
from .boundary_module import BoundaryModule
from .frame_classifier import FrameLogisticRegression
from .fusion_module import ModalFeatureAttnBoundaryMapFusion
from .video_encoder import C3DVideoEncoder
from .MA import *

class BATFD(LightningModule):

    def __init__(self,
        v_encoder: str = "c3d", a_encoder: str = "cnn", frame_classifier: str = "lr",
        ve_features=(64, 96, 128, 128), ae_features=(32, 64, 64), v_cla_feature_in=256, a_cla_feature_in=256,
        boundary_features=(512, 128), boundary_samples=10, temporal_dim=512, max_duration=40,
        weight_frame_loss=2., weight_modal_bm_loss=1., weight_contrastive_loss=0.1, contrast_loss_margin=0.99,
        weight_decay=0.0001, learning_rate=0.0002, distributed=False
    ):
        super().__init__()
        self.save_hyperparameters()

        if v_encoder == "c3d":
            self.video_encoder = C3DVideoEncoder(n_features=ve_features)

        if a_encoder == "cnn":
            self.audio_encoder = CNNAudioEncoder(n_features=ae_features)
        self.attn = Encoder(HANLayer(d_model=256, nhead=1, dim_feedforward=256), num_layers=1)

        if frame_classifier == "lr":
            self.video_frame_classifier = FrameLogisticRegression(n_features=v_cla_feature_in)
            self.audio_frame_classifier = FrameLogisticRegression(n_features=a_cla_feature_in)

        assert self.video_encoder and self.audio_encoder and self.video_frame_classifier and self.audio_frame_classifier

        assert v_cla_feature_in == a_cla_feature_in

        v_bm_in = v_cla_feature_in + 1
        a_bm_in = a_cla_feature_in + 1

        self.video_boundary_module = BoundaryModule(v_bm_in, boundary_features, boundary_samples, temporal_dim,
            max_duration
        )
        self.audio_boundary_module = BoundaryModule(a_bm_in, boundary_features, boundary_samples, temporal_dim,
            max_duration
        )

        self.fusion = ModalFeatureAttnBoundaryMapFusion(v_bm_in, a_bm_in, max_duration)

        self.frame_loss = MaskedFrameLoss()#focal loss
        self.contrast_loss = MaskedContrastLoss(margin=contrast_loss_margin)
        self.bm_loss = MaskedBMLoss(MSELoss())
        self.weight_frame_loss = weight_frame_loss
        self.weight_modal_bm_loss = weight_modal_bm_loss
        self.weight_contrastive_loss = weight_contrastive_loss / (v_cla_feature_in * temporal_dim)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.distributed = distributed

    def forward(self, video: Tensor, audio: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        v_features = self.video_encoder(video)
        a_features = self.audio_encoder(audio)#[1,256,512]
        
        #Mix-Attention
        v_features = v_features.permute(0,2,1)
        a_features = a_features.permute(0,2,1)
        v_features_fra_in, a_features_fra_in = self.attn(v_features, a_features)
        v_features_fra_in = v_features_fra_in.permute(0,2,1)
        a_features_fra_in = a_features_fra_in.permute(0,2,1)
        v_features = v_features.permute(0,2,1)
        a_features = a_features.permute(0,2,1)

        # frame classifiers
        v_frame_cla = self.video_frame_classifier(v_features_fra_in)
        a_frame_cla = self.audio_frame_classifier(a_features_fra_in)

        v_bm_in = torch.column_stack([v_features_fra_in, v_frame_cla])
        a_bm_in = torch.column_stack([v_features_fra_in, a_frame_cla])

        # modal boundary module
        v_bm_map = self.video_boundary_module(v_bm_in)
        a_bm_map = self.audio_boundary_module(a_bm_in)

        # boundary map modal attention fusion
        fusion_bm_map = self.fusion(v_bm_in, a_bm_in, v_bm_map, a_bm_map)
        

        return fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, v_features_fra_in, a_features_fra_in

    def loss_fn(self, fusion_bm_map: Tensor, v_bm_map: Tensor, a_bm_map: Tensor,
        v_frame_cla: Tensor, a_frame_cla: Tensor, label: Tensor, n_frames: Tensor,
        v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label, v_features_fra_in, a_features_fra_in
    ) -> Dict[str, Tensor]:
        fusion_bm_loss = self.bm_loss(fusion_bm_map, label, n_frames)

        v_bm_loss = self.bm_loss(v_bm_map, v_bm_label, n_frames)
        a_bm_loss = self.bm_loss(a_bm_map, a_bm_label, n_frames)

        v_frame_loss = self.frame_loss(v_frame_cla.squeeze(1), v_frame_label, n_frames)
        a_frame_loss = self.frame_loss(a_frame_cla.squeeze(1), a_frame_label, n_frames)

        
        contrast_loss = torch.clip(self.contrast_loss(v_features_fra_in, a_features_fra_in, contrast_label, n_frames)
                                   / (256 * 512), max=1.)

        loss = fusion_bm_loss + \
               self.weight_modal_bm_loss * (a_bm_loss + v_bm_loss) / 2 + \
               self.weight_frame_loss * (a_frame_loss + v_frame_loss) / 2 + \
               self.weight_contrastive_loss * contrast_loss

        return {
            "loss": loss, "fusion_bm_loss": fusion_bm_loss, "v_bm_loss": v_bm_loss, "a_bm_loss": a_bm_loss,
            "v_frame_loss": v_frame_loss, "a_frame_loss": a_frame_loss, "contrast_loss": contrast_loss
        }

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Tensor:
        video, audio, label, n_frames, v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label = batch

        fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, v_features_fra_in, a_features_fra_in = self(video, audio)
        loss_dict = self.loss_fn(fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, label, n_frames,
            v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label, v_features_fra_in, a_features_fra_in
        )
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Tensor:
        video, audio, label, n_frames, v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label = batch

        fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, v_features_fra_in, a_features_fra_in = self(video, audio)
        loss_dict = self.loss_fn(fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, label, n_frames,
            v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label, v_features_fra_in, a_features_fra_in
        )

        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
        video, audio, label, n_frames, v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label = batch
        fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, v_features_fra_in, a_features_fra_in = self(video, audio)
        return fusion_bm_map

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }

    @staticmethod
    def get_meta_attr(meta: Metadata, video: Tensor, audio: Tensor, label: Tensor):
        label_fake = label
        label_real = torch.zeros(label.size(), dtype=label.dtype, device=label.device)

        v_bm_label = label_fake if meta.modify_video else label_real
        a_bm_label = label_fake if meta.modify_audio else label_real

        frame_label_real = torch.zeros(512)
        frame_label_fake = torch.zeros(512)
        for begin, end in meta.fake_periods:
            begin = int(begin * 25)
            end = int(end * 25)
            frame_label_fake[begin: end] = 1

        v_frame_label = frame_label_fake if meta.modify_video else frame_label_real
        a_frame_label = frame_label_fake if meta.modify_audio else frame_label_real

        contrast_label = 0 if meta.modify_audio or meta.modify_video else 1

        return [meta.video_frames, v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label]
