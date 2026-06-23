#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 E001 三头模型: CoverageHead (共享) + ReliabilityHead + NaiveHead

架构:
  CoverageHead:  per-frame 2D CNN [K,1,H,W] → [K,128] → AttentionPool → scalar Ĉ
  ReliabilityHead: MLP(3→32→1) 输入=[depth_consistency, pose_drift, obs_density]
  NaiveHead:     MLP(4→32→1) 输入=[predicted_var, frame_count, obs_density_proxy, optical_flow_mag]

Coverage head 两个臂共享 (byte-identical), 只换弃权排序信号。
可靠性头输入仅来自退化重建本身 (禁止注入参数/引擎侧量)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerFrameCNN(nn.Module):
    """单帧深度图 → 128 维 embedding。输入 [B,1,256,256]。"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        """x: [B,1,256,256] → [B, embed_dim]"""
        return self.fc(self.net(x).flatten(1))


class AttentionPool(nn.Module):
    """K 帧_embedding → 标量。可学习 query attention。"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, frame_embeds, validity_mask=None):
        """
        frame_embeds: [B, K, D]
        validity_mask: [B, K] bool, True=有效帧 (可选)
        返回: [B, 1] 标量覆盖度预测
        """
        B, K, D = frame_embeds.shape
        q = self.query.expand(B, -1, -1)  # [B,1,D]
        # key_padding_mask: True=忽略 (与 PyTorch 约定一致)
        key_mask = None
        if validity_mask is not None:
            key_mask = ~validity_mask  # [B,K], True=padding
        out, _ = self.attn(q, frame_embeds, frame_embeds, key_padding_mask=key_mask)
        return self.fc(out.squeeze(1)).squeeze(-1)  # [B,D]→[B,1]→[B]


class CoverageHead(nn.Module):
    """共享覆盖度头: per-frame CNN → attention pool → scalar Ĉ"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.frame_cnn = PerFrameCNN(embed_dim)
        self.pool = AttentionPool(embed_dim)

    def forward(self, depth_seq, validity_mask=None):
        """分块处理帧避免 OOM: 每次 CHUNK 帧送进 CNN"""
        B, K = depth_seq.shape[:2]
        CHUNK = 64
        embeds_list = []
        for start in range(0, K, CHUNK):
            end = min(start + CHUNK, K)
            chunk = depth_seq[:, start:end]
            flat = chunk.reshape(B * (end - start), *chunk.shape[2:])
            emb = self.frame_cnn(flat)
            embeds_list.append(emb.reshape(B, end - start, -1))
        embeds = torch.cat(embeds_list, dim=1)
        pred = self.pool(embeds, validity_mask)
        return pred


class ReliabilityHead(nn.Module):
    """可靠性头: MLP(3→32→1), 输入仅重建内部量"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features):
        """
        features: [B, 3] = [depth_consistency, pose_drift, obs_density]
        返回: [B] 可靠性分数 (越高=越可靠)
        """
        return self.net(features).squeeze(-1)


class NaiveHead(nn.Module):
    """朴素弃权头: MLP(4→32→1), heteroscedastic NLL predicted variance + 外观代理"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features):
        """
        features: [B, 4] = [predicted_var, frame_count, obs_density_proxy, optical_flow_mag]
        返回: [B] 朴素置信度 (越高=越可信)
        """
        return self.net(features).squeeze(-1)


class P11Model(nn.Module):
    """完整三头模型"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.coverage_head = CoverageHead(embed_dim)
        self.reliability_head = ReliabilityHead()
        self.naive_head = NaiveHead()

    def forward(self, depth_seq, validity_mask=None,
                rel_features=None, naive_features=None):
        """
        depth_seq: [B, K, 1, H, W]
        validity_mask: [B, K] (可选)
        rel_features: [B, 3] 可靠性输入 (可选, eval 时用)
        naive_features: [B, 4] 朴素输入 (可选, eval 时用)
        返回: dict with 'coverage_pred', 'reliability_score' (if rel_features), 'naive_score' (if naive_features)
        """
        cov_pred = self.coverage_head(depth_seq, validity_mask)
        out = {'coverage_pred': cov_pred}
        if rel_features is not None:
            out['reliability_score'] = self.reliability_head(rel_features)
        if naive_features is not None:
            out['naive_score'] = self.naive_head(naive_features)
        return out
