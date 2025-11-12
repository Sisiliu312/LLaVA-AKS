import torch
import torch.nn as nn
import heapq
import numpy as np
from typing import List, Tuple, Dict


class AdaptiveImageTokenPruner(nn.Module):
    """
    Adaptive image token pruning using tree-based hierarchical splitting.
    Similar to video frame selection but operates on image tokens.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_branches: int = 2,  # 树杈数（可修改）
        max_depth: int = 5,
        importance_threshold: float = 0.8,
        std_threshold: float = -100,
        target_token_ratio: float = 0.5,  # 保留token的比例
    ):
        super().__init__()
        self.num_branches = num_branches
        self.max_depth = max_depth
        self.importance_threshold = importance_threshold
        self.std_threshold = std_threshold
        self.target_token_ratio = target_token_ratio
        
        # 可学习的importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def compute_token_importance(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        计算每个token的重要性分数
        Args:
            tokens: [num_tokens, hidden_size]
        Returns:
            scores: [num_tokens]
        """
        with torch.cuda.amp.autocast(enabled=False):
            scores = self.importance_scorer(tokens.float()).squeeze(-1)
        return scores
    
    def tree_split_tokens(
        self,
        token_indices: List[int],
        scores: torch.Tensor,
        depth: int = 0
    ) -> Tuple[List[Dict], List[List[int]]]:
        """
        递归地进行树状分割
        Args:
            token_indices: 当前片段的token索引列表
            scores: 所有token的重要性分数
            depth: 当前递归深度
        Returns:
            segments_info: 每个最终片段的信息（分数和深度）
            segments_indices: 每个最终片段的token索引
        """
        if len(token_indices) == 0:
            return [], []
        
        # 获取当前片段的分数
        segment_scores = scores[token_indices].cpu().numpy()
        
        # 计算统计量
        mean_score = np.mean(segment_scores)
        std_score = np.std(segment_scores)
        
        # 计算top-k分数的均值差异
        k = max(1, len(segment_scores) // self.num_branches)
        top_k_indices = heapq.nlargest(k, range(len(segment_scores)), segment_scores.__getitem__)
        top_k_scores = [segment_scores[i] for i in top_k_indices]
        mean_diff = np.mean(top_k_scores) - mean_score
        
        # 判断是否需要继续分割
        should_not_split = (
            mean_diff > self.importance_threshold and 
            std_score > self.std_threshold
        )
        
        reach_max_depth = depth >= self.max_depth
        too_small = len(token_indices) < self.num_branches * 2
        
        if should_not_split or reach_max_depth or too_small:
            # 不再分割，返回当前片段
            return [{'score': segment_scores, 'depth': depth}], [token_indices]
        
        # 进行树状分割
        branch_size = len(token_indices) // self.num_branches
        split_segments = []
        split_indices = []
        no_split_segments = []
        no_split_indices = []
        
        # 将当前片段分成 num_branches 个子片段
        for i in range(self.num_branches):
            start_idx = i * branch_size
            if i == self.num_branches - 1:
                # 最后一个分支包含所有剩余的tokens
                end_idx = len(token_indices)
            else:
                end_idx = (i + 1) * branch_size
            
            branch_indices = token_indices[start_idx:end_idx]
            
            if len(branch_indices) > 0:
                branch_scores = scores[branch_indices].cpu().numpy()
                
                # 递归处理每个分支
                sub_segments, sub_indices = self.tree_split_tokens(
                    branch_indices, scores, depth + 1
                )
                split_segments.extend(sub_segments)
                split_indices.extend(sub_indices)
        
        return split_segments, split_indices
    
    def select_tokens_from_segments(
        self,
        segments_info: List[Dict],
        segments_indices: List[List[int]],
        scores: torch.Tensor,
        target_num_tokens: int
    ) -> List[int]:
        """
        从各个片段中选择token
        Args:
            segments_info: 每个片段的信息
            segments_indices: 每个片段的token索引
            scores: 所有token的重要性分数
            target_num_tokens: 目标token数量
        Returns:
            selected_indices: 选中的token索引列表
        """
        selected_indices = []
        
        for segment_info, segment_idx in zip(segments_info, segments_indices):
            depth = segment_info['depth']
            # 根据深度分配token数量（深度越小，保留越多）
            segment_quota = int(target_num_tokens / (self.num_branches ** depth))
            segment_quota = max(1, min(segment_quota, len(segment_idx)))
            
            # 在该片段中选择top-k个token
            segment_scores = scores[segment_idx].cpu().numpy()
            top_k_local = heapq.nlargest(
                segment_quota, 
                range(len(segment_scores)), 
                segment_scores.__getitem__
            )
            selected_from_segment = [segment_idx[i] for i in top_k_local]
            selected_indices.extend(selected_from_segment)
        
        # 如果选择的token数量不够，补充一些高分token
        if len(selected_indices) < target_num_tokens:
            remaining = target_num_tokens - len(selected_indices)
            all_scores = scores.cpu().numpy()
            available_indices = [i for i in range(len(all_scores)) if i not in selected_indices]
            
            if len(available_indices) > 0:
                available_scores = [all_scores[i] for i in available_indices]
                additional = heapq.nlargest(
                    min(remaining, len(available_indices)),
                    range(len(available_scores)),
                    available_scores.__getitem__
                )
                selected_indices.extend([available_indices[i] for i in additional])
        
        # 按原始顺序排序
        selected_indices.sort()
        return selected_indices
    
    def prune_tokens(
        self,
        image_features: torch.Tensor,
        target_num_tokens: int = None
    ) -> torch.Tensor:
        """
        对图像token进行自适应剪枝
        Args:
            image_features: [num_tokens, hidden_size] 或 [batch, num_tokens, hidden_size]
            target_num_tokens: 目标保留的token数量（如果为None则使用ratio计算）
        Returns:
            pruned_features: 剪枝后的特征
        """
        original_shape = image_features.shape
        if len(original_shape) == 3:
            # Batch processing
            batch_size = original_shape[0]
            pruned_list = []
            for i in range(batch_size):
                pruned = self._prune_single(image_features[i], target_num_tokens)
                pruned_list.append(pruned)
            return pruned_list  # 返回列表，因为每个样本可能长度不同
        else:
            # Single sample
            return self._prune_single(image_features, target_num_tokens)
    
    def _prune_single(
        self,
        image_features: torch.Tensor,
        target_num_tokens: int = None
    ) -> torch.Tensor:
        """
        对单个样本进行剪枝
        """
        num_tokens = image_features.shape[0]
        
        if target_num_tokens is None:
            target_num_tokens = int(num_tokens * self.target_token_ratio)
        
        target_num_tokens = min(target_num_tokens, num_tokens)
        
        if target_num_tokens >= num_tokens:
            return image_features
        
        # 1. 计算重要性分数
        scores = self.compute_token_importance(image_features)
        
        # 2. 归一化分数
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # 3. 进行树状分割
        all_indices = list(range(num_tokens))
        segments_info, segments_indices = self.tree_split_tokens(
            all_indices, scores_normalized, depth=0
        )
        
        # 4. 从各片段中选择token
        selected_indices = self.select_tokens_from_segments(
            segments_info, segments_indices, scores_normalized, target_num_tokens
        )
        
        # 5. 返回选中的token
        return image_features[selected_indices]
    
    def forward(self, image_features: torch.Tensor, target_num_tokens: int = None):
        """
        前向传播
        """
        return self.prune_tokens(image_features, target_num_tokens)
    

def build_token_pruner(config):

    hidden_size=config.hidden_size if hasattr(config, 'hidden_size') else 4096
    num_branches=config.num_branches if hasattr(config, 'num_branches') else 2
    max_depth=config.max_depth if hasattr(config, 'max_depth') else 6
    importance_threshold=config.importance_threshold if hasattr(config, 'importance_threshold') else 0.8
    target_token_ratio=config.target_token_ratio if hasattr(config, 'target_token_ratio') else 0.5

    
    return AdaptiveImageTokenPruner(hidden_size=hidden_size, num_branches=num_branches, 
                                    max_depth=max_depth, importance_threshold=importance_threshold, 
                                    target_token_ratio=target_token_ratio)
