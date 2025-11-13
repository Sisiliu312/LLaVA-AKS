import torch
import torch.nn as nn
import heapq
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict


class AdaptiveImageTokenPruner(nn.Module):
    """
    Adaptive image token pruning using tree-based hierarchical splitting.
    Similar to video frame selection but operates on image tokens.
    """
    
    def __init__(
        self,
        num_branches: int = 2,
        max_depth: int = 5,
        importance_threshold: float = 0.8,
        std_threshold: float = -100,
        target_token_ratio: float = 0.5,
        verbose: bool = False,  # ✅ 添加 verbose 参数
    ):
        super().__init__()
        self.num_branches = num_branches
        self.max_depth = max_depth
        self.importance_threshold = importance_threshold
        self.std_threshold = std_threshold
        self.target_token_ratio = target_token_ratio
        self.verbose = verbose
        
        # ✅ 用于记录树结构
        self.tree_structure = []

    def compute_token_importance(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        基于自注意力的重要性（training-free）
        """
        tokens_norm = F.normalize(tokens, p=2, dim=-1)
        attention = tokens_norm @ tokens_norm.T
        importance = attention.sum(dim=0)
        return importance
    
    def tree_split_tokens(
        self,
        token_indices: List[int],
        scores: torch.Tensor,
        depth: int = 0,
        node_id: str = "root"  # ✅ 添加节点ID用于追踪
    ) -> Tuple[List[Dict], List[List[int]]]:
        """
        递归地进行树状分割，并记录树结构
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
        
        # ✅ 记录节点信息
        node_info = {
            'node_id': node_id,
            'depth': depth,
            'token_range': (min(token_indices), max(token_indices)),
            'num_tokens': len(token_indices),
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'mean_diff': float(mean_diff),
            'should_not_split': should_not_split,
            'reach_max_depth': reach_max_depth,
            'too_small': too_small,
            'decision': None,
            'children': []
        }
        
        # 决策逻辑
        if should_not_split or reach_max_depth or too_small:
            # 不再分割
            if should_not_split:
                node_info['decision'] = f"KEEP (high importance: mean_diff={mean_diff:.3f} > {self.importance_threshold})"
            elif reach_max_depth:
                node_info['decision'] = f"STOP (max depth={self.max_depth} reached)"
            elif too_small:
                node_info['decision'] = f"STOP (too small: {len(token_indices)} tokens < {self.num_branches*2})"
            
            self.tree_structure.append(node_info)
            return [{'score': segment_scores, 'depth': depth}], [token_indices]
        
        # 进行树状分割
        node_info['decision'] = f"SPLIT into {self.num_branches} branches"
        
        branch_size = len(token_indices) // self.num_branches
        split_segments = []
        split_indices = []
        
        for i in range(self.num_branches):
            start_idx = i * branch_size
            if i == self.num_branches - 1:
                end_idx = len(token_indices)
            else:
                end_idx = (i + 1) * branch_size
            
            branch_indices = token_indices[start_idx:end_idx]
            
            if len(branch_indices) > 0:
                # ✅ 为子节点创建ID
                child_node_id = f"{node_id}.{i}"
                node_info['children'].append(child_node_id)
                
                # 递归处理
                sub_segments, sub_indices = self.tree_split_tokens(
                    branch_indices, scores, depth + 1, child_node_id
                )
                split_segments.extend(sub_segments)
                split_indices.extend(sub_indices)
        
        self.tree_structure.append(node_info)
        return split_segments, split_indices
    
    def print_tree_structure(self):
        """
        ✅ 打印树结构的可视化
        """
        if not self.tree_structure:
            print("No tree structure available. Run pruning first.")
            return
        
        print("\n" + "=" * 80)
        print("TOKEN PRUNING TREE STRUCTURE")
        print("=" * 80)
        
        # 按深度排序
        sorted_nodes = sorted(self.tree_structure, key=lambda x: (x['depth'], x['node_id']))
        
        for node in sorted_nodes:
            indent = "  " * node['depth']
            prefix = "├─" if node['depth'] > 0 else "ROOT"
            
            print(f"\n{indent}{prefix} Node: {node['node_id']}")
            print(f"{indent}   ├─ Depth: {node['depth']}")
            print(f"{indent}   ├─ Token Range: {node['token_range'][0]}-{node['token_range'][1]} ({node['num_tokens']} tokens)")
            print(f"{indent}   ├─ Statistics:")
            print(f"{indent}   │  ├─ Mean Score: {node['mean_score']:.4f}")
            print(f"{indent}   │  ├─ Std Score: {node['std_score']:.4f}")
            print(f"{indent}   │  └─ Mean Diff: {node['mean_diff']:.4f}")
            print(f"{indent}   ├─ Decision: {node['decision']}")
            
            if node['children']:
                print(f"{indent}   └─ Children: {', '.join(node['children'])}")
        
        print("\n" + "=" * 80)
    
    def print_tree_summary(self):
        """
        ✅ 打印树的统计摘要
        """
        if not self.tree_structure:
            print("No tree structure available.")
            return
        
        total_nodes = len(self.tree_structure)
        max_depth = max(node['depth'] for node in self.tree_structure)
        
        # 统计每层的节点数
        depth_counts = {}
        leaf_nodes = []
        
        for node in self.tree_structure:
            depth = node['depth']
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            
            if not node['children']:
                leaf_nodes.append(node)
        
        print("\n" + "=" * 80)
        print("TREE SUMMARY")
        print("=" * 80)
        print(f"Total Nodes: {total_nodes}")
        print(f"Max Depth: {max_depth}")
        print(f"Leaf Nodes: {len(leaf_nodes)}")
        print(f"\nNodes per depth:")
        for depth in sorted(depth_counts.keys()):
            print(f"  Depth {depth}: {depth_counts[depth]} nodes")
        
        print(f"\nLeaf Node Details:")
        for i, leaf in enumerate(leaf_nodes, 1):
            print(f"  Leaf {i}: Tokens {leaf['token_range'][0]}-{leaf['token_range'][1]} "
                  f"({leaf['num_tokens']} tokens, depth={leaf['depth']})")
        
        print("=" * 80 + "\n")
    
    def select_tokens_from_segments(
        self,
        segments_info: List[Dict],
        segments_indices: List[List[int]],
        scores: torch.Tensor,
        target_num_tokens: int
    ) -> List[int]:
        """
        从各个片段中选择token
        """
        selected_indices = []
        
        # ✅ 记录选择过程
        if self.verbose:
            print("\n" + "=" * 80)
            print("TOKEN SELECTION FROM SEGMENTS")
            print("=" * 80)
        
        for seg_idx, (segment_info, segment_idx) in enumerate(zip(segments_info, segments_indices)):
            depth = segment_info['depth']
            segment_quota = int(target_num_tokens / (self.num_branches ** depth))
            segment_quota = max(1, min(segment_quota, len(segment_idx)))
            
            segment_scores = scores[segment_idx].cpu().numpy()
            top_k_local = heapq.nlargest(
                segment_quota, 
                range(len(segment_scores)), 
                segment_scores.__getitem__
            )
            selected_from_segment = [segment_idx[i] for i in top_k_local]
            selected_indices.extend(selected_from_segment)
            
            if self.verbose:
                print(f"\nSegment {seg_idx + 1} (depth={depth}):")
                print(f"  Token range: {min(segment_idx)}-{max(segment_idx)}")
                print(f"  Total tokens: {len(segment_idx)}")
                print(f"  Quota (based on depth): {segment_quota}")
                print(f"  Selected: {len(selected_from_segment)} tokens")
                print(f"  Selected indices: {sorted(selected_from_segment)[:10]}..." if len(selected_from_segment) > 10 else f"  Selected indices: {sorted(selected_from_segment)}")
        
        # 补充token
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
                additional_indices = [available_indices[i] for i in additional]
                selected_indices.extend(additional_indices)
                
                if self.verbose:
                    print(f"\n⚠ Additional token selection:")
                    print(f"  Needed: {remaining} more tokens")
                    print(f"  Added: {len(additional_indices)} tokens")
        
        selected_indices.sort()
        
        if self.verbose:
            print(f"\n✓ Final selection: {len(selected_indices)} tokens")
            print("=" * 80 + "\n")
        
        return selected_indices
    
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
        
        # ✅ 清空之前的树结构
        self.tree_structure = []
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("STARTING TOKEN PRUNING")
            print("=" * 80)
            print(f"Input: {num_tokens} tokens")
            print(f"Target: {target_num_tokens} tokens ({target_num_tokens/num_tokens*100:.1f}%)")
            print(f"Settings: branches={self.num_branches}, max_depth={self.max_depth}")
            print("=" * 80)
        
        # 1. 计算重要性分数
        scores = self.compute_token_importance(image_features)
        
        # 2. 归一化分数
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # 3. 进行树状分割
        all_indices = list(range(num_tokens))
        segments_info, segments_indices = self.tree_split_tokens(
            all_indices, scores_normalized, depth=0, node_id="root"
        )
        
        # ✅ 打印树结构
        if self.verbose:
            self.print_tree_structure()
            self.print_tree_summary()
        
        # 4. 从各片段中选择token
        selected_indices = self.select_tokens_from_segments(
            segments_info, segments_indices, scores_normalized, target_num_tokens
        )
        
        if self.verbose:
            print(f"\n✓ Pruning completed: {num_tokens} → {len(selected_indices)} tokens\n")
        
        # 5. 返回选中的token
        return image_features[selected_indices]
    
    def prune_tokens(
        self,
        image_features: torch.Tensor,
        target_num_tokens: int = None
    ) -> torch.Tensor:
        """
        对图像token进行自适应剪枝
        """
        original_shape = image_features.shape
        if len(original_shape) == 3:
            # Batch processing
            batch_size = original_shape[0]
            pruned_list = []
            for i in range(batch_size):
                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"Processing batch {i+1}/{batch_size}")
                    print(f"{'='*80}")
                pruned = self._prune_single(image_features[i], target_num_tokens)
                pruned_list.append(pruned)
            return pruned_list
        else:
            return self._prune_single(image_features, target_num_tokens)
    
    def forward(self, image_features: torch.Tensor, target_num_tokens: int = None):
        """
        前向传播
        """
        return self.prune_tokens(image_features, target_num_tokens)


def build_token_pruner(config):
    """从 config 构建 token pruner"""
    num_branches = getattr(config, 'pruner_num_branches', 2)
    max_depth = getattr(config, 'pruner_max_depth', 6)
    importance_threshold = getattr(config, 'pruner_importance_threshold', 0.8)
    target_token_ratio = getattr(config, 'pruner_target_ratio', 0.5)
    verbose = getattr(config, 'pruner_verbose', False)
    
    return AdaptiveImageTokenPruner(
        num_branches=num_branches, 
        max_depth=max_depth, 
        importance_threshold=importance_threshold, 
        target_token_ratio=target_token_ratio,
        verbose=verbose
    )