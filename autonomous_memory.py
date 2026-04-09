"""
🤖 自主记忆管理
===

让 HGARN 能够自主管理长期记忆：
1. 新技能提取：问题解决后自动提取技能存入技能宫殿
2. 定期清理：运行遗忘算法，移除低分数技能
3. 自动合并：相似技能自动提示合并
4. 技能使用统计：跟踪使用频率和成功率

集成：
- SkillPalace：宫殿式存储
- AdaptiveSkillForgetting：遗忘评分计算
- TemporalFactGraph：时序记录
"""

import math
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from skill_palace import SkillPalace, Closet
from temporal_facts import TemporalFactGraph, TemporalFact
from skill_forgetting import AdaptiveSkillForgetting


@dataclass
class AutonomyConfig:
    """自主记忆管理配置"""
    # 遗忘配置
    enable_forgetting: bool = True
    forget_threshold: float = 0.1
    # 合并配置
    enable_merge_similar: bool = True
    merge_similarity_threshold: float = 0.85  # 相似度超过这个提示合并
    # 整理频率
    min_skills_for_cleanup: int = 50
    # 是否自动添加新技能
    auto_add_new_skill: bool = True


class AutonomousMemoryManager:
    """自主记忆管理器
    
    职责：
    1. 在问题解决后，自动提取技能存入 SkillPalace
    2. 定期运行遗忘清理
    3. 检测相似技能建议合并
    4. 记录技能使用统计
    """
    
    def __init__(
        self,
        skill_palace: SkillPalace,
        temporal_graph: Optional[TemporalFactGraph] = None,
        forgetting: Optional[AdaptiveSkillForgetting] = None,
        config: Optional[AutonomyConfig] = None,
    ):
        self.palace = skill_palace
        self.graph = temporal_graph
        self.forgetting = forgetting or AdaptiveSkillForgetting(
            lambda_decay=0.01,
            forget_threshold=0.1,
        )
        self.config = config or AutonomyConfig()
    
    def get_current_days(self) -> float:
        """获取当前时间（天数从 epoch 开始）"""
        return datetime.now().timestamp() / (24 * 3600)
    
    def add_new_skill(
        self,
        closet_id: str,
        wing_id: str,
        room_id: str,
        hall_id: str,
        skill_name: str,
        compressed_summary: str,
        original_content: str,
        source_episode: str,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ) -> Tuple[Closet, Optional[TemporalFact]]:
        """添加新技能到记忆宫殿
        
        Returns:
            (closet, fact) - fact 是在时序图谱中的记录，如果 graph 为 None 则 fact 为 None
        """
        from attn_types import Skill
        
        # 创建技能
        skill = Skill(
            skill_id=closet_id,
            skill_name=skill_name,
            usage_count=0,
            success_count=0,
            last_used_days=self.get_current_days(),
        )
        
        # 添加到宫殿
        closet, drawer = self.palace.add_skill(
            closet_id=closet_id,
            wing_id=wing_id,
            room_id=room_id,
            hall_id=hall_id,
            skill=skill,
            compressed_summary=compressed_summary,
            original_content=original_content,
            source_episode=source_episode,
            tags=tags,
            embedding=embedding,
        )
        
        # 添加到时序图谱
        fact = None
        if self.graph:
            fact_id = f"skill_{closet_id}_created"
            fact = self.graph.add_fact(
                fact_id=fact_id,
                subject=closet_id,
                predicate="created",
                obj=skill_name,
                confidence=1.0,
                source_closet_id=closet_id,
                source_episode=source_episode,
            )
        
        return closet, fact
    
    def record_skill_usage(self, closet_id: str, success: bool = True):
        """记录一次技能使用，更新统计"""
        closet = self.palace.get_closet(closet_id)
        if closet is None:
            return
        
        self.forgetting.update_usage(
            closet.skill,
            success=success,
            current_time_days=self.get_current_days(),
        )
        
        # 更新 closet 更新时间
        closet.updated_at = datetime.now().timestamp()
        
        # 保存索引
        self.palace._save_index()
    
    def run_cleanup(self) -> Dict[str, Any]:
        """运行一次清理：
        
        1. 移除分数低于阈值的技能
        2. 返回统计信息
        
        Returns:
            清理统计
        """
        if not self.config.enable_forgetting:
            return {"removed": 0, "skipped": 0, "enabled": False}
        
        current_days = self.get_current_days()
        candidates = self.palace.get_forget_candidates(current_days)
        removed = 0
        
        for candidate in candidates:
            score = self.forgetting.compute_score(
                candidate.skill,
                current_days,
            )
            if score < self.config.forget_threshold:
                self.palace.remove_skill(candidate.closet_id)
                removed += 1
        
        total = len(self.palace.closets)
        return {
            "removed": removed,
            "remaining": total - removed,
            "enabled": True,
            "threshold": self.config.forget_threshold,
        }
    
    def find_similar_skills(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[Closet, float]]:
        """查找与给定嵌入相似的技能，用于建议合并"""
        return self.palace.search_by_text(query_embedding, top_k=top_k)
    
    def get_skill_for_merge(
        self,
        similarity_threshold: Optional[float] = None,
    ) -> List[List[Tuple[Closet, float]]]:
        """找出所有相似度超过阈值的技能组，建议合并
        
        Returns:
            每个元素是一组相似技能，需要合并
        """
        if similarity_threshold is None:
            similarity_threshold = self.config.merge_similarity_threshold
        
        # 遍历所有技能，两两比较相似度
        # 这是 O(n^2)，但技能库不会太大，可接受
        all_closets = list(self.palace.closets.values())
        similar_groups: List[List[Tuple[Closet, float]]] = []
        processed = set()
        
        for i, closet_a in enumerate(all_closets):
            if closet_a.embedding is None:
                continue
            if closet_a.closet_id in processed:
                continue
            
            group = [(closet_a, 1.0)]
            processed.add(closet_a.closet_id)
            
            for j in range(i + 1, len(all_closets)):
                closet_b = all_closets[j]
                if closet_b.embedding is None:
                    continue
                if closet_b.closet_id in processed:
                    continue
                
                # 计算余弦相似度
                import math
                dot = sum(a * b for a, b in zip(closet_a.embedding, closet_b.embedding))
                norm_a = math.sqrt(sum(a * a for a in closet_a.embedding))
                norm_b = math.sqrt(sum(b * b for b in closet_b.embedding))
                if norm_a == 0 or norm_b == 0:
                    sim = 0.0
                else:
                    sim = dot / (norm_a * norm_b)
                
                if sim >= similarity_threshold:
                    group.append((closet_b, sim))
                    processed.add(closet_b.closet_id)
            
            if len(group) >= 2:
                # 按相似度排序
                group.sort(key=lambda x: x[1], reverse=True)
                similar_groups.append(group)
        
        return similar_groups
    
    def get_statistics(self) -> Dict[str, int]:
        """获取记忆统计"""
        palace_stats = self.palace.get_statistics()
        if self.graph:
            graph_stats = self.graph.get_statistics()
            palace_stats.update(graph_stats)
        return palace_stats
    
    def suggest_maintenance(self) -> Dict[str, Any]:
        """建议是否需要维护
        
        Returns:
            {
                "needs_cleanup": bool,
                "needs_merge": bool,
                "total_skills": int,
                "similar_groups_count": int,
            }
        """
        stats = self.get_statistics()
        total_skills = stats.get('closets', 0)
        similar_groups = self.get_skill_for_merge()
        
        needs_cleanup = (
            self.config.enable_forgetting and 
            total_skills >= self.config.min_skills_for_cleanup
        )
        needs_merge = (
            self.config.enable_merge_similar and 
            len(similar_groups) > 0
        )
        
        return {
            "needs_cleanup": needs_cleanup,
            "needs_merge": needs_merge,
            "total_skills": total_skills,
            "similar_groups_count": len(similar_groups),
            "similar_groups": similar_groups,
        }


# 默认实例
def create_default_manager(
    storage_root: str = "./skill_palace",
) -> AutonomousMemoryManager:
    """创建默认配置的自主记忆管理器"""
    from skill_palace import create_default_palace
    from temporal_facts import TemporalFactGraph
    
    palace = create_default_palace(storage_root)
    graph = TemporalFactGraph(f"{storage_root}/temporal_facts.json")
    forgetting = AdaptiveSkillForgetting()
    config = AutonomyConfig()
    
    return AutonomousMemoryManager(palace, graph, forgetting, config)


default_manager = create_default_manager()
