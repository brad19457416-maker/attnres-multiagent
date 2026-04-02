"""
🔥 技能遗忘策略 (v2 新增)
===

改进设计:
1. 基于「使用次数 × 平均成功率 × 指数时间衰减」计算保留分数
2. score < 阈值才遗忘，更智能
3. 常用好用的技能自动保留，很少用又不好用自动清理

公式:
score = (usage_count × average_success_rate) × exp(-λ × days_since_last_use)

- 使用次数越多 → 分数越高，越常用越保留
- 成功率越高 → 分数越高，越好用越保留
- 越久不用 → 指数衰减 → 分数越低，越可能遗忘
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from attn_types import Skill


class AdaptiveSkillForgetting:
    """自适应技能遗忘
    
    特性:
    - 综合考虑使用次数、成功率、时间衰减
    - 可配置衰减系数和遗忘阈值
    - 支持批量清理遗忘技能
    """
    
    def __init__(
        self,
        lambda_decay: float = 0.01,
        forget_threshold: float = 0.1,
        min_usage_for_retention: int = 1,
    ):
        """
        Args:
            lambda_decay: 每天衰减系数，默认 0.01
                - λ 越大 → 衰减越快
                - λ = 0.01 → 约 70 天后衰减到 50%
                - λ = 0.05 → 约 14 天后衰减到 50%
            forget_threshold: 遗忘阈值，score 低于此值遗忘
            min_usage_for_retention: 最少使用次数才保留，默认 1
        """
        self.lambda_decay = lambda_decay
        self.threshold = forget_threshold
        self.min_usage = min_usage_for_retention
    
    def compute_score(
        self,
        skill: Skill,
        current_time_days: float,
    ) -> float:
        """计算技能保留分数
        
        Args:
            skill: 技能信息
            current_time_days: 当前时间（天数，自某个起点）
            
        Returns:
            保留分数，越高越值得保留
        """
        usage_count = max(skill.usage_count, 1)  # 至少 1
        avg_success = skill.average_success_rate
        days_since_used = current_time_days - skill.last_used_days
        
        # 核心公式
        score = (usage_count * avg_success) * math.exp(-self.lambda_decay * days_since_used)
        
        # 使用次数少于最小值，额外惩罚
        if skill.usage_count < self.min_usage:
            score = score * 0.1
        
        return score
    
    def should_forget(
        self,
        skill: Skill,
        current_time_days: float,
    ) -> bool:
        """判断是否应该遗忘这个技能"""
        score = self.compute_score(skill, current_time_days)
        return score < self.threshold
    
    def filter_forget(
        self,
        skills: List[Skill],
        current_time_days: float,
    ) -> List[Skill]:
        """过滤出需要保留的技能
        
        Returns:
            需要保留的技能列表（需要遗忘的被移除）
        """
        retained = []
        forgotten = []
        
        for skill in skills:
            if self.should_forget(skill, current_time_days):
                forgotten.append(skill)
            else:
                retained.append(skill)
        
        return retained, forgotten
    
    def get_forget_candidates(
        self,
        skills: List[Skill],
        current_time_days: float,
    ) -> List[Skill]:
        """获取候选遗忘技能列表"""
        candidates = []
        for skill in skills:
            if self.should_forget(skill, current_time_days):
                candidates.append(skill)
        return candidates
    
    def update_usage(
        self,
        skill: Skill,
        success: bool,
        current_time_days: float,
    ):
        """更新使用记录（使用后调用）"""
        skill.usage_count += 1
        if success:
            skill.success_count += 1
        skill.last_used_days = current_time_days


# 默认实例
default_forgetting = AdaptiveSkillForgetting()
