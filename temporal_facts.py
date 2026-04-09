"""
📅 时序事实存储 (Temporal Facts)
===

借鉴 Graphiti 理念：
- 每个事实都有**有效性窗口**：`valid_from → valid_to`
- 信息更新不删除旧事实，只标记旧事实失效
- 支持查询"X 在 Y 时间是什么状态"
- 每条事实都溯源到原始 closet/episode

支持：
1. 实体-关系-实体 三元组结构
2. 有效性时间窗口
3. 增量更新，保留完整历史
4. 基于时间的查询
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from attn_types import Skill


@dataclass
class TemporalFact:
    """时序事实 - 实体-关系-实体 三元组，带有效性窗口"""
    fact_id: str
    subject: str  # 主体实体
    predicate: str  # 关系
    object: str  # 客体实体
    valid_from: float  # 开始有效时间戳
    valid_to: Optional[float] = None  # 结束有效时间戳，None 表示当前仍有效
    confidence: float = 1.0  # 置信度 0-1
    source_closet_id: Optional[str] = None  # 来源 closet
    source_episode: str = ""  # 来源对话 ID
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid_at(self, timestamp: float) -> bool:
        """检查在给定时间点是否有效"""
        if timestamp < self.valid_from:
            return False
        if self.valid_to is not None and timestamp > self.valid_to:
            return False
        return True
    
    def is_currently_valid(self) -> bool:
        """当前是否有效"""
        now = datetime.now().timestamp()
        return self.is_valid_at(now)
    
    def invalidate(self, timestamp: float):
        """标记这个事实在给定时间后失效"""
        self.valid_to = timestamp


@dataclass
class Entity:
    """实体，用于索引"""
    entity_id: str
    name: str
    description: str = ""
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalFactGraph:
    """时序事实图谱 - 支持时间维度的实体关系查询"""
    
    def __init__(self, storage_path: str = "./skill_palace/temporal_facts.json"):
        self.storage_path = Path(storage_path)
        self.entities: Dict[str, Entity] = {}
        self.facts: Dict[str, TemporalFact] = {}
        
        # 反向索引：subject -> [facts]
        self.subject_index: Dict[str, List[str]] = {}
        # 反向索引：object -> [facts]
        self.object_index: Dict[str, List[str]] = {}
        # 反向索引：predicate -> [facts]
        self.predicate_index: Dict[str, List[str]] = {}
        
        # 加载已保存的数据
        self._load()
    
    def _load(self):
        """从磁盘加载"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for e in data.get('entities', []):
                    entity = Entity(**e)
                    self.entities[entity.entity_id] = entity
                for f in data.get('facts', []):
                    fact = TemporalFact(**f)
                    self.facts[fact.fact_id] = fact
                    self._update_index(fact)
    
    def _save(self):
        """保存到磁盘"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'entities': [e.__dict__ for e in self.entities.values()],
            'facts': [f.__dict__ for f in self.facts.values()],
        }
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _update_index(self, fact: TemporalFact):
        """更新反向索引"""
        if fact.subject not in self.subject_index:
            self.subject_index[fact.subject] = []
        self.subject_index[fact.subject].append(fact.fact_id)
        
        if fact.object not in self.object_index:
            self.object_index[fact.object] = []
        self.object_index[fact.object].append(fact.fact_id)
        
        if fact.predicate not in self.predicate_index:
            self.predicate_index[fact.predicate] = []
        self.predicate_index[fact.predicate].append(fact.fact_id)
    
    def add_entity(self, entity_id: str, name: str, description: str = "") -> Entity:
        """添加实体"""
        if entity_id in self.entities:
            raise ValueError(f"Entity {entity_id} already exists")
        entity = Entity(entity_id=entity_id, name=name, description=description)
        self.entities[entity_id] = entity
        self._save()
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)
    
    def add_fact(
        self,
        fact_id: str,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: Optional[float] = None,
        confidence: float = 1.0,
        source_closet_id: Optional[str] = None,
        source_episode: str = "",
    ) -> TemporalFact:
        """添加一个新事实
        如果之前有同一 subject-predicate 的当前有效事实，自动标记它们失效
        """
        now = datetime.now().timestamp()
        if valid_from is None:
            valid_from = now
        
        # 查找同一 subject-predicate 的当前有效事实，标记失效
        existing = self.query_facts(
            subject=subject,
            predicate=predicate,
            valid_at=now
        )
        for old_fact in existing:
            if old_fact.is_currently_valid():
                old_fact.invalidate(now)
        
        # 创建新事实
        fact = TemporalFact(
            fact_id=fact_id,
            subject=subject,
            predicate=predicate,
            object=obj,
            valid_from=valid_from,
            confidence=confidence,
            source_closet_id=source_closet_id,
            source_episode=source_episode,
        )
        self.facts[fact_id] = fact
        self._update_index(fact)
        self._save()
        return fact
    
    def query_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        valid_at: Optional[float] = None,
        include_invalid: bool = False,
    ) -> List[TemporalFact]:
        """查询事实
        
        参数：
        - subject: 过滤主体
        - predicate: 过滤关系
        - obj: 过滤客体
        - valid_at: 只返回在这个时间点有效的事实
        - include_invalid: 是否包含已失效的事实
        """
        # 基于索引缩小范围
        candidate_ids = None
        
        if subject is not None:
            candidate_ids = set(self.subject_index.get(subject, []))
        
        if predicate is not None:
            pred_ids = set(self.predicate_index.get(predicate, []))
            if candidate_ids is None:
                candidate_ids = pred_ids
            else:
                candidate_ids = candidate_ids.intersection(pred_ids)
        
        if obj is not None:
            obj_ids = set(self.object_index.get(obj, []))
            if candidate_ids is None:
                candidate_ids = obj_ids
            else:
                candidate_ids = candidate_ids.intersection(obj_ids)
        
        # 如果没有过滤条件，就是所有
        if candidate_ids is None:
            candidate_ids = set(self.facts.keys())
        
        # 过滤结果
        results = []
        for fact_id in candidate_ids:
            fact = self.facts[fact_id]
            
            # 时间有效性过滤
            if valid_at is not None and not fact.is_valid_at(valid_at):
                if not include_invalid:
                    continue
            
            if not include_invalid and fact.valid_to is not None:
                # 如果不包含失效，跳过已失效的
                continue
            
            results.append(fact)
        
        # 按置信度降序
        results.sort(key=lambda f: f.confidence, reverse=True)
        return results
    
    def get_current_fact(
        self,
        subject: str,
        predicate: str,
    ) -> Optional[TemporalFact]:
        """获取 subject-predicate 当前有效的事实（最多一个，因为更新会标记旧的失效）"""
        now = datetime.now().timestamp()
        results = self.query_facts(subject=subject, predicate=predicate, valid_at=now)
        # 按置信度排序，返回第一个
        return results[0] if results else None
    
    def get_history(
        self,
        subject: str,
        predicate: str,
    ) -> List[TemporalFact]:
        """获取这个 subject-predicate 的所有历史版本（包括已失效的）"""
        return self.query_facts(
            subject=subject,
            predicate=predicate,
            include_invalid=True
        )
    
    def get_entity_neighbors(self, entity: str, valid_at: Optional[float] = None) -> List[TemporalFact]:
        """获取实体所有关联的事实"""
        results = []
        # 作为主体
        results.extend(self.query_facts(subject=entity, valid_at=valid_at))
        # 作为客体
        results.extend(self.query_facts(obj=entity, valid_at=valid_at))
        # 去重
        seen = set()
        unique = []
        for f in results:
            if f.fact_id not in seen:
                seen.add(f.fact_id)
                unique.append(f)
        # 按置信度排序
        unique.sort(key=lambda f: f.confidence, reverse=True)
        return unique
    
    def get_statistics(self) -> Dict[str, int]:
        return {
            'entities': len(self.entities),
            'facts': len(self.facts),
            'current_valid_facts': sum(1 for f in self.facts.values() if f.is_currently_valid()),
        }
    
    def invalidate_fact(self, fact_id: str, timestamp: Optional[float] = None) -> bool:
        """标记一个事实失效"""
        if fact_id not in self.facts:
            return False
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        self.facts[fact_id].invalidate(timestamp)
        self._save()
        return True


# 默认实例
default_graph = TemporalFactGraph()
