"""
🏛️ 技能宫殿 (Skill Palace)
===

借鉴 MemPalace 理念的结构化技能库存储：
- Wing: 大领域分类（编程、设计、投资、生活等）
- Room: 同一 Wing 下的具体主题
- Hall: 同一 Room 下按记忆类型分类（decisions, discoveries, preferences, problems 等）
- Closet: 技能块（压缩摘要 + 索引）
- Drawer: 原始完整对话/内容（可追溯原文）

组织结构：
wing_id → room_id → hall_id → [closet_id → Closet]
每个 Closet 对应一个 Skill，包含 drawer 指向原文
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from attn_types import Skill
from skill_forgetting import AdaptiveSkillForgetting


# ========== 宫殿结构类型 ==========

@dataclass
class Wing:
    """翼 - 最大分类，对应大领域"""
    wing_id: str
    name: str
    description: str = ""
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Room:
    """房间 - 同一 Wing 下的具体主题"""
    room_id: str
    wing_id: str
    name: str
    description: str = ""
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hall:
    """厅 - 同一 Room 下按记忆类型分类"""
    hall_id: str
    room_id: str
    name: str  # 类型名称: decisions, discoveries, preferences, problems, skills
    description: str = ""
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class Closet:
    """衣橱 - 技能块，包含压缩摘要"""
    closet_id: str
    hall_id: str
    room_id: str
    wing_id: str
    skill: Skill  # 技能信息（包含遗忘评分）
    compressed_summary: str  # AAAK 压缩摘要
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # 向量嵌入，用于检索


@dataclass
class Drawer:
    """抽屉 - 存储原始完整内容"""
    drawer_id: str
    closet_id: str
    original_content: str  # 原始完整对话/内容
    source_episode: str  # 来源对话 ID
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())


# ========== 技能宫殿存储 ==========

class SkillPalace:
    """宫殿式技能库存储
    
    组织结构:
    SkillPalace
    ├── Wings (dict: wing_id → Wing)
    │   └── Room (dict: room_id → Room)
    │       └── Hall (dict: hall_id → Hall)
    │           └── Closet list (closet_id → Closet)
    │               └── Drawer (separate file)
    """
    
    def __init__(
        self,
        storage_root: str = "./skill_palace",
        forgetting: Optional[AdaptiveSkillForgetting] = None,
    ):
        self.root = Path(storage_root)
        self.forgetting = forgetting or AdaptiveSkillForgetting()
        
        # 内存索引
        self.wings: Dict[str, Wing] = {}
        self.rooms: Dict[str, Room] = {}  # room_id → Room
        self.halls: Dict[str, Hall] = {}  # hall_id → Hall
        self.closets: Dict[str, Closet] = {}  # closet_id → Closet
        
        # 确保目录结构存在
        self._ensure_dirs()
        
        # 加载已保存的索引
        self._load_index()
    
    def _ensure_dirs(self):
        """确保目录结构存在"""
        dirs = [
            self.root / "wings",
            self.root / "rooms",
            self.root / "halls",
            self.root / "closets",
            self.root / "drawers",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _index_path(self) -> Path:
        return self.root / "palace_index.json"
    
    def _load_index(self):
        """从磁盘加载索引"""
        idx_path = self._index_path()
        if idx_path.exists():
            with open(idx_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for w in data.get('wings', []):
                    wing = Wing(**w)
                    self.wings[wing.wing_id] = wing
                for r in data.get('rooms', []):
                    room = Room(**r)
                    self.rooms[room.room_id] = room
                for h in data.get('halls', []):
                    hall = Hall(**h)
                    self.halls[hall.hall_id] = hall
                for c in data.get('closets', []):
                    # skill 需要单独构造
                    skill_data = c.pop('skill')
                    skill = Skill(**skill_data)
                    closet = Closet(skill=skill, **c)
                    self.closets[closet.closet_id] = closet
    
    def _save_index(self):
        """保存索引到磁盘"""
        data = {
            'wings': [w.__dict__ for w in self.wings.values()],
            'rooms': [r.__dict__ for r in self.rooms.values()],
            'halls': [h.__dict__ for h in self.halls.values()],
            'closets': [{
                **c.__dict__,
                'skill': c.skill.__dict__,
            } for c in self.closets.values()],
        }
        with open(self._index_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _closet_path(self, closet_id: str) -> Path:
        return self.root / "closets" / f"{closet_id}.json"
    
    def _drawer_path(self, drawer_id: str) -> Path:
        return self.root / "drawers" / f"{drawer_id}.md"
    
    # ========== Wing 操作 ==========
    
    def add_wing(self, wing_id: str, name: str, description: str = "") -> Wing:
        """添加一个新 Wing（大领域）"""
        if wing_id in self.wings:
            raise ValueError(f"Wing {wing_id} already exists")
        wing = Wing(wing_id=wing_id, name=name, description=description)
        self.wings[wing_id] = wing
        self._save_index()
        return wing
    
    def get_wing(self, wing_id: str) -> Optional[Wing]:
        return self.wings.get(wing_id)
    
    def list_wings(self) -> List[Wing]:
        return list(self.wings.values())
    
    # ========== Room 操作 ==========
    
    def add_room(self, room_id: str, wing_id: str, name: str, description: str = "") -> Room:
        """添加一个新 Room（主题）"""
        if room_id in self.rooms:
            raise ValueError(f"Room {room_id} already exists")
        if wing_id not in self.wings:
            raise ValueError(f"Wing {wing_id} does not exist")
        room = Room(room_id=room_id, wing_id=wing_id, name=name, description=description)
        self.rooms[room_id] = room
        # 更新 wing 更新时间
        self.wings[wing_id].updated_at = datetime.now().timestamp()
        self._save_index()
        return room
    
    def get_room(self, room_id: str) -> Optional[Room]:
        return self.rooms.get(room_id)
    
    def list_rooms(self, wing_id: str) -> List[Room]:
        """列出某个 Wing 下的所有 Room"""
        return [r for r in self.rooms.values() if r.wing_id == wing_id]
    
    # ========== Hall 操作 ==========
    
    def add_hall(self, hall_id: str, room_id: str, name: str, description: str = "") -> Hall:
        """添加一个新 Hall（类型分类）
        
        标准 hall 名称：
        - decisions: 决策记录
        - discoveries: 发现/结论
        - preferences: 用户偏好
        - problems: 问题解决方案
        - skills: 可复用技能
        """
        if hall_id in self.halls:
            raise ValueError(f"Hall {hall_id} already exists")
        if room_id not in self.rooms:
            raise ValueError(f"Room {room_id} does not exist")
        hall = Hall(hall_id=hall_id, room_id=room_id, name=name, description=description)
        self.halls[hall_id] = hall
        # 更新 room 更新时间
        self.rooms[room_id].updated_at = datetime.now().timestamp()
        self._save_index()
        return hall
    
    def get_hall(self, hall_id: str) -> Optional[Hall]:
        return self.halls.get(hall_id)
    
    def list_halls(self, room_id: str) -> List[Hall]:
        """列出某个 Room 下的所有 Hall"""
        return [h for h in self.halls.values() if h.room_id == room_id]
    
    # ========== Closet + Drawer 操作 ==========
    
    def add_skill(
        self,
        closet_id: str,
        wing_id: str,
        room_id: str,
        hall_id: str,
        skill: Skill,
        compressed_summary: str,
        original_content: str,
        source_episode: str,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ) -> tuple[Closet, Drawer]:
        """添加一个技能，同时创建 Closet（摘要）和 Drawer（原文）"""
        if closet_id in self.closets:
            raise ValueError(f"Closet {closet_id} already exists")
        
        # 验证路径存在
        if wing_id not in self.wings:
            raise ValueError(f"Wing {wing_id} does not exist")
        if room_id not in self.rooms:
            raise ValueError(f"Room {room_id} does not exist")
        if hall_id not in self.halls:
            raise ValueError(f"Hall {hall_id} does not exist")
        
        # 创建 Closet
        closet = Closet(
            closet_id=closet_id,
            hall_id=hall_id,
            room_id=room_id,
            wing_id=wing_id,
            skill=skill,
            compressed_summary=compressed_summary,
            tags=tags or [],
            embedding=embedding,
        )
        self.closets[closet_id] = closet
        
        # 创建 Drawer
        drawer_id = f"{closet_id}_original"
        drawer = Drawer(
            drawer_id=drawer_id,
            closet_id=closet_id,
            original_content=original_content,
            source_episode=source_episode,
        )
        # 保存 drawer 到文件
        with open(self._drawer_path(drawer_id), 'w', encoding='utf-8') as f:
            f.write(drawer.original_content)
        
        # 更新索引
        self.halls[hall_id].updated_at = datetime.now().timestamp()
        self._save_index()
        
        return closet, drawer
    
    def get_closet(self, closet_id: str) -> Optional[Closet]:
        return self.closets.get(closet_id)
    
    def get_drawer(self, closet_id: str) -> Optional[str]:
        """获取 closet 的原始原文"""
        drawer_id = f"{closet_id}_original"
        drawer_path = self._drawer_path(drawer_id)
        if drawer_path.exists():
            with open(drawer_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def list_closets(self, hall_id: str) -> List[Closet]:
        """列出某个 Hall 下的所有 Closet"""
        return [c for c in self.closets.values() if c.hall_id == hall_id]
    
    def search_by_tags(self, tags: List[str]) -> List[Closet]:
        """按标签搜索 Closet"""
        results = []
        for closet in self.closets.values():
            if any(tag in closet.tags for tag in tags):
                results.append(closet)
        return results
    
    # ========== 技能遗忘 ==========
    
    def get_forget_candidates(self, current_time_days: float) -> List[Closet]:
        """获取需要遗忘的候选 Closet"""
        candidates = []
        for closet in self.closets.values():
            if self.forgetting.should_forget(closet.skill, current_time_days):
                candidates.append(closet)
        return candidates
    
    def remove_skill(self, closet_id: str) -> bool:
        """移除一个技能（同时删除 closet 和 drawer）"""
        if closet_id not in self.closets:
            return False
        
        closet = self.closets[closet_id]
        
        # 删除 drawer
        drawer_id = f"{closet_id}_original"
        drawer_path = self._drawer_path(drawer_id)
        if drawer_path.exists():
            drawer_path.unlink()
        
        # 删除 closet 从索引
        del self.closets[closet_id]
        self._save_index()
        return True
    
    def run_forgetting(self, current_time_days: float) -> int:
        """运行遗忘，移除分数低于阈值的技能，返回移除数量"""
        candidates = self.get_forget_candidates(current_time_days)
        removed = 0
        for candidate in candidates:
            self.remove_skill(candidate.closet_id)
            removed += 1
        return removed
    
    # ========== 检索 ==========
    
    def search_by_text(self, query_embedding: List[float], top_k: int = 5) -> List[tuple[Closet, float]]:
        """基于向量相似度搜索，返回最相似的 Closet"""
        import math
        
        # 计算余弦相似度
        results: List[tuple[float, Closet]] = []
        for closet in self.closets.values():
            if closet.embedding is None:
                continue
            
            # 点积
            dot = sum(a * b for a, b in zip(query_embedding, closet.embedding))
            # 范数
            norm_q = math.sqrt(sum(a * a for a in query_embedding))
            norm_c = math.sqrt(sum(a * a for a in closet.embedding))
            
            if norm_q == 0 or norm_c == 0:
                sim = 0.0
            else:
                sim = dot / (norm_q * norm_c)
            
            results.append((sim, closet))
        
        # 按相似度降序排序
        results.sort(reverse=True, key=lambda x: x[0])
        
        # 返回 top-k
        return [(c, s) for s, c in results[:top_k]]
    
    def get_statistics(self) -> Dict[str, int]:
        """获取宫殿统计信息"""
        return {
            'wings': len(self.wings),
            'rooms': len(self.rooms),
            'halls': len(self.halls),
            'closets': len(self.closets),
        }


# ========== 便捷初始化 ==========

def create_default_palace(storage_root: str = "./skill_palace") -> SkillPalace:
    """创建一个默认配置的技能宫殿，创建标准结构"""
    palace = SkillPalace(storage_root=storage_root)
    
    # 如果是空的，创建默认结构
    if len(palace.wings) == 0:
        # 默认 Wings
        palace.add_wing("development", "开发", "软件开发、架构设计、编程相关")
        palace.add_wing("research", "研究", "AI 研究、论文调研、技术探索")
        palace.add_wing("personal", "个人", "个人信息、偏好、决策")
        palace.add_wing("projects", "项目", "各种具体项目")
    
    return palace


# 默认实例
default_palace = create_default_palace()
