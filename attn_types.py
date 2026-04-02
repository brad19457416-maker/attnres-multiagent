"""
类型定义
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ========== 基础类型 ==========

@dataclass
class SubTask:
    """子任务定义"""
    task_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    can_parallel: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0  # 递归深度，0表示顶层
    parent_task_id: Optional[str] = None  # 父任务ID


@dataclass
class DecompositionResult:
    """任务分解结果（包含递归分解）"""
    original_task: SubTask
    decomposed: bool  # 是否成功分解
    subtasks: List[SubTask] = field(default_factory=list)


@dataclass
class SubTaskResult:
    """子任务执行结果"""
    task_id: str
    task: SubTask
    result: str
    success: bool = True
    token_usage: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockResult:
    """Block执行结果（聚合前）"""
    block_id: int
    subtasks: List[SubTask]
    results: List[SubTaskResult]
    start_idx: int
    end_idx: int


@dataclass
class BlockAggregatedResult:
    """Block聚合结果（聚合后）"""
    block_id: int
    aggregated_result: str
    attention_scores: Dict[str, float]  # task_id -> score
    total_token_usage: int
    original_count: int  # 原始子任务数量
    compressed: bool = True  # 是否压缩了
    gate_value: float = 1.0  # 门控值，用于层次残差连接
    residual_connection: bool = True  # 是否启用残差连接


@dataclass
class HierarchicalLevel:
    """层次化结构中的一层"""
    level_id: int
    blocks: List[BlockAggregatedResult]
    aggregated: str
    gate_score: float  # 该层整体门控分数


@dataclass
class RunResult:
    """完整运行结果"""
    query: str
    final_answer: str
    blocks_processed: int
    subtasks_total: int
    total_tokens: int
    blocks: List[BlockAggregatedResult] = field(default_factory=list)
    hierarchical_levels: List[HierarchicalLevel] = field(default_factory=list)
    success: bool = True
    early_stopped: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ========== v2 新增: 并发控制 ==========

@dataclass
class TaskWithPriority:
    """带优先级的任务，用于动态并发调度"""
    task: SubTask
    priority: float  # 优先级，越高越先执行，预估门控分数作为优先级
    expected_gate: float  # 预估门控分数
    
    def __lt__(self, other):
        # 优先级倒序
        return self.priority < other.priority


# ========== v2 新增: 技能遗忘 ==========

@dataclass
class Skill:
    """技能信息，用于遗忘管理"""
    skill_id: str
    skill_name: str
    usage_count: int = 0
    success_count: int = 0
    last_used_days: float = 0.0  # 上次使用时间（距离现在的天数）
    
    @property
    def average_success_rate(self) -> float:
        """平均成功率"""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


# ========== v2 新增: 工作记忆分区 ==========

@dataclass
class InputArea:
    """输入区 - 存储原始查询和全局约束"""
    original_query: str
    global_constraints: Dict[str, Any] = field(default_factory=dict)
    
    def get_query(self) -> str:
        return self.original_query


@dataclass
class ActiveArea:
    """激活区 - 当前正在处理的任务"""
    current_block_id: Optional[int] = None
    current_level_id: Optional[int] = None
    current_tasks: List[Any] = field(default_factory=list)
    
    def clear(self):
        """清空激活区，准备下一个Block"""
        self.current_block_id = None
        self.current_level_id = None
        self.current_tasks.clear()


@dataclass
class ResultArea:
    """结果区 - 已完成的结果，建立残差连接"""
    processed_levels: List[HierarchicalLevel] = field(default_factory=list)
    
    def add_level(self, level: HierarchicalLevel):
        self.processed_levels.append(level)
    
    def get_all_levels(self) -> List[HierarchicalLevel]:
        return self.processed_levels
    
    def __len__(self):
        return len(self.processed_levels)


@dataclass
class ReverseActivationRequest:
    """反向激活请求"""
    topic: str
    reason: str
    information_gain: float  # 预估信息增益 0-1
    source_level: int  # 来自哪一层
    
    def should_trigger(self, threshold: float = 0.3) -> bool:
        """是否应该触发反向激活 - 增益>阈值才执行"""
        return self.information_gain > threshold


@dataclass
class GatingSignalArea:
    """门控信号区 - 传递反向投射的门控信号"""
    reverse_activation_requests: List[ReverseActivationRequest] = field(default_factory=list)
    
    def add_request(self, request: ReverseActivationRequest):
        self.reverse_activation_requests.append(request)
    
    def get_pending_requests(self, gain_threshold: float = 0.3) -> List[ReverseActivationRequest]:
        """获取待处理的反向激活请求（过滤掉增益不足的）"""
        return [r for r in self.reverse_activation_requests if r.should_trigger(gain_threshold)]
    
    def clear_pending(self):
        """清空已处理的请求"""
        self.reverse_activation_requests.clear()


@dataclass
class SharedArea:
    """共享区 - 所有任务可读的公共信息"""
    public_context: Dict[str, Any] = field(default_factory=dict)
    discovered_facts: List[str] = field(default_factory=list)
    
    def add_fact(self, fact: str):
        self.discovered_facts.append(fact)
    
    def get_context_text(self) -> str:
        """获取共享上下文的文本表示"""
        facts_text = "\n".join([f"- {f}" for f in self.discovered_facts])
        return f"共享发现的事实:\n{facts_text}" if facts_text else ""


@dataclass
class WorkingMemory:
    """工作记忆 - 分区存储设计 (v2 新增)"""
    input_area: InputArea
    active_area: ActiveArea = field(default_factory=ActiveArea)
    result_area: ResultArea = field(default_factory=ResultArea)
    gating_area: GatingSignalArea = field(default_factory=GatingSignalArea)
    shared_area: SharedArea = field(default_factory=SharedArea)
    
    @classmethod
    def create(cls, query: str) -> 'WorkingMemory':
        """从原始查询创建工作记忆"""
        return cls(
            input_area=InputArea(original_query=query)
        )
    
    def get_original_query(self) -> str:
        return self.input_area.get_query()
    
    def clear_active(self):
        """清空激活区，准备下一个Block"""
        self.active_area.clear()
    
    def commit_level_result(self, level: HierarchicalLevel):
        """提交处理完成的层次到结果区"""
        self.result_area.add_level(level)
    
    def add_reverse_activation(self, request: ReverseActivationRequest):
        """添加反向激活请求"""
        self.gating_area.add_request(request)
    
    def get_pending_reverse_activations(self, gain_threshold: float = 0.3) -> List[ReverseActivationRequest]:
        """获取待处理的反向激活请求"""
        return self.gating_area.get_pending_requests(gain_threshold)
    
    def get_previous_levels(self) -> List[HierarchicalLevel]:
        """获取之前处理完的所有层次"""
        return self.result_area.get_all_levels()
    
    def get_full_context(self) -> str:
        """获取完整上下文文本（用于LLM调用）"""
        parts = [
            f"原始查询: {self.get_original_query()}"
        ]
        
        shared = self.shared_area.get_context_text()
        if shared:
            parts.append(shared)
        
        if len(self.result_area) > 0:
            prev_text = "\n".join([
                f"Level {lv.level_id} (门控={lv.gate_score:.3f}):\n{lv.aggregated}"
                for lv in self.result_area.get_all_levels()
            ])
            parts.append(f"之前层次处理结果:\n{prev_text}")
        
        return "\n\n".join(parts)
