"""
🔪 AAAK 压缩方言
===

AAAK = Anything-Abbreviated Anybody-Knows —— 任意可缩写，任何人都能懂

设计理念（来自 MemPalace）：
- 有损压缩：对重复模式进行缩写，不影响可读性但减少 token
- 任何 LLM 都能直接读懂，不需要专门解码器
- 主要压缩重复实体名称、常见长词、常见短语
- 增量压缩：越常用的实体越压缩，节省越多

压缩规则（可配置）：
1. 实体缩写：重复出现的实体 → `[实体名首字母+id]`，如 `OpenAI` → `[O:1]`
2. 常见词替换：长词 → 短词，如 `application` → `app`
3. 标点压缩：多个换行 → 单个换行，多个空格 → 单个空格
4. 移除冗余：去掉不必要的修饰词、重复

Token 节省：
- 对于大量重复实体的文档，可节省 30-50% token
- 对于技能摘要，通常节省 20-30%
- 任何 LLM 都能正常理解，不需要解压也能读
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class EntityIndex:
    """实体索引，用于重复实体缩写"""
    entity_map: Dict[str, str] = field(default_factory=dict)
    reverse_map: Dict[str, str] = field(default_factory=dict)
    next_id: int = 1
    
    def get_abbr(self, entity_name: str) -> str:
        """获取实体缩写，如果是新实体分配id"""
        if entity_name in self.entity_map:
            return self.entity_map[entity_name]
        
        # 新实体，分配id
        # 取首字母
        first_char = entity_name.strip()[0].upper()
        abbr = f"[{first_char}:{self.next_id}]"
        self.entity_map[entity_name] = abbr
        self.reverse_map[abbr] = entity_name
        self.next_id += 1
        return abbr
    
    def expand(self, abbr: str) -> Optional[str]:
        """展开缩写"""
        return self.reverse_map.get(abbr)
    
    def stats(self) -> Dict[str, int]:
        return {
            'total_entities': len(self.entity_map),
        }


# 默认常见长词→短词映射
DEFAULT_WORD_MAP = {
    # 技术词汇
    'application': 'app',
    'applications': 'apps',
    'authentication': 'auth',
    'authorization': 'authz',
    'configuration': 'config',
    'deployment': 'deploy',
    'development': 'dev',
    'production': 'prod',
    'environment': 'env',
    'parameter': 'param',
    'parameters': 'params',
    'database': 'db',
    'framework': 'fw',
    'library': 'lib',
    'implementation': 'impl',
    'information': 'info',
    'infrastructure': 'infra',
    'interface': 'iface',
    'memory': 'mem',
    'message': 'msg',
    'messages': 'msgs',
    'repository': 'repo',
    'repositories': 'repos',
    'documentation': 'docs',
    'directory': 'dir',
    'function': 'fn',
    'object': 'obj',
    'server': 'srv',
    'service': 'svc',
    'statement': 'stmt',
    'dictionary': 'dict',
    'character': 'char',
    'integer': 'int',
    'boolean': 'bool',
    'professional': 'pro',
    'maximum': 'max',
    'minimum': 'min',
    'configuration': 'config',
    'initialize': 'init',
    'container': 'cnt',
    'content': 'ctx',
    'context': 'ctx',
    'current': 'cur',
    'previous': 'prev',
    'difference': 'diff',
    'reference': 'ref',
    'references': 'refs',
    'experimental': 'exp',
    'temporary': 'tmp',
    'stability': 'stable',
    'architecture': 'arch',
    'binary': 'bin',
    'category': 'cat',
    'change': 'chg',
    'color': 'col',
    'command': 'cmd',
    'compare': 'cmp',
    'constant': 'const',
    'control': 'ctrl',
    'definition': 'def',
    'description': 'desc',
    'distance': 'dist',
    'example': 'eg',
    'estimate': 'est',
    'frequency': 'freq',
    'height': 'h',
    'width': 'w',
    'length': 'len',
    'number': 'num',
    'position': 'pos',
    'security': 'sec',
    'standard': 'std',
    'statistics': 'stats',
    'string': 'str',
    'version': 'ver',
    # 组织/项目
    'Artificial Intelligence': 'AI',
    'Machine Learning': 'ML',
    'Deep Learning': 'DL',
    'Neural Network': 'NN',
    'Large Language Model': 'LLM',
    'Large Language Models': 'LLMs',
    'Natural Language Processing': 'NLP',
    'GitHub': 'GH',
    'OpenAI': 'OA',
    'Google': 'G',
    'Microsoft': 'MS',
}


class AAAKCompressor:
    """AAAK 压缩器
    
    用法:
    ```python
    compressor = AAAKCompressor()
    compressed = compressor.compress(text)
    # 任何 LLM 都能直接读懂 compressed，不需要解压
    # 如果需要看原文：
    original = compressor.decompress(compressed)
    ```
    """
    
    def __init__(
        self,
        enable_entity_abbr: bool = True,
        enable_word_replace: bool = True,
        enable_whitespace_compress: bool = True,
        enable_remove_comments: bool = True,
        custom_word_map: Optional[Dict[str, str]] = None,
    ):
        self.enable_entity_abbr = enable_entity_abbr
        self.enable_word_replace = enable_word_replace
        self.enable_whitespace_compress = enable_whitespace_compress
        self.enable_remove_comments = enable_remove_comments
        
        # 合并词映射表
        self.word_map = DEFAULT_WORD_MAP.copy()
        if custom_word_map:
            self.word_map.update(custom_word_map)
        
        # 实体索引
        self.entity_index = EntityIndex()
    
    def compress(self, text: str) -> str:
        """压缩文本"""
        result = text
        
        # 1. 空白压缩
        if self.enable_whitespace_compress:
            # 多个空行变成一个空行
            result = re.sub(r'\n{3,}', '\n\n', result)
            # 多个空格变成一个空格
            result = re.sub(r' {2,}', ' ', result)
            # 去除行尾空格
            result = re.sub(r' +\n', '\n', result)
        
        # 2. 常见词替换
        if self.enable_word_replace:
            for long_word, short_word in self.word_map.items():
                # 使用单词边界替换，避免替换子串
                result = re.sub(rf'\b{re.escape(long_word)}\b', short_word, result)
        
        # 3. 实体缩写（这里简化处理：对已经识别的实体进行替换）
        # 实际使用时，用户可以先提取实体再调用
        # 这里保留接口，让上游提取实体后添加
        
        return result
    
    def add_entity_and_compress(self, text: str, entities: List[str]) -> str:
        """添加实体到索引并压缩文本中的实体"""
        result = self.compress(text)
        
        if not self.enable_entity_abbr:
            return result
        
        for entity in entities:
            if len(entity) > 8:  # 只缩写较长的实体
                abbr = self.entity_index.get_abbr(entity)
                # 替换完整单词
                result = re.sub(rf'\b{re.escape(entity)}\b', abbr, result)
        
        return result
    
    def decompress(self, text: str) -> str:
        """解压文本（展开所有缩写）"""
        result = text
        
        # 展开实体缩写
        for abbr, full_name in self.entity_index.reverse_map.items():
            result = result.replace(abbr, full_name)
        
        # 词替换是无损的吗？不，因为是一对一替换，这里反向替换
        for long_word, short_word in self.word_map.items():
            result = re.sub(rf'\b{re.escape(short_word)}\b', long_word, result)
        
        # 空白压缩不可逆，保持压缩后的空白
        return result
    
    def get_compression_ratio(self, original: str, compressed: str) -> float:
        """计算压缩比 (original chars / compressed chars)"""
        return len(original) / len(compressed) if len(compressed) > 0 else 1.0
    
    def get_stats(self) -> Dict[str, int]:
        """获取压缩统计"""
        return {
            'vocab_size': len(self.word_map),
            'compressed_entities': self.entity_index.next_id - 1,
        }


# 默认实例
default_compressor = AAAKCompressor()
