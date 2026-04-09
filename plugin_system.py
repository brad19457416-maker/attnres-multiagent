"""
🔌 插件化工具系统
===

借鉴 Hermes Agent 思想：
- 所有工具都是插件，核心保持精简
- 插件可插拔，容易扩展
- 支持配置文件加载
- 支持启用/禁用插件

HGARN 插件系统设计：
- 插件 = 工具 + 配置 + 权限
- 每个插件有独立 schema 描述
- 支持等级化配置：全局 → wing → room → 项目
"""

import json
import importlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Type
from pathlib import Path


@dataclass
class PluginParameter:
    """插件参数描述"""
    name: str
    description: str
    type: str = "string"  # string, number, boolean, array, object
    required: bool = False
    default: Optional[Any] = None


@dataclass
class PluginSchema:
    """插件schema描述"""
    name: str
    description: str
    parameters: List[PluginParameter] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = ""
    license: str = "MIT"


@dataclass
class Plugin:
    """插件实例"""
    schema: PluginSchema
    module: Any
    execute: Callable
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    def execute_call(self, **kwargs) -> Any:
        """执行插件"""
        return self.execute(**kwargs, config=self.config)


class PluginManager:
    """插件管理器
    
    支持：
    - 从目录加载插件
    - 启用/禁用插件
    - 获取插件schema
    - 执行插件
    - 多级配置合并（全局 → wing → room）
    """
    
    def __init__(self, plugin_dir: str = "./plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, Plugin] = {}
        self._load_default_plugins()
    
    def _load_default_plugins(self):
        """加载内置默认插件"""
        # 创建插件目录
        self.plugin_dir.mkdir(exist_ok=True, parents=True)
    
    def load_plugin_from_file(self, module_path: str, plugin_name: Optional[str] = None) -> Plugin:
        """从文件加载插件"""
        path = Path(module_path)
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        module = importlib.util.module_from_spec(spec)
        assert spec is not None
        assert spec.loader is not None
        spec.loader.exec_module(module)
        
        # 获取插件schema
        if not hasattr(module, "plugin_schema"):
            raise ValueError(f"Plugin {module_path} must export 'plugin_schema'")
        if not hasattr(module, "execute"):
            raise ValueError(f"Plugin {module_path} must export 'execute' function")
        
        schema = module.plugin_schema
        if plugin_name is None:
            plugin_name = schema.name
        
        plugin = Plugin(
            schema=schema,
            module=module,
            execute=module.execute,
            enabled=True,
        )
        
        self.plugins[plugin_name] = plugin
        return plugin
    
    def load_plugins_from_directory(self, directory: str) -> List[Plugin]:
        """从目录加载所有 *.py 插件"""
        loaded = []
        dir_path = Path(directory)
        for file in dir_path.glob("*.py"):
            if file.name.startswith("_"):
                continue  # 跳过私有模块
            try:
                plugin = self.load_plugin_from_file(str(file))
                loaded.append(plugin)
            except Exception as e:
                print(f"Failed to load plugin {file}: {e}")
        return loaded
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """获取插件"""
        plugin = self.plugins.get(name)
        if plugin is None or not plugin.enabled:
            return None
        return plugin
    
    def list_plugins(self) -> List[Plugin]:
        """列出所有启用的插件"""
        return [p for p in self.plugins.values() if p.enabled]
    
    def list_all_plugins(self) -> List[Plugin]:
        """列出所有插件，包括禁用的"""
        return list(self.plugins.values())
    
    def enable_plugin(self, name: str) -> bool:
        """启用插件"""
        if name in self.plugins:
            self.plugins[name].enabled = True
            return True
        return False
    
    def disable_plugin(self, name: str) -> bool:
        """禁用插件"""
        if name in self.plugins:
            self.plugins[name].enabled = False
            return True
        return False
    
    def set_plugin_config(self, name: str, config: Dict[str, Any]) -> bool:
        """设置插件配置"""
        if name in self.plugins:
            self.plugins[name].config.update(config)
            return True
        return False
    
    def get_enabled_schemas(self) -> List[PluginSchema]:
        """获取所有启用插件的schema"""
        return [p.schema for p in self.plugins.values() if p.enabled]
    
    def execute_plugin(self, name: str, **kwargs) -> Any:
        """执行插件"""
        plugin = self.get_plugin(name)
        if plugin is None:
            raise ValueError(f"Plugin {name} not found or disabled")
        return plugin.execute_call(**kwargs)
    
    def export_openai_schemas(self) -> List[Dict[str, Any]]:
        """导出为 OpenAI Function Calling 格式"""
        schemas = []
        for plugin in self.list_plugins():
            parameters = {
                "type": "object",
                "properties": {},
                "required": [],
            }
            for param in plugin.schema.parameters:
                parameters["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.default is not None:
                    parameters["properties"][param.name]["default"] = param.default
                if param.required:
                    parameters["required"].append(param.name)
            
            schemas.append({
                "type": "function",
                "function": {
                    "name": plugin.schema.name,
                    "description": plugin.schema.description,
                    "parameters": parameters,
                }
            })
        return schemas


# ========== 内置基础插件 ==========

def create_builtin_plugins() -> List[Plugin]:
    """创建内置基础插件"""
    from builtins import list as builtin_list
    
    # 内置计算器插件
    class Calculator:
        plugin_schema = PluginSchema(
            name="calculator",
            description="Calculate mathematical expressions, useful for getting accurate results",
            version="1.0.0",
            parameters=[
                PluginParameter(
                    name="expression",
                    description="The mathematical expression to calculate",
                    type="string",
                    required=True,
                )
            ]
        )
        
        @staticmethod
        def execute(expression: str, config: dict) -> str:
            try:
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"
    
    # 内置文件读取插件
    class ReadFile:
        plugin_schema = PluginSchema(
            name="read_file",
            description="Read content from a file",
            version="1.0.0",
            parameters=[
                PluginParameter(
                    name="path",
                    description="Path to the file",
                    type="string",
                    required=True,
                )
            ]
        )
        
        @staticmethod
        def execute(path: str, config: dict) -> str:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except Exception as e:
                return f"Error reading file: {e}"
    
    # 内置文件列表插件
    class ListDirectory:
        plugin_schema = PluginSchema(
            name="list_directory",
            description="List files and directories in a path",
            version="1.0.0",
            parameters=[
                PluginParameter(
                    name="path",
                    description="Path to list",
                    type="string",
                    required=True,
                )
            ]
        )
        
        @staticmethod
        def execute(path: str, config: dict) -> str:
            try:
                p = Path(path)
                if not p.exists():
                    return f"Path {path} does not exist"
                if p.is_file():
                    return f"It's a file: {path}, size: {p.stat().st_size} bytes"
                items = list(p.iterdir())
                output = [f"Directory: {path}"]
                output.append(f"Found {len(items)} items:")
                for item in items:
                    type_str = "d" if item.is_dir() else "f"
                    output.append(f"  [{type_str}] {item.name}")
                return "\n".join(output)
            except Exception as e:
                return f"Error listing directory: {e}"
    
    return [
        Plugin(schema=Calculator.plugin_schema, module=Calculator, execute=Calculator.execute),
        Plugin(schema=ReadFile.plugin_schema, module=ReadFile, execute=ReadFile.execute),
        Plugin(schema=ListDirectory.plugin_schema, module=ListDirectory, execute=ListDirectory.execute),
    ]


# 默认插件管理器
default_manager = PluginManager()
# 加载内置插件
for plugin in create_builtin_plugins():
    default_manager.plugins[plugin.schema.name] = plugin
