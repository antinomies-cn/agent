import copy
import importlib
import pickle
import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _stub_module(module_name: str, **attrs):
    module = types.ModuleType(module_name)
    if module_name in {"langchain", "langchain_core", "langchain_community"}:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


sys.modules.pop("app.services.master_service", None)

_stub_module("langchain")
_stub_module(
    "langchain.agents",
    AgentExecutor=type("AgentExecutor", (), {}),
    create_openai_tools_agent=lambda *args, **kwargs: object(),
)
_stub_module("langchain.memory", ConversationBufferMemory=type("ConversationBufferMemory", (), {}))
_stub_module("langchain.schema", StrOutputParser=type("StrOutputParser", (), {}))

_stub_module("langchain_core")
_stub_module("langchain_core.messages", AIMessage=object, BaseMessage=object, SystemMessage=object)
_stub_module(
    "langchain_core.prompts",
    ChatPromptTemplate=type("ChatPromptTemplate", (), {"from_messages": staticmethod(lambda *args, **kwargs: object())}),
    MessagesPlaceholder=type("MessagesPlaceholder", (), {}),
)
_stub_module("langchain_core.language_models", BaseChatModel=type("BaseChatModel", (), {}))
_stub_module("langchain_core.outputs", ChatGeneration=object, ChatResult=object)

_stub_module("langchain_community")
_stub_module(
    "langchain_community.chat_message_histories",
    ChatMessageHistory=type("ChatMessageHistory", (), {}),
    RedisChatMessageHistory=type("RedisChatMessageHistory", (), {}),
)

_stub_module(
    "app.tools.mytools",
    astro_current_chart=object(),
    astro_my_sign=object(),
    astro_natal_chart=object(),
    astro_transit_chart=object(),
    search=object(),
    test=object(),
    vector_search=object(),
    xingpan=object(),
)

_stub_module("requests")

CustomProxyLLM = importlib.import_module("app.llm.custom_llm").CustomProxyLLM
Master = importlib.import_module("app.services.master_service").Master


@pytest.mark.parametrize("runtime_cls", [CustomProxyLLM, Master])
def test_runtime_objects_cannot_be_pickled(runtime_cls):
    obj = runtime_cls.__new__(runtime_cls)
    with pytest.raises(TypeError):
        pickle.dumps(obj)


@pytest.mark.parametrize("runtime_cls", [CustomProxyLLM, Master])
def test_runtime_objects_cannot_be_deepcopied(runtime_cls):
    obj = runtime_cls.__new__(runtime_cls)
    with pytest.raises(TypeError):
        copy.deepcopy(obj)
