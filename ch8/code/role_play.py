'''2.4.1 手工编写代码-角色扮演 (Gemini 版本)'''

import os
from camel.societies import RolePlaying
from camel.types import ModelPlatformType, RoleType
from camel.models import ModelFactory
from camel.messages import BaseMessage

# 从环境变量中读取 GEMINI_API_KEY
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("未找到环境变量 GEMINI_API_KEY。请设置该变量。")

# 创建 Gemini 模型实例
model = ModelFactory.create(
    # <-- 1. 主要修改：将 model_type 更新为最新的模型名称
    model_platform=ModelPlatformType.GEMINI,
    model_type="gemini-2.5-flash", 
    api_key=api_key,
    model_config_dict={"temperature": 0.2}
)

# 定义角色和任务
ASSISTANT_ROLE_NAME = "Python Programmer"
USER_ROLE_NAME = "Stock Trader"
TASK_PROMPT = "Develop a trading bot for the stock market"

# RolePlaying 的设置与原版保持一致
role_play_session = RolePlaying(
    assistant_role_name=ASSISTANT_ROLE_NAME,
    assistant_agent_kwargs=dict(model=model),
    user_role_name=USER_ROLE_NAME,
    user_agent_kwargs=dict(model=model),
    task_prompt=TASK_PROMPT,
    with_task_specify=True,
    task_specify_agent_kwargs=dict(model=model)
)

# 对话循环
n = 0
chat_turn_limit = 50

# 创建一个初始消息对象
input_assistant_msg = BaseMessage(
    role_name=USER_ROLE_NAME,
    role_type=RoleType.USER,
    meta_dict=None,
    content=TASK_PROMPT,
)


while n < chat_turn_limit:
    # 获取两个智能体的新一轮输出
    assistant_response, user_response = role_play_session.step(input_assistant_msg)

    # 检查对话是否终止
    if assistant_response.terminated or user_response.terminated:
        print("One of the agents terminated the conversation.")
        break

    # 打印角色扮演的对话内容
    if user_response.msg and user_response.msg.content:
        print(f"AI User ({user_response.msg.role_name}):\n{user_response.msg.content}\n")
    if assistant_response.msg and assistant_response.msg.content:
        print(f"AI Assistant ({assistant_response.msg.role_name}):\n{assistant_response.msg.content}\n")

    # 根据用户智能体的反馈判断任务是否完成
    if "CAMEL_TASK_DONE" in user_response.msg.content:
        break

    # 更新下一轮的输入
    input_assistant_msg = assistant_response.msg
    n += 1