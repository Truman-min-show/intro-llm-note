import os
import google.generativeai as genai

def construct_prompt(other_agents_histories, question, current_round):
    """
    根据其他智能体的历史回答，为当前智能体构造新的提示。
    """
    # 如果是第一轮，智能体之间没有历史回答可参考
    if current_round == 0:
        return f"""Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""

    # 从第二轮开始，构造包含其他智能体答案的提示
    prefix_string = "These are the solutions to the problem from other agents: "
    for history in other_agents_histories:
        # 提取其他智能体在上一轮的回答
        # history[-1] 是最新的内容，即上一轮的 'assistant'/'model' 回答
        agent_response = history[-1]['content']
        response = f"\n\n One agent solution: ```{agent_response}```"
        prefix_string += response

    prefix_string += f"""\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {question}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."""
    return prefix_string

# --- 配置 ---
# 从环境变量中读取 API 密钥
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("未找到环境变量 GEMINI_API_KEY。请设置该变量。")

genai.configure(api_key=api_key)

# --- 辩论设置 ---
agents = 3  # 指定参与的智能体个数
rounds = 2  # 指定迭代轮次上限
question = "Jimmy has $2 more than twice the money Ethel has. If Ethal has $8, how much money is Jimmy having?"  # 用户提出问题

# 为每一个智能体初始化对话历史
agent_histories = [[] for _ in range(agents)]

# 选择模型
model = genai.GenerativeModel('gemini-2.5-pro')

# --- 主辩论循环 ---
for round_num in range(rounds):
    print(f"--- Round {round_num + 1} ---")
    for i in range(agents):
        # 获取除自己以外，其他智能体的历史发言
        other_agents_histories = agent_histories[:i] + agent_histories[i+1:]

        # 构造本次对话的提示
        prompt = construct_prompt(other_agents_histories, question, round_num)

        # 将当前提示（用户角色）添加到当前智能体的历史记录中
        # Gemini 的历史记录需要是 user/model 交替的
        current_turn_context = agent_histories[i] + [{"role": "user", "content": prompt}]
        
        # 将 role: 'assistant' 转换为 role: 'model' 以符合 Gemini 的要求
        gemini_context = []
        for msg in current_turn_context:
            role = 'model' if msg['role'] == 'assistant' else 'user'
            gemini_context.append({'role': role, 'parts': [msg['content']]})

        # 进行发言
        response = model.generate_content(gemini_context)
        content = response.text

        # 将用户提示和模型回答都添加到当前智能体的历史记录中
        agent_histories[i].append({"role": "user", "content": prompt})
        agent_histories[i].append({"role": "assistant", "content": content}) # 使用 'assistant' 以保持内部逻辑一致

        print(f"Agent {i+1}'s Answer:\n{content}\n")
