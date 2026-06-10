"""回复生成器：基于检索上下文生成最终回复"""

import time
from datetime import datetime
from dashscope import Generation


class ResponseGenerator:
    """回复生成器：基于检索上下文生成最终回复"""

    def __init__(self, api_key: str, model: str = "qwen-plus"):
        self.api_key = api_key
        self.model = model

    def _build_prompt(self, user_query: str, rewritten_queries=None,
                      ranked_comments=None, summaries=None,
                      need_retrieval: bool = True, today: datetime | None = None,
                      history: dict | None = None) -> str:
        # 构建上一轮对话上下文
        history_context = ""
        if history and history.get("user") and history.get("assistant"):
            history_context = f"""
【上一轮对话】
用户：{history['user']}
助手：{history['assistant']}
"""

        if not need_retrieval:
            return f"""
你是广州花园酒店的智能客服助手。

{history_context}

用户问题：{user_query}

【回复格式要求】请严格按照以下JSON格式输出：
{{
  "summary": "一句话总结核心答案",
  "key_points": ["要点1", "要点2", "要点3"],
  "details": "详细回答内容",
  "suggestions": "建议或注意事项（若无则填null）"
}}

注意：
- 如果是问候或闲聊，友好回应
- 如果是通用问题，给出简洁准确的回答
- 如果用户的问题是对上一轮对话的追问，请结合上下文理解用户意图
- 语气亲切专业
- 只输出JSON，不要有其他内容
"""

        if not today:
            today = datetime.today()
        date = f"{today.year}年{today.month}月{today.day}日"

        # 构建改写 Query 上下文
        queries_context = ""
        if rewritten_queries:
            queries_context += "【问题解析】\n系统识别到用户可能关注以下方面：\n"
            queries_context += "\n".join(
                [f"- {q['query']}（意图权重为{q['weight']}）" for q in rewritten_queries]
            )
            queries_context += "\n注意：权重信息是用来帮助你区分意图主次的，**不得**向用户输出权重相关信息。"

        # 构建评论上下文
        if ranked_comments:
            comments_context = "【相关用户评论】\n"
            for i, c in enumerate(ranked_comments, 1):
                comments_context += f"""
【评论{i}】
评分: {c['metadata']['score']}（满分5分）
发布日期: {c['metadata']['publish_date']}
评论文本: {c['comment']}
点赞数: {c['metadata']['useful_count']}
评论数: {c['metadata']['review_count']}
房型: {c['metadata']['room_type']}
质量评分: {c['metadata']['quality_score']}
类别: {c['metadata'].get('categories', [])}
"""
        else:
            comments_context = "【未检索到相关用户评论】\n"

        # 构建摘要上下文
        summaries_context = ""
        if summaries:
            summaries_context += "【相关评论摘要】\n"
            for s in summaries:
                summaries_context += f"""
【{s['metadata'].get('category','未分类')}类别摘要】
关键词: {s['metadata']['keywords']}
摘要: {s['summary']}
"""
            summaries_context += """
注意：评论摘要是用来给到你更丰富的概览信息的，但用户只能看到【相关用户评论】的引用而看不到摘要的引用，因此在回复中你可以给出摘要中的模糊信息，\
但**不得过于精确因为用户无法溯源**，也**不得告诉用户你引用了摘要**，**更不得将其当作评论引用输出"评论x"**。若摘要中的信息与用户问题无关，直接忽略即可，**不需要**做出任何额外说明。
"""

        return f"""
你是广州花园酒店的智能客服助手，需要基于用户评论为用户提供准确、高质量、有帮助、简洁的回答。

今天是：{date}

{history_context}

用户问题：{user_query}

{queries_context}

{comments_context}

{summaries_context}

## 回答要求

### 1. 内容要求
- 综合以上评论信息，给出客观、全面的回答
- 如有正面和负面评价，都要提及，保持客观
- 不得大段照抄用户评论，要进行思考提炼总结
- 如果评论信息不足以回答问题，诚实说明
- 不得出现幻觉回复（如帮用户查询酒店实时信息）

### 2. 引用规范
- 引用具体评论时使用 [[ref:N]]（N为评论序号）
- 最多同时引用3条：[[ref:1,3,5]] 或 [[ref:1,3]]等
- 归纳总结时无需标注引用

### 3. 结构化输出格式
**请严格按照以下JSON格式输出，不要有任何额外内容：**

{{
  "summary": "一句话总结核心答案（30字以内）",
  "positive": ["正面观点1", "正面观点2"],
  "negative": ["负面观点1（若无则填[]）"],
  "details": "详细回答内容，可以多段落，使用适当换行",
  "suggestions": "给用户的建议（若无则填null）",
  "confidence": "high/medium/low"
}}

### 4. 输出示例

用户问："酒店的早餐怎么样？"

{{
  "summary": "早餐整体评价较好，品种丰富但价格略高",
  "positive": ["品种丰富，有中西式选择", "味道不错，特别是现煮面档"],
  "negative": ["价格偏高（128元/位）", "高峰期需要排队"],
  "details": "根据近3个月的评论，多数住客对早餐表示满意。自助早餐提供中式点心、西式面包和现煮面档，品质较好。\\n\\n需要注意的是，早餐价格为128元/位，部分住客认为性价比一般。建议错峰用餐，避免8:30-9:30高峰期。",
  "suggestions": "建议提前在APP购买早餐券，比现场便宜20元",
  "confidence": "high"
}}

现在请生成回答：
"""

    def _call_kwargs(self, prompt: str, temperature: float = 0.7) -> dict:
        return dict(
            api_key=self.api_key,
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            result_format="message",
            stream=True,
            incremental_output=True
        )

    def generate(self, user_query: str, rewritten_queries=None, ranked_comments=None,
                 summaries=None, need_retrieval: bool = True, print_response: bool = True,
                 today: datetime | None = None, history: dict | None = None) -> tuple[str, float, float, float]:
        start_time = time.time()
        prompt = self._build_prompt(user_query, rewritten_queries, ranked_comments,
                                    summaries, need_retrieval, today, history)

        completion = Generation.call(**self._call_kwargs(prompt))

        response_content = ""
        ttft_model = 0
        subsequent_time = 0
        first_token_time = 0

        for chunk in completion:
            if chunk.status_code != 200:
                raise RuntimeError(f"回复生成失败: {chunk.message}")

            message = chunk.output.choices[0].message
            if message.content:
                if not ttft_model:
                    ttft_model = time.time() - start_time
                    first_token_time = time.time()
                if print_response:
                    print(message.content, end="", flush=True)
                response_content += message.content

        if print_response and response_content:
            print()

        if ttft_model:
            subsequent_time = time.time() - first_token_time

        generation_time = time.time() - start_time
        return response_content, ttft_model, subsequent_time, generation_time

    def generate_stream(self, user_query: str, rewritten_queries=None, ranked_comments=None,
                        summaries=None, need_retrieval: bool = True,
                        today: datetime | None = None, history: dict | None = None):
        prompt = self._build_prompt(user_query, rewritten_queries, ranked_comments,
                                    summaries, need_retrieval, today, history)

        completion = Generation.call(**self._call_kwargs(prompt))

        for chunk in completion:
            if chunk.status_code != 200:
                raise RuntimeError(f"回复生成失败: {chunk.message}")

            message = chunk.output.choices[0].message
            if message.content:
                yield message.content
