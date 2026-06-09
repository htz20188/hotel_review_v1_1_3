"""意图处理模块：识别、检测、扩展、HyDE 生成"""

import json
import time
from dashscope import Generation


# ── HyDE 条件触发关键词（方向 11：条件 HyDE） ───────────────────────
# 明确属性类问题：直接检索通常已足够，默认不开 HyDE
HYDE_ATTRIBUTE_KEYWORDS = [
    '早餐', '地铁', '停车', '套房', '泳池', '健身房',
    '隔音', '卫生', '价格', '房型', '位置'
]
# 宽泛体验类问题：意图模糊，HyDE 有助于扩大语义召回，默认开启
HYDE_BROAD_KEYWORDS = [
    '整体', '体验', '适合', '推荐', '怎么样', '如何',
    '值得', '满意', '入住感受', '商务', '亲子'
]


def should_use_hyde(query: str) -> bool:
    """条件 HyDE 决策：根据问题类型决定是否启用 HyDE。

    规则简单可解释：
    - 命中"明确属性关键词"（如早餐、套房、价格等）→ 不开 HyDE，直接检索即可；
    - 命中"宽泛体验关键词"（如整体、体验、推荐等）→ 开启 HyDE 扩大召回；
    - 两者都未命中（较模糊的问题）→ 默认开启 HyDE。

    注意：属性关键词优先级更高。例如"套房整体怎么样"虽含"整体/怎么样"，
    但其核心是具体房型属性，因此判为不需要 HyDE。
    """
    for kw in HYDE_ATTRIBUTE_KEYWORDS:
        if kw in query:
            return False
    for kw in HYDE_BROAD_KEYWORDS:
        if kw in query:
            return True
    return True


class IntentRecognizer:
    """意图识别器：判断问题是否需要检索知识库"""

    def __init__(self, api_key: str, model: str = "qwen-flash"):
        self.api_key = api_key
        self.model = model

    def recognize(self, query: str) -> bool:
        system_prompt = """你是广州花园酒店的意图分类器。根据用户的问题，判断是否需要检索酒店评论知识库。

分类规则：
- RETRIEVAL：问题涉及酒店的设施、服务、房间、位置、餐饮、价格、体验等具体信息，需要检索评论才能回答
- DIRECT：问候、闲聊、常识性问题等，不涉及该酒店的具体信息，可以直接回答

只回复 RETRIEVAL 或 DIRECT，不要输出任何其他内容。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        response = Generation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            result_format="message"
        )

        if response.status_code == 200:
            intent = response.output.choices[0].message.content.strip()
            return intent == "RETRIEVAL"
        else:
            raise RuntimeError(f"意图识别失败: {response.message}")


class IntentDetector:
    """意图检测器：提取房型约束与时效性需求"""

    def __init__(self, llm_client, exact_room_types: list, fuzzy_room_types: list):
        self.llm_client = llm_client
        self.exact_room_types = exact_room_types
        self.fuzzy_room_types = fuzzy_room_types

    def detect(self, query: str) -> dict:
        prompt = f"""
你是一个酒店智能客服助手，需要分析用户查询并提取关键信息。

【任务】
从用户查询中提取以下信息：
1. 房型约束：用户是否提到特定房型
2. 时效性需求：用户是否关注最新信息

【精确房型列表】
{json.dumps(self.exact_room_types, ensure_ascii=False)}

【模糊房型列表】
{json.dumps(self.fuzzy_room_types, ensure_ascii=False)}

【房型检测规则】
- 优先检测精确房型，如检测到则填入 room_type，若模棱两可或只能检测到模糊房型则视为未检测到，填入 None。填入的内容只能是【精确房型列表】中的房型名称或 None
- 如未检测到精确房型，尝试检测模糊房型，如检测到则填入 fuzzy_room_type，若模棱两可则视为未检测到，填入 None。填入的内容只能是【模糊房型列表】中的房型名称或 None
- 如都未检测到，两者均为 None

【时效性判断标准】
- clear: 用户明确提到"最近"、"今年"、"最新"、"现在"等词汇
- implied: 用户隐含关注当前现状，但未明确表达，表现弱时效性
- None: 用户未表现出时效性关注

【用户查询】
{query}

【输出格式】
严格以 JSON 格式输出：
{{
    "room_type": "花园大床房" 或 None,
    "fuzzy_room_type": "大床房" 或 None,
    "time_sensitivity": "clear" 或 "implied" 或 None
}}
"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.1)
                response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(response)
                if data['room_type'] and data['room_type'] not in self.exact_room_types:
                    data['room_type'] = None
                if data['fuzzy_room_type'] and data['fuzzy_room_type'] not in self.fuzzy_room_types:
                    data['fuzzy_room_type'] = None
                if data['time_sensitivity'] and data['time_sensitivity'] not in ['clear', 'implied']:
                    data['time_sensitivity'] = None
                return data
            except Exception as e:
                print(f"意图检测第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("意图检测失败，已返回全 None 字典")
        return {"room_type": None, "fuzzy_room_type": None, "time_sensitivity": None}


class IntentExpander:
    """意图扩展器：改写 Query 并计算权重"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def expand(self, query: str) -> list[dict] | None:
        prompt = f"""
你是广州花园酒店评论 RAG 系统的复杂 Query 理解器。

你的目标不是直接回答用户，而是把用户问题改写成更适合检索真实酒店评论的查询。

【任务】
1. 判断用户问题中的核心意图，尤其关注复杂 Query 中的多意图、比较、约束和场景信息。
2. 生成 1-3 个改写后的检索 Query，每个 Query 只聚焦一个清晰关注点。
3. 为每个改写 Query 分配权重，表示该关注点在原问题中的重要性。

【用户查询】
{query}

【复杂 Query 处理规则】
- 多意图问题：拆成多个子 Query，例如亲子、交通、早餐、房间、服务、价格等分别检索。
- 比较型问题：保留比较对象，例如“套房和普通房间区别大吗？”应改写为“套房和普通房间在空间、设施和入住体验上有什么差异？”。
- 约束型问题：保留房型、人群、时间等约束，例如“最近花园大床房隔音怎么样？”必须保留“最近”“花园大床房”“隔音”。
- 模糊体验型问题：可拆成服务、房间、交通、餐饮、性价比等主要维度，但不要发散到用户完全没有提到的方面。
- 明确单意图问题：只生成 1 个 Query，不要为了凑数量强行扩展。

【改写要求】
- 每个 Query 应该是自然、可检索的中文问句。
- 每个 Query 必须保留用户原问题中的关键限制条件。
- 不要加入用户没有表达或无法从评论中检索验证的需求，例如实时房态、今日价格、当前活动。
- 不要输出解释、分析过程或 Markdown。
- rewritten_queries 数组长度必须为 1-3。
- weight 只能使用 0.2、0.4、0.6、0.8、1.0，且总和必须等于 1.0。

【输出格式】
严格以 JSON 格式输出：
{{
    "rewritten_queries": [
        {{"query": "酒店是否适合亲子入住？", "weight": 0.4}},
        {{"query": "酒店交通是否便利？", "weight": 0.4}},
        {{"query": "酒店早餐体验怎么样？", "weight": 0.2}}
    ]
}}
"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.3)
                data = self._parse_json_response(response)
                return self._normalize_queries(data.get('rewritten_queries'), query)
            except Exception as e:
                print(f"意图扩展第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("意图扩展失败，已回退到原始 Query")
        return self._fallback_query(query)

    def _parse_json_response(self, response: str) -> dict:
        """解析模型返回，兼容 ```json 包裹和前后冗余文本。"""
        text = response.replace('```json', '').replace('```', '').strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise

    def _normalize_queries(self, queries, original_query: str) -> list[dict]:
        """清洗改写结果，保证接口稳定、权重合法、异常时可回退。"""
        if not isinstance(queries, list):
            raise TypeError("rewritten_queries 数据类型错误: 期望 list")

        cleaned = []
        seen = set()
        for item in queries:
            if not isinstance(item, dict):
                continue
            rewritten_query = str(item.get('query', '')).strip()
            if not rewritten_query or rewritten_query in seen:
                continue
            try:
                weight = float(item.get('weight', 0))
            except (TypeError, ValueError):
                weight = 0
            cleaned.append({'query': rewritten_query, 'weight': weight})
            seen.add(rewritten_query)
            if len(cleaned) == 3:
                break

        if not cleaned:
            return self._fallback_query(original_query)

        raw_weights = [max(item['weight'], 0) for item in cleaned]
        if sum(raw_weights) <= 0:
            raw_weights = [1.0 / len(cleaned)] * len(cleaned)
        weights = self._snap_weights(raw_weights)

        for item, weight in zip(cleaned, weights):
            item['weight'] = weight
        return cleaned

    def _snap_weights(self, weights: list[float]) -> list[float]:
        """将权重规整到 0.2 的倍数，并保证总和为 1.0。"""
        n = min(max(len(weights), 1), 3)
        if n == 1:
            return [1.0]

        total = sum(weights)
        normalized = [w / total for w in weights[:n]]
        units = [max(1, round(w * 5)) for w in normalized]

        while sum(units) > 5:
            idx = max(range(n), key=lambda i: units[i])
            if units[idx] > 1:
                units[idx] -= 1
            else:
                break
        while sum(units) < 5:
            idx = max(range(n), key=lambda i: normalized[i] - units[i] / 5)
            units[idx] += 1

        return [round(unit / 5, 1) for unit in units]

    def _fallback_query(self, query: str) -> list[dict]:
        return [{'query': query, 'weight': 1.0}]


class HyDEGenerator:
    """假设性回复生成器：为单个 Query 生成假设回复用于增强检索

    支持两种生成模式（方向 11：HyDE 优化）：
    - "full": 原始逻辑，生成 3 条假设评论（2 正 1 负），召回更全但延迟更高；
    - "light": 轻量模式，仅生成 1 条综合性假设评论，延迟更低。

    mode 既可在构造时指定，也可在调用 generate() 时临时覆盖。
    """

    def __init__(self, llm_client, mode: str = "full"):
        self.llm_client = llm_client
        self.mode = mode

    def generate(self, query: str, mode: str | None = None) -> list[str]:
        """生成假设性评论。mode 为 None 时使用实例默认 self.mode。"""
        effective_mode = mode or self.mode
        if effective_mode == "light":
            return self._generate_light(query)
        return self._generate_full(query)

    def _generate_light(self, query: str) -> list[str]:
        """轻量 HyDE：只生成 1 条综合性假设评论（同时含正负面信息）。"""
        prompt = f"""请基于用户问题生成 1 条可能出现在酒店评论中的综合性假设评论，\
同时包含可能的正面和负面信息，不要编造具体酒店名称，不要输出解释，只输出假设评论文本。

【用户问题】
{query}
"""
        for i in range(2):
            try:
                # 该 client 默认开启 json 模式，这里关闭以获取纯文本
                response = self.llm_client.generate(prompt, temperature=0.7, json=False)
                text = response.replace('```json', '').replace('```', '').strip()
                # 容错：若模型仍返回了 JSON，尝试抽取其中的文本
                if text.startswith('{') or text.startswith('['):
                    try:
                        data = json.loads(text)
                        if isinstance(data, dict):
                            vals = list(data.values())
                            text = str(vals[0]) if vals else text
                        elif isinstance(data, list) and data:
                            text = str(data[0])
                    except Exception:
                        pass
                if text:
                    return [text]
                raise ValueError("空的假设评论")
            except Exception as e:
                print(f"轻量假设性回复生成第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("轻量假设性回复生成失败，已返回原查询")
        return [query]

    def _generate_full(self, query: str) -> list[str]:
        prompt = f"""
你是一个酒店评论撰写者，需要为以下查询生成假设性的评论回复。

【查询】
{query}

【任务】
针对上述查询，生成3条假设性的酒店评论：
- 2条正面评论：积极评价酒店相关方面
- 1条负面评论：指出可能存在的不足

【要求】
- 每条评论50-100字
- 评论要具体、真实，包含细节
- 评论风格要像真实用户写的
- 尽量增大3条评论之间的差异性

【输出格式】
严格以 JSON 格式输出：
{{
    "hypothetical_responses": [
        "正面评论1",
        "正面评论2",
        "负面评论"
    ]
}}
"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.7)
                response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(response)
                responses = data['hypothetical_responses']
                if isinstance(responses, list):
                    return responses
                else:
                    raise TypeError(f"responses 数据类型错误")
            except Exception as e:
                print(f"假设性回复生成第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("假设性回复生成失败，已返回原查询")
        return [query]
