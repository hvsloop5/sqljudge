"""
Text2SQL奖励模型评估框架
基于SQLJudge-Bench评估方案，从三个维度评估奖励模型：
1. 严格准确率 (SA): 对称评估下的判断正确性
2. 偏好不稳定性 (PI): 位置交换后的判断翻转比例
3. 全局逻辑一致性 (GLC): 传递性公理违反程度
"""

import os
import re
import json
import time
import asyncio
import argparse
from itertools import permutations
from typing import Optional

import pandas as pd
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 提示词模板（统一用于SQL和拒绝回答评估）
SYSTEM_MESSAGE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user's Text2SQL question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better.

Your evaluation should consider the following factors:
- Whether the response appropriately addresses the question based on the given database schema and evidence
- The correctness and relevance of the response (whether it is a valid SQL query or a reasonable refusal to answer)
- The accuracy of understanding schema information and question intent
- The helpfulness and clarity of the response

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain response formats. Be as objective as possible.

Begin your evaluation by thinking through the problem step by step. Then output your final verdict by strictly following this format: <answer>A</answer> if Response A is better, and <answer>B</answer> if Response B is better."""


def load_config():
    """加载环境配置"""
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key:
        raise ValueError("API_KEY not found in .env file")
    return api_key, base_url


def create_client(api_key: str, base_url: Optional[str] = None) -> OpenAI:
    """创建OpenAI客户端"""
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def create_async_client(api_key: str, base_url: Optional[str] = None) -> AsyncOpenAI:
    """创建异步OpenAI客户端"""
    if base_url:
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    return AsyncOpenAI(api_key=api_key)


def build_user_message(database_schema: str, question: str, evidence: str,
                       response_a: str, response_b: str) -> str:
    """构建统一的用户消息（适用于SQL和拒绝回答评估）"""
    return f"""[User Question]
{question}

[Database Engine]
SQLite

[Database Schema]
{database_schema}

[Evidence]
{evidence}

[The Start of Response A]
{response_a}
[The End of Response A]

[The Start of Response B]
{response_b}
[The End of Response B]"""


def call_api(client: OpenAI, model: str, system_msg: str, user_msg: str,
             max_retries: int = 3, retry_delay: float = 2.0) -> tuple[str, str]:
    """
    调用API获取响应（同步版本）
    返回: (raw_response, parsed_answer)
    parsed_answer: "A", "B", or "TIE"
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0,
                max_tokens=2048
            )
            raw_response = response.choices[0].message.content
            parsed = parse_answer(raw_response)
            return raw_response, parsed
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return f"ERROR: {str(e)}", "ERROR"
    return "ERROR: Max retries exceeded", "ERROR"


async def call_api_async(client: AsyncOpenAI, model: str, system_msg: str, user_msg: str,
                         max_retries: int = 3, retry_delay: float = 2.0) -> tuple[str, str]:
    """
    调用API获取响应（异步版本）
    返回: (raw_response, parsed_answer)
    parsed_answer: "A", "B", or "TIE"
    """
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0,
                max_tokens=3000
            )
            raw_response = response.choices[0].message.content
            parsed = parse_answer(raw_response)
            return raw_response, parsed
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                return f"ERROR: {str(e)}", "ERROR"
    return "ERROR: Max retries exceeded", "ERROR"


def parse_answer(response: str) -> str:
    """从响应中解析答案"""
    match = re.search(r'<answer>\s*([AB])\s*</answer>', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 尝试其他模式
    if "tie" in response.lower() or "equal" in response.lower():
        return "TIE"
    return "UNKNOWN"


def load_jsonl(filepath: str) -> list[dict]:
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_progress(filepath: str) -> dict:
    """加载已处理的进度"""
    if os.path.exists(filepath):
        return {item['id']: item for item in load_jsonl(filepath)}
    return {}


def save_result(filepath: str, result: dict):
    """追加保存单条结果"""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def evaluate_pair_sql(client: OpenAI, model: str, item: dict, item_id: str) -> dict:
    """评估单个SQL偏好对（同步版本）"""
    pair = item['pair']
    pair_type = pair['type']
    result = {
        'id': item_id,
        'db_id': item.get('db_id', ''),
        'scenario_id': item.get('scenario_id', ''),
        'pair_type': pair_type
    }

    database_schema = item.get('database_schema', '')
    question = item.get('question', '')
    evidence = item.get('evidence', '')

    if pair_type in ['pos_neg_pref', 'homogenization']:
        positive_sql = pair['positive_sql']
        negative_sql = pair['negative_sql']

        # 正向评估: positive在A位置
        user_msg_forward = build_user_message(
            database_schema, question, evidence, positive_sql, negative_sql)
        raw_forward, parsed_forward = call_api(client, model, SYSTEM_MESSAGE, user_msg_forward)

        # 反向评估: negative在A位置
        user_msg_backward = build_user_message(
            database_schema, question, evidence, negative_sql, positive_sql)
        raw_backward, parsed_backward = call_api(client, model, SYSTEM_MESSAGE, user_msg_backward)

        result.update({
            'positive_sql': positive_sql,
            'negative_sql': negative_sql,
            'forward_raw': raw_forward,
            'forward_parsed': parsed_forward,  # 期望A（positive更好）
            'backward_raw': raw_backward,
            'backward_parsed': parsed_backward,  # 期望B（positive更好）
        })

    elif pair_type == 'transitivity_chain':
        # 获取所有SQL
        sqls = {}
        for key in ['sql_e', 'sql_s', 'sql_m', 'sql_x']:
            if key in pair:
                sqls[key] = pair[key]

        pairs_results = []
        for p in pair['pairs']:
            sql_a_key = p['sql_a']
            sql_b_key = p['sql_b']
            sql_a = sqls[sql_a_key]
            sql_b = sqls[sql_b_key]

            # 正向评估
            user_msg_forward = build_user_message(
                database_schema, question, evidence, sql_a, sql_b)
            raw_forward, parsed_forward = call_api(client, model, SYSTEM_MESSAGE, user_msg_forward)

            # 反向评估
            user_msg_backward = build_user_message(
                database_schema, question, evidence, sql_b, sql_a)
            raw_backward, parsed_backward = call_api(client, model, SYSTEM_MESSAGE, user_msg_backward)

            pairs_results.append({
                'pair_id': p['pair_id'],
                'sql_a_key': sql_a_key,
                'sql_b_key': sql_b_key,
                'expected': p['expected_preference'],
                'forward_raw': raw_forward,
                'forward_parsed': parsed_forward,
                'backward_raw': raw_backward,
                'backward_parsed': parsed_backward
            })

        result['pairs_results'] = pairs_results
        result['sqls'] = sqls

    return result


async def evaluate_pair_sql_async(client: AsyncOpenAI, model: str, item: dict, item_id: str) -> dict:
    """评估单个SQL偏好对（异步版本）"""
    pair = item['pair']
    pair_type = pair['type']
    result = {
        'id': item_id,
        'db_id': item.get('db_id', ''),
        'scenario_id': item.get('scenario_id', ''),
        'pair_type': pair_type
    }

    database_schema = item.get('database_schema', '')
    question = item.get('question', '')
    evidence = item.get('evidence', '')

    if pair_type in ['pos_neg_pref', 'homogenization']:
        positive_sql = pair['positive_sql']
        negative_sql = pair['negative_sql']

        # 构建消息
        user_msg_forward = build_user_message(
            database_schema, question, evidence, positive_sql, negative_sql)
        user_msg_backward = build_user_message(
            database_schema, question, evidence, negative_sql, positive_sql)

        # 并发执行正向和反向评估
        (raw_forward, parsed_forward), (raw_backward, parsed_backward) = await asyncio.gather(
            call_api_async(client, model, SYSTEM_MESSAGE, user_msg_forward),
            call_api_async(client, model, SYSTEM_MESSAGE, user_msg_backward)
        )

        result.update({
            'positive_sql': positive_sql,
            'negative_sql': negative_sql,
            'forward_raw': raw_forward,
            'forward_parsed': parsed_forward,
            'backward_raw': raw_backward,
            'backward_parsed': parsed_backward,
        })

    elif pair_type == 'transitivity_chain':
        # 获取所有SQL
        sqls = {}
        for key in ['sql_e', 'sql_s', 'sql_m', 'sql_x']:
            if key in pair:
                sqls[key] = pair[key]

        # 收集所有需要的API调用任务
        tasks = []
        task_info = []
        for p in pair['pairs']:
            sql_a_key = p['sql_a']
            sql_b_key = p['sql_b']
            sql_a = sqls[sql_a_key]
            sql_b = sqls[sql_b_key]

            user_msg_forward = build_user_message(
                database_schema, question, evidence, sql_a, sql_b)
            user_msg_backward = build_user_message(
                database_schema, question, evidence, sql_b, sql_a)

            tasks.append(call_api_async(client, model, SYSTEM_MESSAGE, user_msg_forward))
            tasks.append(call_api_async(client, model, SYSTEM_MESSAGE, user_msg_backward))
            task_info.append({
                'pair_id': p['pair_id'],
                'sql_a_key': sql_a_key,
                'sql_b_key': sql_b_key,
                'expected': p['expected_preference']
            })

        # 并发执行所有API调用
        responses = await asyncio.gather(*tasks)

        # 组装结果
        pairs_results = []
        for i, info in enumerate(task_info):
            raw_forward, parsed_forward = responses[i * 2]
            raw_backward, parsed_backward = responses[i * 2 + 1]
            pairs_results.append({
                **info,
                'forward_raw': raw_forward,
                'forward_parsed': parsed_forward,
                'backward_raw': raw_backward,
                'backward_parsed': parsed_backward
            })

        result['pairs_results'] = pairs_results
        result['sqls'] = sqls

    return result


def evaluate_pair_reject(client: OpenAI, model: str, item: dict, item_id: str) -> dict:
    """评估单个拒绝回答偏好对（同步版本）"""
    pair = item['pair']
    result = {
        'id': item_id,
        'db_id': item.get('db_id', ''),
        'scenario_id': item.get('scenario_id', ''),
        'pair_type': pair['type']
    }

    reject_schema = pair.get('reject_schema', item.get('database_schema', ''))
    question = item.get('question', '')
    evidence = item.get('evidence', '')
    positive_rej = pair['positive_rej']
    negative_rej = pair['negative_rej']

    # 正向评估: positive在A位置
    user_msg_forward = build_user_message(
        reject_schema, question, evidence, positive_rej, negative_rej)
    raw_forward, parsed_forward = call_api(client, model, SYSTEM_MESSAGE, user_msg_forward)

    # 反向评估: negative在A位置
    user_msg_backward = build_user_message(
        reject_schema, question, evidence, negative_rej, positive_rej)
    raw_backward, parsed_backward = call_api(client, model, SYSTEM_MESSAGE, user_msg_backward)

    result.update({
        'positive_rej': positive_rej,
        'negative_rej': negative_rej,
        'positive_type': pair.get('positive_type', ''),
        'negative_type': pair.get('negative_type', ''),
        'forward_raw': raw_forward,
        'forward_parsed': parsed_forward,
        'backward_raw': raw_backward,
        'backward_parsed': parsed_backward,
        'recall_rate': pair.get('recall_rate'),
        'precision_rate': pair.get('precision_rate')
    })

    return result


async def evaluate_pair_reject_async(client: AsyncOpenAI, model: str, item: dict, item_id: str) -> dict:
    """评估单个拒绝回答偏好对（异步版本）"""
    pair = item['pair']
    result = {
        'id': item_id,
        'db_id': item.get('db_id', ''),
        'scenario_id': item.get('scenario_id', ''),
        'pair_type': pair['type']
    }

    reject_schema = pair.get('reject_schema', item.get('database_schema', ''))
    question = item.get('question', '')
    evidence = item.get('evidence', '')
    positive_rej = pair['positive_rej']
    negative_rej = pair['negative_rej']

    # 构建消息
    user_msg_forward = build_user_message(
        reject_schema, question, evidence, positive_rej, negative_rej)
    user_msg_backward = build_user_message(
        reject_schema, question, evidence, negative_rej, positive_rej)

    # 并发执行正向和反向评估
    (raw_forward, parsed_forward), (raw_backward, parsed_backward) = await asyncio.gather(
        call_api_async(client, model, SYSTEM_MESSAGE, user_msg_forward),
        call_api_async(client, model, SYSTEM_MESSAGE, user_msg_backward)
    )

    result.update({
        'positive_rej': positive_rej,
        'negative_rej': negative_rej,
        'positive_type': pair.get('positive_type', ''),
        'negative_type': pair.get('negative_type', ''),
        'forward_raw': raw_forward,
        'forward_parsed': parsed_forward,
        'backward_raw': raw_backward,
        'backward_parsed': parsed_backward,
        'recall_rate': pair.get('recall_rate'),
        'precision_rate': pair.get('precision_rate')
    })

    return result


def is_correct(forward_parsed: str, backward_parsed: str) -> bool:
    """判断是否严格正确：正向选A且反向选B"""

    return forward_parsed == 'A' and backward_parsed == 'B'


def is_unstable(forward_parsed: str, backward_parsed: str) -> bool:
    """判断是否不稳定：位置交换后判断不满足反对称性"""
    # 稳定: forward=A且backward=B，或forward=B且backward=A，或两次都是TIE
    if forward_parsed == 'A' and backward_parsed == 'B':
        return False
    if forward_parsed == 'B' and backward_parsed == 'A':
        return False
    if forward_parsed == 'TIE' and backward_parsed == 'TIE':
        return False
    return True


def parse_single_result(parsed: str, is_forward: bool) -> int:
    """
    解析单次评估结果，返回获胜者标识
    返回: 1 (A位置的选项胜出), 2 (B位置的选项胜出), -1 (平局/TIE/UNKNOWN)
    """
    if parsed == 'A':
        return 1  # A位置胜出
    elif parsed == 'B':
        return 2  # B位置胜出
    else:
        return -1  # 平局或未知


def get_stable_preference(forward: str, backward: str) -> int:
    """
    获取稳定的偏好判断（用于非TOV计算场景）
    返回: +1 (A优于B), -1 (B优于A), 0 (不确定/TIE)
    """
    if forward == 'A' and backward == 'B':
        return 1
    if forward == 'B' and backward == 'A':
        return -1
    if forward == 'TIE' and backward == 'TIE':
        return 0
    return 0  # 不稳定视为不确定


def generate_weak_orders(n: int) -> list[list[list[int]]]:
    """
    生成n个元素的所有弱全序（有序分区）
    返回分组列表，每个分组内元素并列，组间有严格顺序
    n=3时有13种，n=4时有75种
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[[0]]]

    elements = list(range(n))

    def get_all_partitions(items):
        """生成所有集合分区"""
        if not items:
            return [[]]
        first = items[0]
        rest_partitions = get_all_partitions(items[1:])
        result = []
        for part in rest_partitions:
            for i in range(len(part)):
                new_part = [g[:] for g in part]
                new_part[i] = new_part[i] + [first]
                result.append(new_part)
            result.append(part + [[first]])
        return result

    partitions = get_all_partitions(elements)
    orders = []

    # 对每个分区，生成组的所有排列（有序分区）
    for part in partitions:
        for perm in permutations(part):
            orders.append(list(perm))

    return orders


def compute_tov(compare_results: list[dict], sql_keys: list[str]) -> int:
    """
    计算弱全序违反度 (TOV)
    参考 calculate-IPIandTOV.py 的计算逻辑
    compare_results: 比较结果列表，每个元素包含 sql_a_key, sql_b_key, forward_parsed, backward_parsed
    sql_keys: SQL键列表，用于生成排序
    """
    # 构建键到索引的映射
    key_to_idx = {k: i for i, k in enumerate(sql_keys)}
    n = len(sql_keys)

    min_changes = float('inf')

    for ranking in generate_weak_orders(n):
        current_changes = 0
        # 构建排名映射：元素索引 -> 排名
        rank_map = {idx: r for r, group in enumerate(ranking) for idx in group}

        for comp in compare_results:
            idx_a = key_to_idx[comp['sql_a_key']]
            idx_b = key_to_idx[comp['sql_b_key']]

            rank_a = rank_map[idx_a]
            rank_b = rank_map[idx_b]

            # 根据排名确定期望的获胜者
            # 期望：排名小的胜出（排名小表示更好）
            if rank_a < rank_b:
                expected_winner = 1  # A位置的选项应该胜出
            elif rank_b < rank_a:
                expected_winner = 2  # B位置的选项应该胜出
            else:
                expected_winner = -1  # 并列，期望平局

            # 正向评估：A位置是 sql_a，B位置是 sql_b
            forward_result = parse_single_result(comp['forward_parsed'], True)
            if forward_result != expected_winner:
                current_changes += 1

            # 反向评估：A位置是 sql_b，B位置是 sql_a
            # 反向时期望也要反转
            if rank_a < rank_b:
                expected_winner_backward = 2  # sql_a更好，但在B位置，所以期望B胜出
            elif rank_b < rank_a:
                expected_winner_backward = 1  # sql_b更好，但在A位置，所以期望A胜出
            else:
                expected_winner_backward = -1  # 并列

            backward_result = parse_single_result(comp['backward_parsed'], False)
            if backward_result != expected_winner_backward:
                current_changes += 1

        min_changes = min(min_changes, current_changes)

    return min_changes


def compute_metrics(sql_results: list[dict], reject_results: list[dict]) -> dict:
    """计算所有评估指标"""
    # 分类SQL结果
    sql_pair_results = [r for r in sql_results if r['pair_type'] in ['pos_neg_pref', 'homogenization']]
    sql_chain_results = [r for r in sql_results if r['pair_type'] == 'transitivity_chain']

    # 计算SQL偏好严格准确率和不稳定性
    sql_correct = 0
    sql_unstable = 0
    sql_pair_count = len(sql_pair_results)

    for r in sql_pair_results:
        if is_correct(r['forward_parsed'], r['backward_parsed']):
            sql_correct += 1
        if is_unstable(r['forward_parsed'], r['backward_parsed']):
            sql_unstable += 1

    sa_sql = sql_correct / sql_pair_count if sql_pair_count > 0 else 0
    pi_sql = sql_unstable / sql_pair_count if sql_pair_count > 0 else 0

    # 计算拒绝回答严格准确率和不稳定性
    ref_correct = 0
    ref_unstable = 0
    ref_count = len(reject_results)

    for r in reject_results:
        if is_correct(r['forward_parsed'], r['backward_parsed']):
            ref_correct += 1
        if is_unstable(r['forward_parsed'], r['backward_parsed']):
            ref_unstable += 1

    sa_ref = ref_correct / ref_count if ref_count > 0 else 0
    pi_ref = ref_unstable / ref_count if ref_count > 0 else 0

    # 计算整体指标
    total_pair = sql_pair_count + ref_count
    sa_all = (sql_correct + ref_correct) / total_pair if total_pair > 0 else 0
    pi_all = (sql_unstable + ref_unstable) / total_pair if total_pair > 0 else 0

    # 计算全局逻辑一致性 (GLC)
    tov_sum = 0
    chain_count = len(sql_chain_results)

    for r in sql_chain_results:
        pairs_results = r.get('pairs_results', [])
        sql_keys = list(r.get('sqls', {}).keys())

        # 使用新的 compute_tov 函数
        tov = compute_tov(pairs_results, sql_keys)
        tov_sum += tov

    glc_sql = tov_sum / chain_count if chain_count > 0 else 0

    return {
        'SA_sql': sa_sql,
        'SA_ref': sa_ref,
        'SA_all': sa_all,
        'PI_sql': pi_sql,
        'PI_ref': pi_ref,
        'PI_all': pi_all,
        'GLC_sql': glc_sql,
        'sql_pair_count': sql_pair_count,
        'sql_chain_count': chain_count,
        'ref_count': ref_count
    }


def compute_detailed_metrics(sql_results: list[dict], reject_results: list[dict]) -> pd.DataFrame:
    """按场景计算详细指标"""
    records = []

    # SQL偏好按场景分析
    sql_by_scenario = {}
    for r in sql_results:
        scenario = r.get('scenario_id', 'unknown')
        if scenario not in sql_by_scenario:
            sql_by_scenario[scenario] = []
        sql_by_scenario[scenario].append(r)

    for scenario, items in sql_by_scenario.items():
        pair_items = [r for r in items if r['pair_type'] in ['pos_neg_pref', 'homogenization']]
        chain_items = [r for r in items if r['pair_type'] == 'transitivity_chain']

        if pair_items:
            correct = sum(1 for r in pair_items if is_correct(r['forward_parsed'], r['backward_parsed']))
            unstable = sum(1 for r in pair_items if is_unstable(r['forward_parsed'], r['backward_parsed']))
            records.append({
                'category': 'SQL',
                'scenario_id': scenario,
                'type': pair_items[0]['pair_type'],
                'count': len(pair_items),
                'SA': correct / len(pair_items),
                'PI': unstable / len(pair_items),
                'GLC': None
            })

        if chain_items:
            tov_sum = 0
            for r in chain_items:
                pairs_results = r.get('pairs_results', [])
                sql_keys = list(r.get('sqls', {}).keys())

                # 使用新的 compute_tov 函数
                tov = compute_tov(pairs_results, sql_keys)
                tov_sum += tov

            records.append({
                'category': 'SQL',
                'scenario_id': scenario,
                'type': 'transitivity_chain',
                'count': len(chain_items),
                'SA': None,
                'PI': None,
                'GLC': tov_sum / len(chain_items)
            })

    # 拒绝回答按 (scenario_id, pair_type) 分析
    ref_by_group = {}
    for r in reject_results:
        key = (r.get('scenario_id', 'unknown'), r.get('pair_type', 'unknown'))
        ref_by_group.setdefault(key, []).append(r)

    for (scenario, pair_type), items in ref_by_group.items():
        correct = sum(1 for r in items if is_correct(r['forward_parsed'], r['backward_parsed']))
        unstable = sum(1 for r in items if is_unstable(r['forward_parsed'], r['backward_parsed']))
        records.append({
            'category': 'Reject',
            'scenario_id': scenario,
            'type': pair_type,
            'count': len(items),
            'SA': correct / len(items),
            'PI': unstable / len(items),
            'GLC': None
        })

    return pd.DataFrame(records)


async def run_evaluation_async(model_name: str, data_dir: str = '.', output_dir: str = '.',
                               batch_size: int = 8, batch_delay: float = 2.0):
    """运行完整评估流程（异步版本）"""
    api_key, base_url = load_config()
    client = create_async_client(api_key, base_url)

    try:
        # 输出文件路径
        sql_output = os.path.join(output_dir, f'{model_name}_sql_results.jsonl')
        reject_output = os.path.join(output_dir, f'{model_name}_reject_results.jsonl')
        metrics_output = os.path.join(output_dir, f'{model_name}_metrics.xlsx')

        # 加载数据
        sql_data = load_jsonl(os.path.join(data_dir, 'pair_sql.jsonl'))
        reject_data = load_jsonl(os.path.join(data_dir, 'pair_reject.jsonl'))

        # 加载已处理进度
        sql_progress = load_progress(sql_output)
        reject_progress = load_progress(reject_output)

        print(f"Model: {model_name}")
        print(f"Batch size: {batch_size}, Batch delay: {batch_delay}s")
        print(f"SQL data: {len(sql_data)} items, {len(sql_progress)} processed")
        print(f"Reject data: {len(reject_data)} items, {len(reject_progress)} processed")

        # 筛选未处理的SQL数据
        sql_pending = [(i, item) for i, item in enumerate(sql_data) if f"sql_{i}" not in sql_progress]

        # 批量评估SQL偏好
        print(f"\nEvaluating SQL preferences ({len(sql_pending)} pending)...")
        for batch_start in tqdm(range(0, len(sql_pending), batch_size), desc="SQL batches"):
            batch = sql_pending[batch_start:batch_start + batch_size]

            # 并发处理当前批次
            tasks = [
                evaluate_pair_sql_async(client, model_name, item, f"sql_{i}")
                for i, item in batch
            ]
            results = await asyncio.gather(*tasks)

            # 保存结果
            for result in results:
                save_result(sql_output, result)
                sql_progress[result['id']] = result

            # 批次间休息，防止API限流
            if batch_start + batch_size < len(sql_pending):
                await asyncio.sleep(batch_delay)

        # 筛选未处理的拒绝回答数据
        reject_pending = [(i, item) for i, item in enumerate(reject_data) if f"reject_{i}" not in reject_progress]

        # 批量评估拒绝回答偏好
        print(f"\nEvaluating reject preferences ({len(reject_pending)} pending)...")
        for batch_start in tqdm(range(0, len(reject_pending), batch_size), desc="Reject batches"):
            batch = reject_pending[batch_start:batch_start + batch_size]

            # 并发处理当前批次
            tasks = [
                evaluate_pair_reject_async(client, model_name, item, f"reject_{i}")
                for i, item in batch
            ]
            results = await asyncio.gather(*tasks)

            # 保存结果
            for result in results:
                save_result(reject_output, result)
                reject_progress[result['id']] = result

            # 批次间休息，防止API限流
            if batch_start + batch_size < len(reject_pending):
                await asyncio.sleep(batch_delay)

        # 计算指标
        print("\nComputing metrics...")
        sql_results = list(sql_progress.values())
        reject_results = list(reject_progress.values())

        metrics = compute_metrics(sql_results, reject_results)
        detailed_metrics = compute_detailed_metrics(sql_results, reject_results)

        # 打印指标
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"\nStrict Accuracy (SA):")
        print(f"  SA_sql:  {metrics['SA_sql']:.4f} ({metrics['sql_pair_count']} pairs)")
        print(f"  SA_ref:  {metrics['SA_ref']:.4f} ({metrics['ref_count']} pairs)")
        print(f"  SA_all:  {metrics['SA_all']:.4f}")
        print(f"\nPreference Instability (PI):")
        print(f"  PI_sql:  {metrics['PI_sql']:.4f}")
        print(f"  PI_ref:  {metrics['PI_ref']:.4f}")
        print(f"  PI_all:  {metrics['PI_all']:.4f}")
        print(f"\nGlobal Logical Consistency (GLC):")
        print(f"  GLC_sql: {metrics['GLC_sql']:.4f} ({metrics['sql_chain_count']} chains)")
        print("=" * 50)

        # 保存到Excel
        with pd.ExcelWriter(metrics_output, engine='openpyxl') as writer:
            # 总体指标
            summary_df = pd.DataFrame([{
                'Metric': 'SA_sql', 'Value': metrics['SA_sql'], 'Count': metrics['sql_pair_count']
            }, {
                'Metric': 'SA_ref', 'Value': metrics['SA_ref'], 'Count': metrics['ref_count']
            }, {
                'Metric': 'SA_all', 'Value': metrics['SA_all'], 'Count': metrics['sql_pair_count'] + metrics['ref_count']
            }, {
                'Metric': 'PI_sql', 'Value': metrics['PI_sql'], 'Count': metrics['sql_pair_count']
            }, {
                'Metric': 'PI_ref', 'Value': metrics['PI_ref'], 'Count': metrics['ref_count']
            }, {
                'Metric': 'PI_all', 'Value': metrics['PI_all'], 'Count': metrics['sql_pair_count'] + metrics['ref_count']
            }, {
                'Metric': 'GLC_sql', 'Value': metrics['GLC_sql'], 'Count': metrics['sql_chain_count']
            }])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # 详细指标
            detailed_metrics.to_excel(writer, sheet_name='By_Scenario', index=False)

        print(f"\nResults saved to {metrics_output}")
        return metrics

    finally:
        # 关闭异步客户端，防止 Windows 上的 Event loop is closed 错误
        await client.close()


def run_evaluation(model_name: str, data_dir: str = '.', output_dir: str = '.',
                   batch_size: int = 8, batch_delay: float = 2.0):
    """运行完整评估流程（入口函数）"""
    return asyncio.run(run_evaluation_async(model_name, data_dir, output_dir, batch_size, batch_delay))


def main():
    parser = argparse.ArgumentParser(description='Text2SQL Reward Model Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Model name to evaluate')
    parser.add_argument('--data_dir', type=str, default='./eval_data', help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of concurrent requests per batch (default: 8)')
    parser.add_argument('--batch_delay', type=float, default=2.0, help='Delay in seconds between batches (default: 2.0)')

    args = parser.parse_args()

    run_evaluation(args.model, args.data_dir, args.output_dir, args.batch_size, args.batch_delay)


if __name__ == '__main__':
    main()
