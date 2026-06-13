"""
评估结果后处理脚本
用于重新处理评估过程中因API错误导致的失败数据
"""

import os
import re
import json
import asyncio
import argparse
from typing import Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# 复用 evaluate.py 的提示词模板
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


def create_async_client(api_key: str, base_url: Optional[str] = None) -> AsyncOpenAI:
    """创建异步OpenAI客户端"""
    if base_url:
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    return AsyncOpenAI(api_key=api_key)


def build_user_message(database_schema: str, question: str, evidence: str,
                       response_a: str, response_b: str) -> str:
    """构建用户消息"""
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


async def call_api_async(client: AsyncOpenAI, model: str, system_msg: str, user_msg: str,
                         max_retries: int = 3, retry_delay: float = 2.0) -> tuple[str, str]:
    """异步调用API"""
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


def save_jsonl(filepath: str, data: list[dict]):
    """保存JSONL文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def find_error_records(data: list[dict], data_type: str) -> list[dict]:
    """
    查找包含错误的记录
    返回: [{'index': 行索引, 'record': 原始记录, 'errors': 错误类型列表}]
    """
    error_records = []

    for idx, record in enumerate(data):
        errors = []

        if data_type == 'sql' and record.get('pair_type') == 'transitivity_chain':
            # 处理传递链类型
            pairs_results = record.get('pairs_results', [])
            for pair_idx, pair in enumerate(pairs_results):
                if pair.get('forward_parsed') == 'ERROR':
                    errors.append(('chain_forward', pair_idx))
                if pair.get('backward_parsed') == 'ERROR':
                    errors.append(('chain_backward', pair_idx))
        else:
            # 处理普通偏好对类型
            if record.get('forward_parsed') == 'ERROR':
                errors.append(('forward',))
            if record.get('backward_parsed') == 'ERROR':
                errors.append(('backward',))

        if errors:
            error_records.append({
                'index': idx,
                'record': record,
                'errors': errors
            })

    return error_records


async def reprocess_record(client: AsyncOpenAI, model: str, record: dict,
                           errors: list, data_type: str, eval_data: dict) -> dict:
    """重新处理单条错误记录"""
    updated_record = record.copy()

    # 获取上下文信息
    database_schema = eval_data.get('database_schema', '')
    question = eval_data.get('question', '')
    evidence = eval_data.get('evidence', '')

    if data_type == 'sql' and record.get('pair_type') == 'transitivity_chain':
        # 处理传递链
        sqls = record.get('sqls', {})
        pairs_results = [p.copy() for p in record.get('pairs_results', [])]

        tasks = []
        task_info = []

        for error_type, pair_idx in errors:
            pair = pairs_results[pair_idx]
            sql_a = sqls[pair['sql_a_key']]
            sql_b = sqls[pair['sql_b_key']]

            if error_type == 'chain_forward':
                user_msg = build_user_message(database_schema, question, evidence, sql_a, sql_b)
                tasks.append(call_api_async(client, model, SYSTEM_MESSAGE, user_msg))
                task_info.append(('chain_forward', pair_idx))
            elif error_type == 'chain_backward':
                user_msg = build_user_message(database_schema, question, evidence, sql_b, sql_a)
                tasks.append(call_api_async(client, model, SYSTEM_MESSAGE, user_msg))
                task_info.append(('chain_backward', pair_idx))

        if tasks:
            results = await asyncio.gather(*tasks)
            for (error_type, pair_idx), (raw, parsed) in zip(task_info, results):
                if error_type == 'chain_forward':
                    pairs_results[pair_idx]['forward_raw'] = raw
                    pairs_results[pair_idx]['forward_parsed'] = parsed
                else:
                    pairs_results[pair_idx]['backward_raw'] = raw
                    pairs_results[pair_idx]['backward_parsed'] = parsed

        updated_record['pairs_results'] = pairs_results

    else:
        # 处理普通偏好对
        if data_type == 'sql':
            response_a = record.get('positive_sql', '')
            response_b = record.get('negative_sql', '')
        else:  # reject
            database_schema = eval_data.get('pair', {}).get('reject_schema', database_schema)
            response_a = record.get('positive_rej', '')
            response_b = record.get('negative_rej', '')

        tasks = []
        task_info = []

        for error in errors:
            if error[0] == 'forward':
                user_msg = build_user_message(database_schema, question, evidence, response_a, response_b)
                tasks.append(call_api_async(client, model, SYSTEM_MESSAGE, user_msg))
                task_info.append('forward')
            elif error[0] == 'backward':
                user_msg = build_user_message(database_schema, question, evidence, response_b, response_a)
                tasks.append(call_api_async(client, model, SYSTEM_MESSAGE, user_msg))
                task_info.append('backward')

        if tasks:
            results = await asyncio.gather(*tasks)
            for error_type, (raw, parsed) in zip(task_info, results):
                if error_type == 'forward':
                    updated_record['forward_raw'] = raw
                    updated_record['forward_parsed'] = parsed
                else:
                    updated_record['backward_raw'] = raw
                    updated_record['backward_parsed'] = parsed

    return updated_record


async def post_process_async(file_path: str, data_type: str, eval_data_path: str,
                             model: str, batch_size: int = 8, batch_delay: float = 2.0):
    """后处理主函数"""
    api_key, base_url = load_config()
    client = create_async_client(api_key, base_url)

    try:
        # 加载结果文件
        data = load_jsonl(file_path)
        print(f"Loaded {len(data)} records from {file_path}")

        # 加载原始评估数据（用于获取 schema、question、evidence）
        eval_data_list = load_jsonl(eval_data_path)
        eval_data_map = {f"{data_type}_{i}": item for i, item in enumerate(eval_data_list)}

        # 查找错误记录
        error_records = find_error_records(data, data_type)
        print(f"Found {len(error_records)} records with errors")

        if not error_records:
            print("No errors to process.")
            return

        # 统计错误类型
        total_errors = sum(len(r['errors']) for r in error_records)
        print(f"Total error items to reprocess: {total_errors}")

        # 批量处理
        for batch_start in tqdm(range(0, len(error_records), batch_size), desc="Processing batches"):
            batch = error_records[batch_start:batch_start + batch_size]

            tasks = []
            for err_record in batch:
                record = err_record['record']
                errors = err_record['errors']
                eval_data = eval_data_map.get(record['id'], {})
                tasks.append(reprocess_record(client, model, record, errors, data_type, eval_data))

            results = await asyncio.gather(*tasks)

            # 更新原始数据
            for err_record, updated_record in zip(batch, results):
                data[err_record['index']] = updated_record

            # 批次间休息
            if batch_start + batch_size < len(error_records):
                await asyncio.sleep(batch_delay)

        # 保存更新后的文件
        save_jsonl(file_path, data)
        print(f"\nUpdated file saved to {file_path}")

        # 检查是否还有错误
        remaining_errors = find_error_records(data, data_type)
        if remaining_errors:
            print(f"Warning: {len(remaining_errors)} records still have errors. Consider running again.")
        else:
            print("All errors have been resolved.")

    finally:
        await client.close()


def post_process(file_path: str, data_type: str, eval_data_path: str,
                 model: str, batch_size: int = 8, batch_delay: float = 2.0):
    """后处理入口函数"""
    asyncio.run(post_process_async(file_path, data_type, eval_data_path,
                                   model, batch_size, batch_delay))


def main():
    parser = argparse.ArgumentParser(description='Post-process evaluation results with errors')
    parser.add_argument('--file', type=str, required=True, help='Path to the result file to process')
    parser.add_argument('--type', type=str, required=True, choices=['sql', 'reject'],
                        help='Type of ground truth (sql or reject)')
    parser.add_argument('--eval_data', type=str, required=True,
                        help='Path to original evaluation data file (pair_sql.jsonl or pair_reject.jsonl)')
    parser.add_argument('--model', type=str, required=True, help='Model name to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--batch_delay', type=float, default=2.0, help='Delay between batches (default: 2.0)')

    args = parser.parse_args()

    post_process(args.file, args.type, args.eval_data, args.model, args.batch_size, args.batch_delay)


if __name__ == '__main__':
    main()
