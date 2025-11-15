# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import logging
import os
import random
import re

import requests
from openai import OpenAI
from PIL import Image

from math_verify import parse, verify

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


openai_api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
# 默认给一个镜像基础 URL；你也可以改成自己常用的那个
openai_api_base_list = [
    os.environ.get("LLM_AS_A_JUDGE_BASE", "EMPTY"),
]

client_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
        timeout=3600,
    )
    client_list.append(client)

# 模型名从环境变量读取，默认 gpt-4o-mini
model_name_list = [
    os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")
] * len(client_list)


# =========================
# 模板 & 工具函数（基本照搬原始代码）
# =========================
def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str, ground_truth, question):
    """
    对应你原始代码里的 get_prompt，用于 VQA / 一般任务的 1/0 Judgement 方案。
    """
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"
    return full_prompt


# =========================
# 通用（带 tool 奖励）的 score —— 对应原来的 compute_score
# =========================
def compute_score_common(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    # pattern = re.compile(r'<\|im_start\|>assistant(.*?)$', re.DOTALL)  # 匹配最后一个 target 后的所有内容
    # match = pattern.search(predict_str)
    # if match:
    #     answer_text = match.group(1).strip()
    #     print(f'DEBUG{answer_text=}')
    # else:
    #     answer_text = ""

    question_text = extra_info['question']
    full_prompt = get_prompt(answer_text, ground_truth, question_text)

    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        seed = random.randint(0, 1000000),
        temperature=0.3,
    )
    response = chat_response.choices[0].message.content.strip()
    # print(response)
    if 'Judgement:' in response:
        response = response.split('Judgement:')[-1].strip()
        if '1' in response:
            acc_reward = 1.0
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        if response == '1':
            acc_reward = 1.0
        elif response == '0':
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0

    return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward


# =========================
# 数学相关：rule + 生成式双重验证
# =========================
def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)

""" def rule_math_verify(ground_truth, model_answer):
    try:
        gold = parse(ground_truth, parsing_timeout=None)
        answer = parse(model_answer, parsing_timeout=None)
        return verify(gold, answer)
    except Exception as e:
        print(f"[MathVerify] error: {e} | gt={ground_truth!r} | ans={model_answer!r}")
        return False """

""" def rule_math_verify(ground_truth: str, model_answer: str) -> bool:
    # 可选：提前过滤空字符串
    if not ground_truth or not model_answer:
        return False

    try:
        # 1) 解析时关闭 parse 的内部 timeout（避免 signal）
        gold = parse(
            ground_truth,
            parsing_timeout=None,
            raise_on_error=False,   # 0.8.0 新增，可选，避免内部再抛异常
        )
        answer = parse(
            model_answer,
            parsing_timeout=None,
            raise_on_error=False,
        )

        # 2) 比较时也关闭 verify 的 timeout（同理避免 signal）
        return bool(
            verify(
                gold,
                answer,
                verification_timeout=None,  # 关键：关掉 verify 的 signal timeout
                raise_on_error=False,
            )
        )
    except Exception as e:
        print(f"[MathVerify] error: {e} | gt={ground_truth!r} | ans={model_answer!r}")
        return False """


def generative_verify(query, ground_truth, model_answer):
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    for it in range(8):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR math] generative_verify error: {e}')
            continue
    
    judgement = response.split('## Equivalence Judgement')[-1].lower()
    if 'true' in judgement and 'false' not in judgement:
        return True
    elif 'false' in judgement and 'true' not in judgement:
        return False
    else:
        print(f' [ERROR math] verify bug output: ')


def compute_score_math(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    model_answer = ""
    predict_no_think = predict_str.split('</think>')[-1].strip()
    answer_pattern = r'\\boxed{([^}]+)}'
    answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
    if len(answer_list) == 0:
        acc_reward = 0.0
        is_format_error = True
    else:
        if len(answer_list) > 1:
            is_format_error = True

        model_answer = answer_list[-1]
        if rule_math_verify(ground_truth, model_answer):
            acc_reward = 1.0
        else:
            acc_reward = 1.0 if generative_verify(extra_info['question'], ground_truth, model_answer) else 0.0
    
    format_reward = -1.0 if is_format_error else 0.0
    #print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {model_answer=}, {acc_reward=}, {format_reward=}')
    return 1.2 * acc_reward + 0.4 * format_reward

##入口
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, *args, **kwargs) -> float:
    if data_source in ['vstar', 'chart']:
        res = compute_score_common(solution_str, ground_truth, extra_info)

    elif data_source in ['thinklite_eureka']:
        res = compute_score_math(solution_str, ground_truth, extra_info)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


if __name__ == "__main__":
    pass

