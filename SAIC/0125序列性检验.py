import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
import time
import re
import datetime
import json
from collections import defaultdict

# 加载环境变量
load_dotenv()


# ====================== 枚举定义 ======================

class ClassificationStep(Enum):
    """分类步骤枚举"""
    INITIAL = "initial"  # 初始状态
    ATOMICITY_CHECK = "atomicity_check"  # 原子性检查
    CONDITIONAL_CHECK = "conditional_check"  # 条件性检查（新增）
    USER_ACTION_CHECK = "user_action_check"  # 用户动作检查
    SYSTEM_CONDITION_CHECK = "system_condition_check"  # 系统条件检查
    FINAL_CLASSIFICATION = "final_classification"  # 最终分类


class RequirementType(Enum):
    """需求类型枚举"""
    COMPOSITE = "composite"  # 复合型
    CONDITIONAL = "conditional"  # 条件型（新增）
    INTERACTIVE = "interactive"  # 交互型
    SEQUENTIAL = "sequential"  # 顺序型
    STRUCTURAL = "structural"  # 结构型
    UNKNOWN = "unknown"  # 未知


class ActionType(Enum):
    """步骤动作类型"""
    CHECK_ATOMICITY = "check_atomicity"
    CHECK_CONDITIONAL = "check_conditional"  # 新增
    CHECK_USER_ACTION = "check_user_action"
    CHECK_SYSTEM_CONDITION = "check_system_condition"
    MAKE_FINAL_CLASSIFICATION = "make_final_classification"
    REQUEST_CLARIFICATION = "request_clarification"


# ====================== 数据类 ======================

@dataclass
class ClassificationState:
    """分类过程状态"""
    step: ClassificationStep  # 当前步骤
    requirement_text: str  # 原始需求文本
    intermediate_results: Dict[str, Any]  # 中间结果
    history: List[Dict]  # 历史步骤记录
    is_complete: bool = False  # 是否完成
    final_classification: Optional[RequirementType] = None  # 最终分类

    def to_dict(self):
        return {
            "step": self.step.value,
            "requirement": self.requirement_text[:100] + "..." if len(
                self.requirement_text) > 100 else self.requirement_text,
            "intermediate_results": self.intermediate_results,
            "history_length": len(self.history),
            "is_complete": self.is_complete,
            "final_classification": self.final_classification.value if self.final_classification else None
        }


@dataclass
class StepDecision:
    """步骤决策结果"""
    step: ClassificationStep
    action: ActionType
    decision: bool  # True/False 决策结果
    confidence: float  # 置信度 0.0-1.0
    reasoning: str  # 推理过程
    evidence: List[str]  # 支持证据（从需求文本中提取）
    timestamp: str

    def to_dict(self):
        return {
            "step": self.step.value,
            "action": self.action.value,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning[:200] + "..." if len(self.reasoning) > 200 else self.reasoning,
            "evidence": self.evidence,
            "timestamp": self.timestamp
        }


@dataclass
class ClassificationEpisode:
    """完整分类过程记录"""
    requirement_id: str
    requirement_text: str
    steps: List[StepDecision]
    final_classification: RequirementType
    ground_truth: Optional[RequirementType] = None
    total_reward: float = 0.0
    step_rewards: List[float] = None
    efficiency_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.step_rewards is None:
            self.step_rewards = []
        if self.efficiency_metrics is None:
            self.efficiency_metrics = {}

    def add_step(self, step_decision: StepDecision, step_reward: float):
        self.steps.append(step_decision)
        self.step_rewards.append(step_reward)

    def to_dict(self):
        return {
            "requirement_id": self.requirement_id,
            "requirement_text": self.requirement_text[:150] + "..." if len(
                self.requirement_text) > 150 else self.requirement_text,
            "steps_count": len(self.steps),
            "final_classification": self.final_classification.value,
            "ground_truth": self.ground_truth.value if self.ground_truth else None,
            "total_reward": self.total_reward,
            "step_rewards": self.step_rewards,
            "is_correct": self.ground_truth == self.final_classification if self.ground_truth else None,
            "efficiency": self.efficiency_metrics
        }


# ====================== 强化学习智能体 ======================

class ClassificationAgent:
    """需求分类智能体 - 执行多步骤决策"""

    def __init__(self, llm_client, reward_calculator=None):
        self.llm_client = llm_client
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.state: Optional[ClassificationState] = None
        self.current_episode: Optional[ClassificationEpisode] = None

        # 定义决策树状态转移 - 修改后的流程
        self.decision_tree = {
            ClassificationStep.INITIAL: {
                "next_step": ClassificationStep.ATOMICITY_CHECK,
                "action": ActionType.CHECK_ATOMICITY
            },
            ClassificationStep.ATOMICITY_CHECK: {
                "decision_false": {
                    "classification": RequirementType.COMPOSITE,
                    "is_terminal": True
                },
                "decision_true": {
                    "next_step": ClassificationStep.USER_ACTION_CHECK,
                    "action": ActionType.CHECK_USER_ACTION
                }
            },
            ClassificationStep.USER_ACTION_CHECK: {
                "decision_true": {
                    "classification": RequirementType.INTERACTIVE,
                    "is_terminal": True
                },
                "decision_false": {
                    "next_step": ClassificationStep.SYSTEM_CONDITION_CHECK,
                    "action": ActionType.CHECK_SYSTEM_CONDITION
                }
            },
            ClassificationStep.SYSTEM_CONDITION_CHECK: {
                "decision_true": {
                    "classification": RequirementType.SEQUENTIAL,
                    "is_terminal": True
                },
                "decision_false": {
                    "next_step": ClassificationStep.CONDITIONAL_CHECK,  # 放到最后
                    "action": ActionType.CHECK_CONDITIONAL
                }
            },
            ClassificationStep.CONDITIONAL_CHECK: {  # 最后检查条件型
                "decision_true": {
                    "classification": RequirementType.CONDITIONAL,  # 条件型需求
                    "is_terminal": True
                },
                "decision_false": {
                    "classification": RequirementType.STRUCTURAL,  # 如果都不是，就是结构型
                    "is_terminal": True
                }
            }
        }

    def reset(self, requirement_text: str, requirement_id: str = None):
        """重置智能体状态"""
        if requirement_id is None:
            requirement_id = f"req_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(requirement_text) % 10000}"

        self.state = ClassificationState(
            step=ClassificationStep.INITIAL,
            requirement_text=requirement_text,
            intermediate_results={},
            history=[]
        )

        self.current_episode = ClassificationEpisode(
            requirement_id=requirement_id,
            requirement_text=requirement_text,
            steps=[],
            final_classification=RequirementType.UNKNOWN
        )

        return self.state

    def step(self) -> Tuple[StepDecision, ClassificationState, bool]:
        """执行一步决策"""
        if self.state.is_complete:
            return None, self.state, True

        # 获取当前步骤配置
        current_config = self.decision_tree.get(self.state.step, {})

        if not current_config:
            raise ValueError(f"未知步骤: {self.state.step}")

        # 执行当前步骤的决策
        step_decision = self._execute_step_decision(self.state.step)

        # 记录步骤
        self.state.history.append(step_decision.to_dict())
        self.current_episode.add_step(step_decision, 0.0)

        # 处理决策结果，更新状态
        is_terminal = self._process_decision_result(step_decision)

        # 如果是终止状态，计算最终奖励
        if is_terminal:
            self.state.is_complete = True
            self.current_episode.final_classification = self.state.final_classification

            # 计算最终奖励
            if self.reward_calculator:
                total_reward, step_rewards = self.reward_calculator.calculate_episode_reward(
                    self.current_episode
                )
                self.current_episode.total_reward = total_reward
                self.current_episode.step_rewards = step_rewards

        return step_decision, self.state, is_terminal

    def _execute_step_decision(self, step: ClassificationStep) -> StepDecision:
        """执行具体的步骤决策"""

        # 构建针对当前步骤的提示词
        prompt = self._build_step_prompt(step, self.state.requirement_text)

        # 调用LLM进行决策
        response = self.llm_client.query_decision(prompt, step)

        # 解析响应
        decision, confidence, reasoning, evidence = self._parse_llm_response(
            response, step
        )

        # 创建决策记录
        current_config = self.decision_tree[step]
        action = current_config.get("action",
                                    ActionType.MAKE_FINAL_CLASSIFICATION if step == ClassificationStep.FINAL_CLASSIFICATION
                                    else ActionType.CHECK_ATOMICITY
                                    )

        return StepDecision(
            step=step,
            action=action,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=datetime.datetime.now().strftime("%H:%M:%S")
        )

    def _process_decision_result(self, step_decision: StepDecision) -> bool:
        """处理决策结果，更新状态并返回是否为终止状态"""
        current_config = self.decision_tree.get(self.state.step, {})

        # 根据决策结果确定下一步
        decision_key = "decision_true" if step_decision.decision else "decision_false"

        if decision_key in current_config:
            next_config = current_config[decision_key]

            if next_config.get("is_terminal", False):
                # 终止状态，设置最终分类
                self.state.final_classification = next_config["classification"]
                self.state.step = ClassificationStep.FINAL_CLASSIFICATION
                return True
            else:
                # 非终止状态，转移到下一步
                self.state.step = next_config["next_step"]
                return False
        else:
            # 默认情况下继续到下一步或终止
            if "next_step" in current_config:
                self.state.step = current_config["next_step"]
                return False
            else:
                # 没有下一步，设置为终止
                self.state.is_complete = True
                return True

    def _build_step_prompt(self, step: ClassificationStep, requirement_text: str) -> str:
        """构建步骤特定的提示词"""

        base_prompt = f"""你是一位软件需求分析专家。请分析以下需求描述：

需求描述："{requirement_text}"

"""

        if step == ClassificationStep.ATOMICITY_CHECK:
            return base_prompt + """请判断这个需求描述是否满足"原子性"（Atomic）？

原子性需求的特征：
1. 描述一个单一、完整的功能点
2. 不能进一步分解为更小的独立需求
3. 有明确的完成标准
4. 实现后能够独立交付价值

请分析：
1. 这个需求是否可以分解为多个子需求？
2. 它是否描述了一个完整的功能单元？
3. 实现这个需求是否有明确的完成标准？

请按照以下格式回复：
【分析】你的分析过程
【决策】true/false（true表示满足原子性，false表示不满足）
【置信度】0.0-1.0之间的数字
【证据】从需求文本中提取的支持你决策的关键词或短语，用逗号分隔"""

        elif step == ClassificationStep.USER_ACTION_CHECK:
            return base_prompt + """请判断这个需求执行的条件中，是否包含了"用户必须亲自实施的动作"？

用户亲自实施的动作包括：
1. 用户点击、输入、选择等交互操作
2. 用户发起请求或命令
3. 用户提供输入数据
4. 用户确认或批准
5. 用户触发某个流程

系统自动执行的动作不算，例如：
1. 系统定时任务
2. 系统自动响应
3. 系统后台处理

请分析：
1. 需求描述中提到了哪些动作？
2. 这些动作是否需要用户亲自执行？
3. 是否有明确的用户交互环节？

请按照以下格式回复：
【分析】你的分析过程
【决策】true/false（true表示包含用户必须亲自实施的动作，false表示不包含）
【置信度】0.0-1.0之间的数字
【证据】从需求文本中提取的支持你决策的关键词或短语，用逗号分隔"""

        elif step == ClassificationStep.SYSTEM_CONDITION_CHECK:
            return base_prompt + """请判断这个需求执行的条件中，是否包含了"系统必须满足的条件"？

系统必须满足的条件包括：
1. 系统状态条件（如：当系统处于X状态时）
2. 数据条件（如：当数据满足Y条件时）
3. 时间条件（如：每天特定时间）
4. 事件条件（如：当某事件发生时）
5. 前置条件（如：必须先完成A才能执行B）

注意：这不同于条件型需求，这里强调的是系统自身的条件，而不是外部条件

请分析：
1. 需求执行是否有前提条件？
2. 这些条件是否需要系统主动检查或满足？
3. 是否有明确的系统状态依赖？

请按照以下格式回复：
【分析】你的分析过程
【决策】true/false（true表示包含系统必须满足的条件，false表示不包含）
【置信度】0.0-1.0之间的数字
【证据】从需求文本中提取的支持你决策的关键词或短语，用逗号分隔"""

        elif step == ClassificationStep.CONDITIONAL_CHECK:  # 新增条件性检查
            return base_prompt + """请判断这个需求是否属于"条件型需求"？

条件型需求（Conditional）的核心特征：
用于描述系统功能生效或用户操作得以执行的强制性前提条件，其核心是界定"必要条件"（即"只有满足条件A，才允许执行B"）。

典型表述包括：
- "只有…才…"、"必须…才可…"、"需…才…"
- "须"、"需"、"只"、"不得"、"不能"等情态动词
- "当且仅当"、"除非...否则不"等强调必要条件的表述

关键判断标准：
1. 必须是强制性前提条件，条件不满足则功能/操作不可执行
2. 强调"只有满足X，才能做Y"的逻辑关系
3. 条件必须是功能执行的必要约束，而非可选条件

【特别注意区分】：
- ❌ 大部分带有"如果…"的句子若仅描述事件触发后的系统反应（即"充分条件"），则属于交互性需求而非条件性需求
- ✅ 关键区别在于是否强调功能执行必须满足的先行约束
- ❌ 系统默认功能、通用能力不算条件型需求
- ✅ 必须是特定功能的特定前提条件

典型示例：
✅ 条件型需求：
- "只有管理员权限的用户才能删除系统配置"
- "必须完成实名认证才可进行交易操作"
- "当且仅当库存数量大于0时，用户才能下单购买"
- "用户年龄不得小于18岁，否则不能注册账号"

❌ 非条件型需求（可能是交互型）：
- "如果用户点击保存按钮，系统将保存当前编辑的内容"
- "当用户登录成功后，系统显示欢迎页面"
- "用户可以选择上传个人头像"

请分析：
1. 需求是否描述了一个强制性前提条件？
2. 是否强调"只有...才能..."的必要关系？
3. 条件不满足时，相关功能是否完全不可执行？
4. 是否使用了"必须"、"不得"、"只有...才..."等强调必要性的表述？

请按照以下格式回复：
【分析】你的分析过程
【决策】true/false（true表示是条件型需求，false表示不是条件型需求）
【置信度】0.0-1.0之间的数字
【证据】从需求文本中提取的支持你决策的关键词或短语，用逗号分隔"""

        else:
            return base_prompt + """请对这个需求进行分类。

请按照以下格式回复：
【分析】你的分析过程
【分类】composite/interactive/sequential/structural/conditional
【置信度】0.0-1.0之间的数字
【证据】从需求文本中提取的支持你分类的关键词或短语，用逗号分隔"""

    def _parse_llm_response(self, response: str, step: ClassificationStep) -> Tuple[bool, float, str, List[str]]:
        """解析LLM响应"""

        try:
            # 提取分析部分
            analysis_match = re.search(r'【分析】\s*([^【】]+)', response, re.DOTALL)
            reasoning = analysis_match.group(1).strip() if analysis_match else ""

            # 提取决策/分类部分
            if step == ClassificationStep.FINAL_CLASSIFICATION:
                # 最终分类
                classification_match = re.search(r'【分类】\s*(\w+)', response)
                if classification_match:
                    class_str = classification_match.group(1).lower()
                    # 将分类字符串映射到布尔决策（对于最终步骤，这里简化处理）
                    decision = True  # 占位符
                else:
                    decision = True
            else:
                # 布尔决策
                decision_match = re.search(r'【决策】\s*(true|false)', response, re.IGNORECASE)
                if decision_match:
                    decision_str = decision_match.group(1).lower()
                    decision = decision_str == "true"
                else:
                    decision = False

            # 提取置信度
            confidence_match = re.search(r'【置信度】\s*([0-9]*\.?[0-9]+)', response)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.5

            # 提取证据
            evidence_match = re.search(r'【证据】\s*([^【】]+)', response)
            if evidence_match:
                evidence_text = evidence_match.group(1).strip()
                evidence = [e.strip() for e in evidence_text.split(',') if e.strip()]
            else:
                evidence = []

            return decision, confidence, reasoning, evidence

        except Exception as e:
            print(f"解析LLM响应失败: {e}")
            return False, 0.5, f"解析失败: {str(e)}", []


# ====================== LLM客户端 ======================

class DoubaoLLMClient:
    """豆包LLM客户端"""

    def __init__(self, model_name: str = "doubao-seed-1-6-250615"):
        self.model_name = model_name

        # 初始化豆包客户端
        api_key = os.getenv('DOUBAO_API_KEY')
        if not api_key:
            raise ValueError("DOUBAO_API_KEY环境变量未设置")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

        # 系统提示词
        self.system_prompt = """你是一位专业的软件需求分析师，擅长分析需求的特性并进行分类。
你的任务是严格按照决策树规则进行分析，并提供详细的推理过程。"""

    def query_decision(self, prompt: str, step: ClassificationStep) -> str:
        """查询LLM进行决策"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                stream=False
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"LLM查询失败: {e}")
            return f"""【分析】由于技术问题，无法完成分析。
【决策】false
【置信度】0.5
【证据】无"""


# ====================== 奖励计算器 ======================

class RewardCalculator:
    """奖励计算器 - 基于多步骤决策过程"""

    def __init__(self, config=None):
        self.config = config or self._default_config()

    def _default_config(self):
        return {
            "weights": {
                "accuracy": 0.40,
                "efficiency": 0.25,
                "confidence_quality": 0.20,
                "reasoning_quality": 0.15,
            },
            "step_rewards": {
                "correct_decision": 0.3,
                "incorrect_decision": -0.4,
                "high_confidence_correct": 0.1,
                "high_confidence_wrong": -0.2,
                "evidence_found": 0.05,
                "step_efficiency": 0.1,
            },
            "final_rewards": {
                "correct_classification": 2.0,
                "incorrect_classification": -1.5,
                "early_correct": 0.5,
                "optimal_path": 0.3,
            },
            "thresholds": {
                "high_confidence": 0.8,
                "low_confidence": 0.3,
                "max_steps": 5,  # 更新最大步骤数
                "optimal_step_counts": {  # 各类别的最优步骤数
                    "composite": 1,  # 第1步就终止
                    "interactive": 2,  # 第2步终止
                    "sequential": 3,  # 第3步终止
                    "conditional": 4,  # 第4步终止（最后检查）
                    "structural": 4  # 第4步终止（在条件型之后）
                }
            }
        }

    def calculate_episode_reward(self, episode: ClassificationEpisode) -> Tuple[float, List[float]]:
        """计算完整回合的奖励"""
        step_rewards = []

        # 1. 计算步骤奖励
        for i, step in enumerate(episode.steps):
            step_reward = self._calculate_step_reward(step, i, episode)
            step_rewards.append(step_reward)

        # 2. 计算最终奖励
        final_reward = self._calculate_final_reward(episode)

        # 3. 组合总奖励
        total_reward = sum(step_rewards) + final_reward

        # 4. 归一化（可选）
        total_reward = self._normalize_reward(total_reward, episode)

        return total_reward, step_rewards

    def _calculate_step_reward(self, step: StepDecision, step_index: int, episode: ClassificationEpisode) -> float:
        """计算单步奖励"""
        reward = 0.0

        # 置信度质量奖励
        reward += self._confidence_quality_reward(step.confidence)

        # 证据提取奖励
        if step.evidence and len(step.evidence) > 0:
            reward += self.config["step_rewards"]["evidence_found"]

        # 推理质量评估
        if len(step.reasoning) > 50:
            reward += 0.05

        return reward

    def _calculate_final_reward(self, episode: ClassificationEpisode) -> float:
        """计算最终奖励"""
        if episode.ground_truth is None:
            return 0.0

        reward = 0.0

        # 1. 最终分类正确性
        is_correct = episode.final_classification == episode.ground_truth

        if is_correct:
            reward += self.config["final_rewards"]["correct_classification"]

            # 2. 检查是否是最优路径
            optimal_steps = self.config["thresholds"]["optimal_step_counts"].get(
                episode.ground_truth.value, 3
            )

            if len(episode.steps) <= optimal_steps:
                reward += self.config["final_rewards"]["optimal_path"]

            # 3. 提前正确终止奖励
            if len(episode.steps) < optimal_steps:
                reward += self.config["final_rewards"]["early_correct"]
        else:
            reward += self.config["final_rewards"]["incorrect_classification"]

        # 4. 步骤效率惩罚/奖励
        if len(episode.steps) > self.config["thresholds"]["max_steps"]:
            reward -= 0.5

        return reward

    def _confidence_quality_reward(self, confidence: float) -> float:
        """置信度质量奖励"""
        if 0.4 <= confidence <= 0.7:
            return 0.1
        elif 0.7 < confidence <= 0.9:
            return 0.05
        elif confidence > 0.9:
            return -0.1
        else:
            return -0.05

    def _normalize_reward(self, reward: float, episode: ClassificationEpisode) -> float:
        """归一化奖励到合理范围"""
        max_possible = (
                len(episode.steps) * 0.5 +
                self.config["final_rewards"]["correct_classification"] +
                self.config["final_rewards"]["optimal_path"] +
                self.config["final_rewards"]["early_correct"]
        )

        min_possible = (
                len(episode.steps) * -0.5 +
                self.config["final_rewards"]["incorrect_classification"] -
                0.5
        )

        if max_possible > min_possible:
            normalized = -1 + 3 * (reward - min_possible) / (max_possible - min_possible)
            return max(-1.0, min(2.0, normalized))
        else:
            return max(-1.0, min(2.0, reward))


# ====================== 处理器和主程序 ======================

class SequentialClassificationProcessor:
    """序列化分类处理器"""

    def __init__(self, llm_client=None, agent=None):
        self.llm_client = llm_client or DoubaoLLMClient()
        self.agent = agent or ClassificationAgent(self.llm_client)
        self.results = []
        self.statistics = defaultdict(list)

    def classify_requirement(self, requirement_text: str,
                             requirement_id: str = None,
                             ground_truth: RequirementType = None) -> ClassificationEpisode:
        """对单个需求进行分类"""

        print(f"\n{'=' * 60}")
        print(f"🔍 开始分类需求: {requirement_text[:80]}...")
        if ground_truth:
            print(f"📝 真实类别: {ground_truth.value}")
        print(f"{'=' * 60}")

        # 重置智能体
        state = self.agent.reset(requirement_text, requirement_id)

        step_count = 0
        is_complete = False

        while not is_complete and step_count < 10:
            step_count += 1

            print(f"\n📋 步骤 {step_count}: {state.step.value}")

            # 执行一步决策
            step_decision, new_state, is_complete = self.agent.step()

            if step_decision:
                print(f"  决策: {'通过' if step_decision.decision else '不通过'}")
                print(f"  置信度: {step_decision.confidence:.3f}")
                print(f"  证据: {', '.join(step_decision.evidence[:3])}" +
                      ("..." if len(step_decision.evidence) > 3 else ""))

            state = new_state

            if is_complete:
                print(f"\n✅ 分类完成!")
                print(f"  最终分类: {state.final_classification.value}")
                print(f"  总步骤数: {step_count}")
                break

        # 获取完整的episode
        episode = self.agent.current_episode
        episode.ground_truth = ground_truth

        # 计算奖励
        if self.agent.reward_calculator:
            total_reward, step_rewards = self.agent.reward_calculator.calculate_episode_reward(episode)
            episode.total_reward = total_reward
            episode.step_rewards = step_rewards
            print(f"  总奖励: {total_reward:.3f}")

        # 记录统计
        self._record_statistics(episode)

        return episode

    def _record_statistics(self, episode: ClassificationEpisode):
        """记录统计信息"""
        self.results.append(episode)

        if episode.ground_truth:
            is_correct = episode.final_classification == episode.ground_truth
            self.statistics["accuracy"].append(1 if is_correct else 0)

        self.statistics["steps"].append(len(episode.steps))
        self.statistics["rewards"].append(episode.total_reward)
        self.statistics["confidences"].append(
            np.mean([s.confidence for s in episode.steps]) if episode.steps else 0
        )

    def process_dataset(self, dataset: List[Dict]) -> List[ClassificationEpisode]:
        """处理数据集"""
        episodes = []

        print(f"\n🚀 开始批量处理 {len(dataset)} 条需求")
        print(f"{'=' * 60}")

        for i, item in enumerate(dataset):
            requirement_text = item.get('requirement', item.get('content', ''))

            # 提取真实标签（如果存在）
            ground_truth = None
            if 'label' in item and item['label']:
                label_str = item['label'].strip().lower()
                for req_type in RequirementType:
                    if req_type.value == label_str:
                        ground_truth = req_type
                        break

            # 分类需求
            episode = self.classify_requirement(
                requirement_text=requirement_text,
                requirement_id=f"req_{i + 1:04d}",
                ground_truth=ground_truth
            )

            episodes.append(episode)

            # 进度显示
            if (i + 1) % 5 == 0:
                print(f"\n📊 进度: {i + 1}/{len(dataset)}")
                if self.statistics.get("accuracy"):
                    accuracy = np.mean(self.statistics["accuracy"]) * 100
                    print(f"  当前准确率: {accuracy:.1f}%")

            # 避免API速率限制
            if i < len(dataset) - 1:
                time.sleep(1)

        return episodes

    def save_results(self, episodes: List[ClassificationEpisode], output_file: str):
        """保存结果到Excel"""
        results_data = []

        for i, episode in enumerate(episodes):
            # 构建步骤历史字符串
            steps_summary = []
            for j, step in enumerate(episode.steps):
                steps_summary.append(
                    f"{step.step.value}: {'✓' if step.decision else '✗'}({step.confidence:.2f})"
                )

            row_data = {
                '序号': i + 1,
                '需求ID': episode.requirement_id,
                '需求内容': episode.requirement_text,
                '步骤数': len(episode.steps),
                '步骤详情': ' → '.join(steps_summary),
                '最终分类': episode.final_classification.value,
                '真实分类': episode.ground_truth.value if episode.ground_truth else "N/A",
                '是否一致': '✅ 一致' if (
                        episode.ground_truth and
                        episode.final_classification == episode.ground_truth
                ) else '❌ 不一致' if episode.ground_truth else 'N/A',
                '总奖励': round(episode.total_reward, 3),
                '平均置信度': round(np.mean([s.confidence for s in episode.steps]) if episode.steps else 0, 3),
                '证据数量': sum(len(s.evidence) for s in episode.steps),
                '处理时间': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # 添加每个步骤的详细信息
            for j, step in enumerate(episode.steps):
                row_data[f'步骤{j + 1}_决策'] = step.step.value
                row_data[f'步骤{j + 1}_结果'] = '通过' if step.decision else '不通过'
                row_data[f'步骤{j + 1}_置信度'] = step.confidence
                row_data[f'步骤{j + 1}_证据'] = ', '.join(step.evidence[:3]) + (
                    '...' if len(step.evidence) > 3 else ''
                )

            results_data.append(row_data)

        df = pd.DataFrame(results_data)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.',
                    exist_ok=True)

        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ 结果已保存到: {output_file}")

        # 保存详细报告
        self._save_detailed_report(episodes, output_file)

    def _save_detailed_report(self, episodes: List[ClassificationEpisode], output_file: str):
        """保存详细分析报告"""
        report_file = output_file.replace('.xlsx', '_详细报告.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("软件需求序列化分类系统 - 详细分析报告\n")
            f.write("分类流程: 原子性 → 用户动作 → 系统条件 → 条件性 → 最终分类\n")  # 更新流程描述
            f.write("=" * 80 + "\n\n")

            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总需求数量: {len(episodes)}\n\n")

            # 统计信息
            f.write("📊 整体统计信息\n")
            f.write("-" * 80 + "\n")

            # 准确率
            episodes_with_truth = [e for e in episodes if e.ground_truth]
            if episodes_with_truth:
                correct_count = sum(1 for e in episodes_with_truth
                                    if e.final_classification == e.ground_truth)
                accuracy = correct_count / len(episodes_with_truth) * 100
                f.write(f"准确率: {accuracy:.1f}% ({correct_count}/{len(episodes_with_truth)})\n")

            # 步骤统计
            step_counts = [len(e.steps) for e in episodes]
            f.write(f"平均步骤数: {np.mean(step_counts):.2f}\n")
            f.write(f"最少步骤: {np.min(step_counts)}\n")
            f.write(f"最多步骤: {np.max(step_counts)}\n")

            # 奖励统计
            rewards = [e.total_reward for e in episodes]
            f.write(f"平均奖励: {np.mean(rewards):.3f}\n")
            f.write(f"最高奖励: {np.max(rewards):.3f}\n")
            f.write(f"最低奖励: {np.min(rewards):.3f}\n\n")

            # 分类分布
            f.write("📈 分类分布\n")
            f.write("-" * 80 + "\n")

            from collections import Counter
            classification_counts = Counter([e.final_classification.value for e in episodes])
            for class_type, count in classification_counts.items():
                percentage = count / len(episodes) * 100
                f.write(f"{class_type}: {count}条 ({percentage:.1f}%)\n")

            f.write("\n")

            # 步骤效率分析
            f.write("⚡ 步骤效率分析\n")
            f.write("-" * 80 + "\n")

            # 各分类的最优步骤数
            optimal_steps = {
                "composite": 1,
                "conditional": 2,  # 新增
                "interactive": 3,
                "sequential": 4,
                "structural": 4
            }

            for class_type in ["composite", "conditional", "interactive", "sequential", "structural"]:
                class_episodes = [e for e in episodes
                                  if e.final_classification.value == class_type]
                if class_episodes:
                    avg_steps = np.mean([len(e.steps) for e in class_episodes])
                    optimal = optimal_steps.get(class_type, 3)
                    f.write(f"{class_type}: 平均{avg_steps:.1f}步 (最优: {optimal}步)\n")

            f.write("\n")

            # 不一致案例分析
            f.write("🔍 不一致案例分析\n")
            f.write("-" * 80 + "\n")

            inconsistencies = []
            for e in episodes_with_truth:
                if e.final_classification != e.ground_truth:
                    inconsistencies.append(e)

            if inconsistencies:
                f.write(f"发现 {len(inconsistencies)} 个不一致案例:\n\n")

                for i, e in enumerate(inconsistencies[:10]):
                    f.write(f"{i + 1}. 需求ID: {e.requirement_id}\n")
                    f.write(f"   模型分类: {e.final_classification.value}\n")
                    f.write(f"   真实分类: {e.ground_truth.value}\n")
                    f.write(f"   步骤: {len(e.steps)}步\n")

                    # 显示错误发生的位置
                    for j, step in enumerate(e.steps):
                        f.write(f"   步骤{j + 1}({step.step.value}): {'✓' if step.decision else '✗'}\n")

                    f.write(f"   需求内容: {e.requirement_text[:100]}...\n\n")
            else:
                f.write("没有发现不一致案例。\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✅ 详细报告已保存到: {report_file}")

    def print_summary_statistics(self):
        """打印统计摘要"""
        print(f"\n{'=' * 60}")
        print("📊 序列化分类系统 - 统计摘要")
        print(f"{'=' * 60}")

        if not self.statistics:
            print("暂无统计数据")
            return

        # 准确率
        if self.statistics.get("accuracy"):
            accuracy = np.mean(self.statistics["accuracy"]) * 100
            print(f"🎯 准确率: {accuracy:.1f}%")

        # 步骤统计
        if self.statistics.get("steps"):
            avg_steps = np.mean(self.statistics["steps"])
            min_steps = np.min(self.statistics["steps"])
            max_steps = np.max(self.statistics["steps"])
            print(f"📈 步骤统计: 平均{avg_steps:.1f}步 (范围: {min_steps}-{max_steps})")

        # 奖励统计
        if self.statistics.get("rewards"):
            avg_reward = np.mean(self.statistics["rewards"])
            min_reward = np.min(self.statistics["rewards"])
            max_reward = np.max(self.statistics["rewards"])
            print(f"🏆 奖励统计: 平均{avg_reward:.3f} (范围: {min_reward:.3f}-{max_reward:.3f})")

        # 置信度统计
        if self.statistics.get("confidences"):
            avg_conf = np.mean(self.statistics["confidences"])
            print(f"📊 平均置信度: {avg_conf:.3f}")

        print(f"\n处理需求总数: {len(self.results)}")


# ====================== 主函数 ======================

def main():
    """主函数"""
    # 文件路径配置
    dataset_file = "dataset.xlsx"
    output_file = "sequential_classification_results.xlsx"

    print("=" * 80)
    print("🔧 软件需求序列化分类系统")
    print("分类流程: 原子性 → 用户动作 → 系统条件 → 条件性 → 最终分类")  # 更新流程描述
    print("=" * 80)

    # 检查环境变量
    print(f"\n🔧 环境检查:")
    api_key = os.getenv('DOUBAO_API_KEY')

    if api_key:
        print(f"✅ DOUBAO_API_KEY 已设置")
    else:
        print(f"❌ DOUBAO_API_KEY 未设置")
        return

    # 检查数据文件
    if not os.path.exists(dataset_file):
        print(f"❌ 数据文件不存在: {dataset_file}")
        return

    try:
        # 加载数据
        print(f"\n📁 加载数据集...")
        df = pd.read_excel(dataset_file)

        # 转换为需要的格式
        dataset = []
        for _, row in df.iterrows():
            if 'requirement' in df.columns:
                item = {'requirement': str(row['requirement'])}
                if 'label' in df.columns and pd.notna(row.get('label')):
                    item['label'] = str(row['label'])
                dataset.append(item)
            elif len(df.columns) > 0:
                item = {'requirement': str(row.iloc[0])}
                dataset.append(item)

        print(f"✅ 成功加载 {len(dataset)} 条需求")

        # 创建处理器并运行
        print(f"\n🚀 开始序列化分类处理...")

        processor = SequentialClassificationProcessor()
        episodes = processor.process_dataset(dataset)

        # 保存结果
        processor.save_results(episodes, output_file)

        # 打印统计信息
        processor.print_summary_statistics()

        # 显示前几个案例
        print(f"\n🔍 前3个分类案例:")
        for i, episode in enumerate(episodes[:3]):
            print(f"\n{'=' * 60}")
            print(f"案例 {i + 1}:")
            print(f"需求: {episode.requirement_text[:80]}...")
            print(f"步骤数: {len(episode.steps)}")

            for j, step in enumerate(episode.steps):
                print(f"  步骤{j + 1}({step.step.value}): {'通过' if step.decision else '不通过'} "
                      f"(置信度: {step.confidence:.2f})")

            print(f"最终分类: {episode.final_classification.value}")
            if episode.ground_truth:
                print(f"真实分类: {episode.ground_truth.value}")
                print(f"是否一致: {'✅ 是' if episode.final_classification == episode.ground_truth else '❌ 否'}")
            print(f"总奖励: {episode.total_reward:.3f}")

        print(f"\n{'=' * 80}")
        print("🎉 序列化分类处理完成!")
        print(f"📊 结果文件: {output_file}")
        print(f"📄 详细报告: {output_file.replace('.xlsx', '_详细报告.txt')}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()