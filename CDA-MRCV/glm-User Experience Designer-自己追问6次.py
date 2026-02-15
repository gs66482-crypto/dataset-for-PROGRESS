import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
import time
import re
import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import pearsonr

# 加载环境变量
load_dotenv()


@dataclass
class ModelPrediction:
    """存储模型预测结果"""
    label: str
    confidence: float
    model_name: str
    attempts: List[Dict] = None
    human_label: str = None
    match_result: bool = False
    total_attempts: int = 0
    final_reason: str = ""
    calibration_error: float = None
    pearson_correlation: float = None
    agreement_status: str = ""  # 记录LLM内部一致状态


@dataclass
class DataPoint:
    """数据点类"""
    content: str
    prediction: ModelPrediction = None
    final_label: str = None
    human_label: str = None


class SingleRoleGLMClient:
    """单角色GLM模型客户端类 - 仅User Experience Designer"""

    def __init__(self, name: str = "glm-4.5", model_name: str = "glm-4.5"):
        self.name = name
        self.model_name = model_name

        # GLM API配置 - 使用阿里云DashScope
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY环境变量未设置")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 只定义User Experience Designer角色
        self.role = {
            "name": "User Experience Designer",
            "system_prompt": """你是一位用户体验设计师(User Experience Designer)，专注于用户交互、界面设计和用户体验优化。你从用户旅程、交互设计、可用性、可访问性和情感化设计角度分析需求。你关注需求如何影响用户满意度、易用性、学习曲线和整体用户体验。你的分析侧重于需求如何被"用户感知和使用"。

请按照以下要求分析：
1. 从用户体验角度深入分析需求
2. 选择最合适的分类标签
3. 评估你的判断置信度（0.0-1.0）
4. 提供清晰的分析理由""",
            "focus_areas": ["用户体验", "交互设计", "可用性", "可访问性", "用户满意度"]
        }

        print(f"✅ 成功创建单角色GLM模型客户端: {self.name}")
        print(f"  使用的专业角色: {self.role['name']}")
        print(f"  🔄 新的尝试机制: 至少2次独立标注，不一致才继续")
        print(f"  📝 后续标注优先考虑新类别")
        print(f"  📊 包含置信度分析: ECE和皮尔逊检验")
        print(f"  🌐 API配置: 阿里云DashScope - GLM模型")
        print(f"  模型版本: {self.model_name}")
        print(f"  ⚠️  注意: GLM模型需要使用流式模式")

    def analyze_with_independent_voting(self, text: str, labels: List[str], human_label: str = None) -> ModelPrediction:
        """独立投票机制分析：至少执行2次，不一致才继续，后续优先新类别"""
        print(f"\n🔍 开始独立投票分析: '{text[:50]}...'")
        print(f"  🏷️  可用标签: {', '.join(labels)}")
        print("-" * 60)

        attempts = []
        max_attempts = min(6, len(labels) + 1)  # 最多6次或标签数+1次
        tried_labels = set()  # 记录已经尝试过的标签
        final_label = ""
        final_confidence = 0.0
        match_result = False
        final_reason = ""
        agreement_status = ""

        # 第一步：至少执行2次独立标注
        print(f"    📌 步骤1: 执行至少2次独立标注...")

        # 第一次独立标注
        print(f"      第1次独立标注...")
        attempt1 = self._independent_vote(text, labels, 1, tried_labels, None)
        attempts.append(attempt1)
        tried_labels.add(attempt1["predicted_label"])

        # 第二次独立标注（不知道第一次结果，但可以看到已经尝试的标签）
        print(f"      第2次独立标注（不知道第1次结果）...")
        attempt2 = self._independent_vote(text, labels, 2, tried_labels, None)
        attempts.append(attempt2)
        tried_labels.add(attempt2["predicted_label"])

        # 检查前两次是否一致
        label1 = attempt1["predicted_label"]
        label2 = attempt2["predicted_label"]
        confidence1 = attempt1["confidence"]
        confidence2 = attempt2["confidence"]

        if label1 == label2:
            # 前两次一致，结束标注
            agreement_status = "前2次一致"
            final_label = label1
            final_confidence = (confidence1 + confidence2) / 2  # 取平均置信度
            print(f"    ✅ 前2次标注一致: '{final_label}' (置信度: {final_confidence:.3f})")
            print(f"    📌 标注结束，无需更多尝试")
        else:
            # 前两次不一致，需要继续
            agreement_status = f"前2次不一致({label1} vs {label2})"
            print(f"    ⚠️ 前2次标注不一致: '{label1}' vs '{label2}'")
            print(f"    🔄 触发第3次标注（优先考虑新类别）...")

            # 第三次标注：优先考虑新类别
            attempt3 = self._independent_vote(text, labels, 3, tried_labels, attempts[:2])
            attempts.append(attempt3)
            tried_labels.add(attempt3["predicted_label"])

            # 检查是否达成多数一致
            labels_so_far = [a["predicted_label"] for a in attempts]
            label_counts = {}
            for label in labels_so_far:
                label_counts[label] = label_counts.get(label, 0) + 1

            # 找出最多的标签
            max_count = max(label_counts.values())
            majority_labels = [l for l, c in label_counts.items() if c == max_count]

            if len(majority_labels) == 1 and max_count >= 2:
                # 已达成多数一致（至少2票）
                agreement_status += f" → 第3次后达成一致"
                final_label = majority_labels[0]
                # 取该标签所有投票的平均置信度
                relevant_confidences = [a["confidence"] for a in attempts if a["predicted_label"] == final_label]
                final_confidence = np.mean(relevant_confidences) if relevant_confidences else 0.5
                print(f"    ✅ 第3次后达成多数一致: '{final_label}' (置信度: {final_confidence:.3f})")
                print(f"    📌 标注结束")
            else:
                # 仍未达成一致，继续尝试（优先新类别）
                print(f"    ⚠️ 第3次后仍未达成多数一致，继续尝试（优先新类别）...")

                # 继续尝试直到达成一致或达到最大次数
                for attempt_num in range(4, max_attempts + 1):
                    # 如果已经尝试了所有标签，停止尝试
                    if len(tried_labels) >= len(labels):
                        print(f"    ℹ️ 已尝试所有{len(tried_labels)}个标签，停止尝试")
                        break

                    print(f"      第{attempt_num}次标注（优先新类别）...")
                    attempt = self._independent_vote(text, labels, attempt_num, tried_labels, attempts)
                    attempts.append(attempt)
                    tried_labels.add(attempt["predicted_label"])

                    # 重新检查是否达成一致
                    labels_so_far = [a["predicted_label"] for a in attempts]
                    label_counts = {}
                    for label in labels_so_far:
                        label_counts[label] = label_counts.get(label, 0) + 1

                    max_count = max(label_counts.values())
                    majority_labels = [l for l, c in label_counts.items() if c == max_count]

                    if len(majority_labels) == 1 and max_count >= 2:
                        agreement_status += f" → 第{attempt_num}次后达成一致"
                        final_label = majority_labels[0]
                        relevant_confidences = [a["confidence"] for a in attempts if
                                                a["predicted_label"] == final_label]
                        final_confidence = np.mean(relevant_confidences) if relevant_confidences else 0.5
                        print(f"    ✅ 第{attempt_num}次后达成多数一致: '{final_label}'")
                        print(f"    📌 标注结束")
                        break
                    elif attempt_num == max_attempts or len(tried_labels) >= len(labels):
                        # 达到最大尝试次数或已尝试所有标签仍未达成一致
                        agreement_status += f" → 最大{len(attempts)}次后仍不一致"
                        # 选择出现次数最多的标签，如果平局则选择置信度最高的
                        max_votes = max(label_counts.values())
                        candidate_labels = [l for l, c in label_counts.items() if c == max_votes]

                        if len(candidate_labels) == 1:
                            final_label = candidate_labels[0]
                        else:
                            # 平局：选择置信度最高的那个
                            best_label = None
                            best_confidence = -1
                            for label in candidate_labels:
                                # 计算该标签的平均置信度
                                label_confidences = [a["confidence"] for a in attempts if a["predicted_label"] == label]
                                avg_confidence = np.mean(label_confidences) if label_confidences else 0
                                if avg_confidence > best_confidence:
                                    best_confidence = avg_confidence
                                    best_label = label
                            final_label = best_label

                        # 计算最终置信度
                        relevant_confidences = [a["confidence"] for a in attempts if
                                                a["predicted_label"] == final_label]
                        final_confidence = np.mean(relevant_confidences) if relevant_confidences else 0.5

                        tried_count = len(tried_labels)
                        total_labels = len(labels)
                        print(f"    ⚠️ 最大{len(attempts)}次后仍未达成一致（已尝试{tried_count}/{total_labels}个标签）")
                        print(f"    📍 最终选择: '{final_label}' (置信度: {final_confidence:.3f})")

                        # 生成详细的原因说明
                        vote_summary = []
                        for label, count in label_counts.items():
                            # 获取该标签的所有置信度
                            label_confs = [a["confidence"] for a in attempts if a["predicted_label"] == label]
                            avg_conf = np.mean(label_confs) if label_confs else 0
                            vote_summary.append(f"'{label}': {count}票(平均置信度{avg_conf:.3f})")

                        final_reason = f"经过{len(attempts)}次投票，分布为: {', '.join(vote_summary)}，最终选择'{final_label}'"
                        break

        # 第二步：与人类标注比较（仅用于最终检验）
        if human_label:
            match_result = (final_label == human_label)
            if match_result:
                print(f"    🎯 与人类标注比较: ✅ 一致 (人类标注: {human_label})")
            else:
                print(f"    🎯 与人类标注比较: ❌ 不一致 (模型: {final_label}, 人类: {human_label})")
                if not final_reason:  # 如果还没有原因，添加一个
                    final_reason = f"模型经过{len(attempts)}次投票选择'{final_label}'，但人类标注为'{human_label}'"
        else:
            print(f"    ℹ️ 无人类标注可用于比较")
            match_result = True  # 默认认为一致

        print(f"\n📊 最终结果:")
        print(f"  LLM内部状态: {agreement_status}")
        print(f"  模型最终预测: {final_label}")
        print(f"  最终置信度: {final_confidence:.3f}")
        print(f"  总投票次数: {len(attempts)}")
        print(f"  尝试过的标签: {', '.join(sorted(tried_labels))}")
        if human_label:
            print(f"  人类标注: {human_label}")
            print(f"  是否一致: {'✅ 是' if match_result else '❌ 否'}")
        if final_reason:
            print(f"  原因说明: {final_reason}")

        return ModelPrediction(
            label=final_label,
            confidence=final_confidence,
            model_name=f"{self.name}-independent-voting",
            attempts=attempts,
            human_label=human_label,
            match_result=match_result,
            total_attempts=len(attempts),
            final_reason=final_reason,
            agreement_status=agreement_status
        )

    def _independent_vote(self, text: str, labels: List[str], attempt_num: int,
                          tried_labels: set, previous_attempts: Optional[List[Dict]] = None) -> Dict:
        """独立投票，优先考虑新类别 - 使用流式模式"""
        print(f"        🔄 正在进行第{attempt_num}次投票...")

        try:
            # 构建提示词，根据尝试次数调整
            prompt = self._build_vote_prompt_with_priority(text, labels, attempt_num, tried_labels, previous_attempts)

            # GLM模型需要使用流式模式
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.role["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=self._get_temperature_for_attempt(attempt_num),
                max_tokens=300,
                stream=True,  # GLM模型需要启用流式模式
                extra_body={"stream_mode": "normal"}  # 添加额外参数
            )

            # 收集流式响应
            response_content = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content

            predicted_label = self._extract_label_from_response(response_content, labels)
            confidence = self._extract_confidence_from_response(response_content)

            if predicted_label:
                print(f"          ✅ 第{attempt_num}次投票: {predicted_label} (置信度: {confidence:.3f})")

                # 检查是否是新标签
                if predicted_label in tried_labels:
                    print(f"          ℹ️  注意: 选择了已尝试过的标签'{predicted_label}'")
                else:
                    print(f"          🌟 选择了新标签'{predicted_label}'")

                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "is_new_label": predicted_label not in tried_labels,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
            else:
                # 如果提取标签失败，选择一个未尝试的标签（如果可能）
                untried_labels = [l for l in labels if l not in tried_labels]
                if untried_labels:
                    # 优先选择未尝试的标签
                    fallback_label = untried_labels[0]
                    is_new = True
                else:
                    # 如果所有标签都尝试过了，随机选择一个
                    fallback_label = np.random.choice(labels)
                    is_new = False

                print(
                    f"          ⚠️ 第{attempt_num}次投票标签提取失败，选择: {fallback_label} {'(新标签)' if is_new else '(已尝试标签)'}")
                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": fallback_label,
                    "confidence": 0.5,
                    "is_new_label": is_new,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
        except Exception as e:
            print(f"          ❌ 第{attempt_num}次投票异常: {str(e)[:100]}")
            # 异常时也优先选择新标签
            untried_labels = [l for l in labels if l not in tried_labels]
            if untried_labels:
                fallback_label = untried_labels[0]
                is_new = True
            else:
                fallback_label = labels[0]
                is_new = False

            return {
                "attempt_number": attempt_num,
                "response": str(e),
                "predicted_label": fallback_label,
                "confidence": 0.5,
                "is_new_label": is_new,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }

    def _get_temperature_for_attempt(self, attempt_num: int) -> float:
        """根据尝试次数调整温度 - 最简实用版"""
        temperature_map = {
            1: 0.3,  # 第一次：确定
            2: 0.6,  # 开始发散
            3: 0.8,  # 最大发散
            4: 0.8,  # 保持发散
            5: 0.6,  # 开始收敛
            6: 0.3,  # 最后确定
        }
        return temperature_map.get(attempt_num, 0.5)

    def _build_vote_prompt_with_priority(self, text: str, labels: List[str], attempt_num: int,
                                         tried_labels: set, previous_attempts: Optional[List[Dict]] = None) -> str:
        """构建投票提示，优先考虑新类别"""
        labels_text = "\n".join([f"- {label}" for label in labels])

        if attempt_num == 1:
            # 第一次投票：普通提示
            return f"""作为用户体验设计师，请独立分析以下需求：

需求描述：
"{text}"

可选分类标签：
{labels_text}

请从用户体验角度进行独立分析，并给出你的判断：

【用户体验分析】[详细分析原因]
【分类标签】[必须从上述标签中选择一个]
【置信度】[0.0-1.0之间的小数]"""

        elif attempt_num == 2:
            # 第二次投票：独立分析，但提示已有尝试
            tried_text = f"注意：已经有人尝试过以下标签: {', '.join(sorted(tried_labels))}" if tried_labels else ""

            return f"""作为用户体验设计师，请再次独立分析以下需求：

需求描述：
"{text}"

{tried_text}

可选分类标签：
{labels_text}

请给出独立的分析判断：

【重新分析】[独立分析]
【分类标签】[必须从上述标签中选择一个]
【置信度】[0.0-1.0之间的小数]"""

        else:
            # 第三次及以后的投票：优先考虑新类别
            untried_labels = [l for l in labels if l not in tried_labels]

            # 构建历史信息
            history_text = ""
            if previous_attempts:
                history_text = "之前的投票历史：\n"
                for attempt in previous_attempts:
                    history_text += f"第{attempt['attempt_number']}次: {attempt['predicted_label']} "
                    history_text += f"(置信度: {attempt['confidence']:.3f})\n"

            priority_text = ""
            if untried_labels:
                if len(untried_labels) == 1:
                    priority_text = f"\n重要提示：只剩下一个未尝试的标签'{untried_labels[0]}'，请优先考虑它。"
                else:
                    priority_text = f"\n重要提示：请优先考虑以下未尝试过的标签: {', '.join(untried_labels)}"
            else:
                priority_text = "\n注意：所有标签都已尝试过，请综合考虑所有历史投票。"

            return f"""作为用户体验设计师，请综合考虑之前的分析，给出最终判断：

需求描述：
"{text}"

{history_text}

{priority_text}

可选分类标签：
{labels_text}

请综合考虑所有分析，特别关注未尝试过的标签可能性：

【综合分析】[综合考虑历史，特别分析未尝试标签的可能性]
【分类标签】[必须从上述标签中选择一个]
【置信度】[0.0-1.0之间的小数]"""

    def _extract_label_from_response(self, response: str, labels: List[str]) -> str:
        """从响应中提取标签"""
        if not response or not labels:
            return labels[0] if labels else "Unknown"

        response_clean = response.strip()

        # 1. 尝试从【分类标签】格式中提取
        label_pattern = r'【分类标签】\s*[:：]?\s*(.+)'
        label_match = re.search(label_pattern, response_clean, re.IGNORECASE | re.MULTILINE)
        if label_match:
            extracted = label_match.group(1).strip().strip('.,!?;:"\'')
            for label in labels:
                if label.lower() == extracted.lower():
                    return label

        # 2. 在整个响应中搜索标签
        for label in labels:
            if label.lower() in response_clean.lower():
                return label

        # 3. 尝试其他可能的格式
        alt_patterns = [
            r'分类[:：]\s*(.+)',
            r'标签[:：]\s*(.+)',
            r'选择[:：]\s*(.+)',
            r'结果为[:：]\s*(.+)',
            r'最终分类[:：]\s*(.+)'
        ]

        for pattern in alt_patterns:
            match = re.search(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted = match.group(1).strip().strip('.,!?;:"\'')
                for label in labels:
                    if label.lower() in extracted.lower():
                        return label

        # 4. 尝试查找加粗或引号内的标签
        bold_pattern = r'\*\*(.+?)\*\*|「(.+?)」|『(.+?)』|"(.+?)"|\'(.+?)\''
        bold_matches = re.findall(bold_pattern, response_clean)
        for match in bold_matches:
            for group in match:
                if group:
                    for label in labels:
                        if label.lower() == group.lower():
                            return label

        return ""  # 返回空字符串，让调用者处理

    def _extract_confidence_from_response(self, response: str) -> float:
        """从响应中提取置信度"""
        if not response:
            return 0.5

        response_clean = response.strip()

        confidence_patterns = [
            r'【置信度】\s*[:：]?\s*([0-9]*\.?[0-9]+)',
            r'置信度[:：]\s*([0-9]*\.?[0-9]+)',
            r'confidence[:：]\s*([0-9]*\.?[0-9]+)',
            r'置信度\s*=\s*([0-9]*\.?[0-9]+)',
            r'置信水平[:：]\s*([0-9]*\.?[0-9]+)'
        ]

        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                    return confidence
                except ValueError:
                    continue

        # 尝试查找0-1之间的数字（更宽松的匹配）
        number_pattern = r'(0?\.\d{1,3}|1\.0{1,3}|0\.\d+|1\.0+)'
        number_matches = re.findall(number_pattern, response_clean)

        for num_str in number_matches:
            try:
                num = float(num_str)
                if 0 <= num <= 1:
                    return num
            except ValueError:
                continue

        # 尝试查找百分比
        percent_pattern = r'(\d{1,3})%'
        percent_matches = re.findall(percent_pattern, response_clean)
        for percent_str in percent_matches:
            try:
                percent = float(percent_str)
                if 0 <= percent <= 100:
                    return percent / 100.0
            except ValueError:
                continue

        return 0.5


class SingleRoleProcessor:
    """单角色模型处理器 - 仅User Experience Designer"""

    def __init__(self, single_role_client: SingleRoleGLMClient):
        self.single_role_client = single_role_client
        self.start_time = None
        self.end_time = None
        print(f"单角色模型处理器初始化完成，使用{single_role_client.role['name']}角色")
        print(f"🔄 独立投票机制: 至少2次，不一致才继续")
        print(f"🎯 后续标注: 优先考虑新类别")

    def process_dataset(self, dataset_with_labels: List[Dict], labels: List[str]) -> List[DataPoint]:
        """处理带标签的数据集"""
        data_points = []

        self.start_time = datetime.datetime.now()
        print(f"🚀 独立投票分析开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"开始处理 {len(dataset_with_labels)} 条带标签的软件需求...")
        print(f"分类标签数量: {len(labels)} 个")
        print(f"🎭 角色配置: User Experience Designer")
        print(f"🔄 投票机制: 至少2次独立标注 → 不一致 → 优先新类别 → 最多{min(6, len(labels) + 1)}次")

        for i, item in enumerate(dataset_with_labels):
            text = item.get('requirement', item.get('content', ''))
            human_label = item.get('label', None)

            if i % 5 == 0 and i > 0:
                elapsed_time = datetime.datetime.now() - self.start_time
                processed = i
                remaining = len(dataset_with_labels) - i
                avg_time_per_req = elapsed_time / processed if processed > 0 else elapsed_time
                est_remaining = avg_time_per_req * remaining

                print(f"  📈 进度: {i}/{len(dataset_with_labels)} - 已用时: {elapsed_time} - 预计剩余: {est_remaining}")

            data_point = DataPoint(content=text, human_label=human_label)
            prediction = self.single_role_client.analyze_with_independent_voting(text, labels, human_label)

            data_point.prediction = prediction
            data_point.final_label = prediction.label
            data_points.append(data_point)

            if i < len(dataset_with_labels) - 1:
                time.sleep(1)  # 避免API限制

        self.end_time = datetime.datetime.now()
        return data_points

    def calculate_calibration_metrics(self, results: List[DataPoint]):
        """计算校准度指标：ECE和皮尔逊检验"""
        confidences = []
        accuracies = []

        for dp in results:
            if dp.human_label and dp.prediction:
                confidences.append(dp.prediction.confidence)
                accuracies.append(1 if dp.prediction.match_result else 0)

        if len(confidences) < 10:
            print("⚠️  样本量不足，无法进行可靠的校准度分析")
            return None, None

        # 计算ECE（Expected Calibration Error）
        ece = self._calculate_ece(confidences, accuracies)

        # 计算皮尔逊相关系数
        if len(set(confidences)) > 1 and len(set(accuracies)) > 1:
            pearson_corr, p_value = pearsonr(confidences, accuracies)
        else:
            pearson_corr, p_value = 0.0, 1.0

        print(f"\n📊 置信度分析结果:")
        print(f"  ECE（预期校准误差）: {ece:.4f}")
        print(f"  皮尔逊相关系数: {pearson_corr:.4f} (p值: {p_value:.4f})")
        print(f"  样本数量: {len(confidences)}")

        # 解释结果
        if ece < 0.05:
            print(f"  ✅ ECE < 0.05，模型校准度良好")
        elif ece < 0.1:
            print(f"  ⚠️  0.05 ≤ ECE < 0.1，模型校准度一般")
        else:
            print(f"  ❌ ECE ≥ 0.1，模型校准度较差")

        if abs(pearson_corr) > 0.3:
            direction = "正" if pearson_corr > 0 else "负"
            print(f"  ✅ 皮尔逊相关系数 |r| > 0.3，置信度与准确性{direction}相关")
        elif abs(pearson_corr) > 0.1:
            print(f"  ⚠️  0.1 ≤ |相关系数| ≤ 0.3，相关性较弱")
        else:
            print(f"  ℹ️  |相关系数| < 0.1，置信度与准确性无明显线性关系")

        return ece, pearson_corr

    def _calculate_ece(self, confidences: List[float], accuracies: List[int], n_bins: int = 10) -> float:
        """计算Expected Calibration Error"""
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return ece

    def save_results(self, results: List[DataPoint], output_file: str):
        """保存最终结果到Excel文件"""
        results_data = []

        for i, data_point in enumerate(results):
            attempts_info = ""
            attempts_summary = []
            tried_labels_set = set()
            new_label_count = 0

            if data_point.prediction and data_point.prediction.attempts:
                for a in data_point.prediction.attempts:
                    label = a["predicted_label"]
                    is_new = a.get("is_new_label", False)
                    star = "🌟" if is_new else ""
                    attempts_summary.append(f"尝试{a['attempt_number']}: {label}{star}({a['confidence']:.3f})")

                    # 统计尝试过的标签和新标签数量
                    tried_labels_set.add(label)
                    if is_new:
                        new_label_count += 1

                attempts_info = "; ".join(attempts_summary)

            row_data = {
                '序号': i + 1,
                '需求内容': data_point.content,
                'LLM内部状态': data_point.prediction.agreement_status if data_point.prediction else "",
                '人类标注标签': data_point.human_label or "N/A",
                '模型最终预测': data_point.final_label,
                '是否一致': '✅ 一致' if data_point.prediction and data_point.prediction.match_result else '❌ 不一致',
                '总投票次数': data_point.prediction.total_attempts if data_point.prediction else 0,
                '尝试标签数': len(tried_labels_set),
                '新标签数': new_label_count,
                '不一致原因': data_point.prediction.final_reason if data_point.prediction else "",
                '投票历史记录': attempts_info,
                '最终置信度': round(data_point.prediction.confidence, 3) if data_point.prediction else 0.0,
                '备注': data_point.prediction.agreement_status if data_point.prediction else ""
            }
            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

        # 保存到Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='预测结果', index=False)

            # 添加统计信息sheet
            stats_df = self._create_statistics_sheet(results)
            stats_df.to_excel(writer, sheet_name='统计信息', index=False)

            # 添加标签探索分析sheet
            exploration_df = self._create_exploration_analysis_sheet(results)
            exploration_df.to_excel(writer, sheet_name='标签探索分析', index=False)

        print(f"\n✅ 独立投票分析结果已保存到: {output_file}")

        # 计算并保存置信度分析
        ece, pearson_corr = self.calculate_calibration_metrics(results)

        # 保存详细报告
        self._save_detailed_report(results, output_file, ece, pearson_corr)

    def _create_statistics_sheet(self, results: List[DataPoint]) -> pd.DataFrame:
        """创建统计信息sheet"""
        total = len(results)
        labeled = sum(1 for dp in results if dp.human_label)
        matched = sum(1 for dp in results if dp.prediction and dp.prediction.match_result)

        # LLM内部一致状态统计
        agreement_stats = {}
        vote_distribution = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        # 标签探索统计
        total_labels_explored = 0
        avg_labels_per_req = 0

        for dp in results:
            if dp.prediction:
                # 记录一致状态
                status = dp.prediction.agreement_status
                agreement_stats[status] = agreement_stats.get(status, 0) + 1

                # 记录投票次数分布
                attempts = dp.prediction.total_attempts
                if attempts in vote_distribution:
                    vote_distribution[attempts] += 1

                # 统计尝试过的标签数量
                tried_labels = set()
                for attempt in dp.prediction.attempts:
                    tried_labels.add(attempt["predicted_label"])
                total_labels_explored += len(tried_labels)

        avg_labels_per_req = total_labels_explored / total if total > 0 else 0

        stats_data = []
        stats_data.append(["总需求数量", total])
        stats_data.append(["有人类标注的数量", labeled])
        if labeled > 0:
            stats_data.append(["与人类标注一致的数量", matched])
            stats_data.append(["一致率", f"{(matched / labeled * 100):.1f}%"])

        stats_data.append(["", ""])  # 空行

        # 标签探索统计
        stats_data.append(["标签探索统计", ""])
        stats_data.append(["总尝试标签数（去重）", total_labels_explored])
        stats_data.append(["平均每需求尝试标签数", f"{avg_labels_per_req:.2f}"])

        stats_data.append(["", ""])  # 空行

        # LLM内部一致状态
        stats_data.append(["LLM内部一致状态统计", ""])
        for status, count in sorted(agreement_stats.items()):
            stats_data.append([status, f"{count}条 ({(count / total * 100):.1f}%)"])

        stats_data.append(["", ""])  # 空行

        # 投票次数分布
        stats_data.append(["投票次数分布", ""])
        for attempts, count in sorted(vote_distribution.items()):
            if count > 0:
                stats_data.append([f"{attempts}次投票", f"{count}条 ({(count / total * 100):.1f}%)"])

        return pd.DataFrame(stats_data, columns=["指标", "数值"])

    def _create_exploration_analysis_sheet(self, results: List[DataPoint]) -> pd.DataFrame:
        """创建标签探索分析sheet"""
        exploration_data = []

        # 统计每个需求的标签探索情况
        for i, dp in enumerate(results):
            if dp.prediction and dp.prediction.attempts:
                tried_labels = set()
                new_label_count = 0
                exploration_path = []

                for attempt in dp.prediction.attempts:
                    label = attempt["predicted_label"]
                    is_new = label not in tried_labels
                    tried_labels.add(label)

                    if is_new:
                        new_label_count += 1
                        exploration_path.append(f"{label}🌟")
                    else:
                        exploration_path.append(f"{label}")

                exploration_data.append({
                    '序号': i + 1,
                    '最终标签': dp.final_label,
                    '总投票次数': dp.prediction.total_attempts,
                    '尝试标签数': len(tried_labels),
                    '新标签数': new_label_count,
                    '探索路径': ' → '.join(exploration_path),
                    'LLM状态': dp.prediction.agreement_status
                })

        return pd.DataFrame(exploration_data)

    def _save_detailed_report(self, results: List[DataPoint], output_file: str, ece: float, pearson_corr: float):
        """保存详细分析报告"""
        report_file = output_file.replace('.xlsx', '_详细报告.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GLM模型-独立投票分类系统分析报告\n")
            f.write("特殊配置: 至少2次独立标注 → 不一致才继续 → 优先新类别 → 最多6次\n")
            f.write("人类标注: 仅用于最终检验\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总需求数量: {len(results)}\n")

            # 一致率统计
            labeled = sum(1 for dp in results if dp.human_label)
            matched = sum(1 for dp in results if dp.prediction and dp.prediction.match_result)

            f.write("\n📊 与人类标注一致率统计:\n")
            f.write("-" * 80 + "\n")
            if labeled > 0:
                f.write(f"有人类标注的样本数: {labeled}\n")
                f.write(f"一致样本数: {matched}\n")
                f.write(f"一致率: {(matched / labeled * 100):.1f}%\n")

            # 标签探索统计
            total_labels_explored = 0
            for dp in results:
                if dp.prediction:
                    tried_labels = set()
                    for attempt in dp.prediction.attempts:
                        tried_labels.add(attempt["predicted_label"])
                    total_labels_explored += len(tried_labels)

            avg_labels_per_req = total_labels_explored / len(results) if results else 0

            f.write("\n🔍 标签探索统计:\n")
            f.write("-" * 80 + "\n")
            f.write(f"总尝试标签数（去重）: {total_labels_explored}\n")
            f.write(f"平均每需求尝试标签数: {avg_labels_per_req:.2f}\n")

            # LLM内部一致统计
            agreement_stats = {}
            for dp in results:
                if dp.prediction and dp.prediction.agreement_status:
                    status = dp.prediction.agreement_status
                    agreement_stats[status] = agreement_stats.get(status, 0) + 1

            f.write("\n🤖 LLM内部一致状态统计:\n")
            f.write("-" * 80 + "\n")
            for status, count in sorted(agreement_stats.items()):
                f.write(f"{status}: {count}条 ({(count / len(results) * 100):.1f}%)\n")

            # 置信度分析
            if ece is not None and pearson_corr is not None:
                f.write("\n📈 置信度分析:\n")
                f.write("-" * 80 + "\n")
                f.write(f"ECE（预期校准误差）: {ece:.4f}\n")
                f.write(f"皮尔逊相关系数: {pearson_corr:.4f}\n")

                # 解释ECE
                if ece < 0.05:
                    f.write("ECE解释: < 0.05，模型校准度良好\n")
                elif ece < 0.1:
                    f.write("ECE解释: 0.05-0.1，模型校准度一般\n")
                else:
                    f.write("ECE解释: ≥ 0.1，模型校准度较差\n")

            # 投票效率分析
            total_votes = sum(dp.prediction.total_attempts for dp in results if dp.prediction)
            avg_votes = total_votes / len(results) if results else 0

            f.write("\n🔄 投票效率分析:\n")
            f.write("-" * 80 + "\n")
            f.write(f"总投票次数: {total_votes}\n")
            f.write(f"平均每需求投票次数: {avg_votes:.2f}\n")

            # 新标签探索效率
            total_new_labels = 0
            for dp in results:
                if dp.prediction:
                    tried_labels = set()
                    for attempt in dp.prediction.attempts:
                        if attempt.get("is_new_label", False):
                            total_new_labels += 1

            avg_new_per_vote = total_new_labels / total_votes if total_votes > 0 else 0

            f.write(f"总新标签探索次数: {total_new_labels}\n")
            f.write(f"平均每次投票探索新标签数: {avg_new_per_vote:.2f}\n")

            # 不一致案例分析
            inconsistencies = []
            for i, dp in enumerate(results):
                if dp.human_label and dp.prediction and not dp.prediction.match_result:
                    tried_labels = set()
                    for attempt in dp.prediction.attempts:
                        tried_labels.add(attempt["predicted_label"])

                    inconsistencies.append({
                        'index': i + 1,
                        'model': dp.final_label,
                        'human': dp.human_label,
                        'attempts': dp.prediction.total_attempts,
                        'tried_labels': len(tried_labels),
                        'status': dp.prediction.agreement_status,
                        'reason': dp.prediction.final_reason[:100] if dp.prediction.final_reason else ""
                    })

            if inconsistencies:
                f.write(f"\n💭 与人类标注不一致案例分析 (共{len(inconsistencies)}条):\n")
                f.write("-" * 80 + "\n")
                for inc in inconsistencies[:10]:  # 只显示前10条
                    f.write(f"第{inc['index']}条: \n")
                    f.write(
                        f"  模型预测: '{inc['model']}' (投票{inc['attempts']}次, 尝试{inc['tried_labels']}个标签)\n")
                    f.write(f"  人类标注: '{inc['human']}'\n")
                    f.write(f"  LLM状态: {inc['status']}\n")
                    if inc['reason']:
                        f.write(f"  原因: {inc['reason']}\n")
                if len(inconsistencies) > 10:
                    f.write(f"... 还有{len(inconsistencies) - 10}条不一致案例\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✅ 详细分析报告已保存到: {report_file}")

    def print_statistics(self, results: List[DataPoint]):
        """打印统计信息"""
        print(f"\n{'=' * 60}")
        print("📊 GLM模型-独立投票分类统计信息")
        print(f"🔍 标签探索优先机制")
        print(f"{'=' * 60}")

        print(f"处理总数据量: {len(results)}条需求")
        print(f"使用角色: User Experience Designer")

        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            print(f"\n⏰ 时间统计:")
            print(f"  总运行时间: {total_duration}")
            if len(results) > 0:
                avg_time_per_req = total_duration / len(results)
                print(f"  平均每条需求处理时间: {avg_time_per_req}")

        # 与人类标注一致率
        labeled = sum(1 for dp in results if dp.human_label)
        matched = sum(1 for dp in results if dp.prediction and dp.prediction.match_result)

        print(f"\n🎯 与人类标注一致率:")
        if labeled > 0:
            match_rate = matched / labeled * 100
            print(f"  有人类标注的样本数: {labeled}")
            print(f"  一致样本数: {matched}")
            print(f"  一致率: {match_rate:.1f}%")
        else:
            print(f"  无人类标注数据")

        # 标签探索统计
        total_labels_explored = 0
        total_new_labels = 0
        for dp in results:
            if dp.prediction:
                tried_labels = set()
                for attempt in dp.prediction.attempts:
                    tried_labels.add(attempt["predicted_label"])
                    if attempt.get("is_new_label", False):
                        total_new_labels += 1
                total_labels_explored += len(tried_labels)

        avg_labels_per_req = total_labels_explored / len(results) if results else 0

        print(f"\n🔍 标签探索统计:")
        print(f"  总尝试标签数（去重）: {total_labels_explored}")
        print(f"  平均每需求尝试标签数: {avg_labels_per_req:.2f}")
        print(f"  总新标签探索次数: {total_new_labels}")

        # LLM内部一致状态
        agreement_stats = {}
        for dp in results:
            if dp.prediction and dp.prediction.agreement_status:
                status = dp.prediction.agreement_status
                agreement_stats[status] = agreement_stats.get(status, 0) + 1

        print(f"\n🤖 LLM内部一致状态:")
        for status, count in sorted(agreement_stats.items()):
            percentage = count / len(results) * 100
            print(f"  {status}: {count}条 ({percentage:.1f}%)")

        # 投票次数分布
        vote_dist = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        total_votes = 0
        for dp in results:
            if dp.prediction:
                attempts = dp.prediction.total_attempts
                total_votes += attempts
                if attempts in vote_dist:
                    vote_dist[attempts] += 1

        print(f"\n🔄 投票次数分布:")
        for attempts, count in sorted(vote_dist.items()):
            if count > 0:
                percentage = count / len(results) * 100
                print(f"  {attempts}次投票: {count}条 ({percentage:.1f}%)")

        print(f"  平均投票次数: {total_votes / len(results):.2f}")


# 保留原有的DataLoader类（不变）
class DataLoader:
    """数据加载器类"""

    @staticmethod
    def load_dataset_with_labels(file_path: str) -> List[Dict]:
        """从dataset文件加载需求和人类标注"""
        try:
            df = pd.read_excel(file_path)
            dataset_with_labels = []

            if 'requirement' not in df.columns:
                print(f"⚠️  文件中没有'requirement'列，尝试使用第一列作为需求内容")
                requirement_col = df.columns[0]
            else:
                requirement_col = 'requirement'

            if 'label' not in df.columns:
                print(f"⚠️  文件中没有'label'列，将无法进行标注对比")
                label_col = None
            else:
                label_col = 'label'

            for _, row in df.iterrows():
                if pd.notna(row[requirement_col]):
                    item = {'requirement': str(row[requirement_col]).strip()}
                    if label_col and pd.notna(row.get(label_col)):
                        item['label'] = str(row[label_col]).strip()
                    dataset_with_labels.append(item)

            print(f"✅ 成功加载 {len(dataset_with_labels)} 条带标签的测试需求")
            return dataset_with_labels

        except Exception as e:
            print(f"❌ 加载数据集文件出错: {e}")
            return []

    @staticmethod
    def load_categories_and_explanations(file_path: str) -> Dict[str, str]:
        """从concept文件加载类别和解释"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            categories = {}
            for _, row in df.iterrows():
                if pd.notna(row['category']):
                    category = str(row['category']).strip()
                    explanation = str(row['explanation']).strip() if pd.notna(row.get('explanation')) else ""
                    categories[category] = explanation
            print(f"✅ 成功加载 {len(categories)} 个分类标签")
            return categories
        except Exception as e:
            print(f"❌ 加载类别文件出错: {e}")
            return {}


def main():
    """主函数"""
    # 文件路径配置
    dataset_file = "dataset.xlsx"
    concept_file = "1123Concept.xlsx"
    output_file = "glm_UXDesigner_priority_new_labels.xlsx"

    data_loader = DataLoader()

    print("=" * 80)
    print("🎨 GLM模型-独立投票分类系统（优先新类别）")
    print("🔄 投票机制: 至少2次独立标注 → 不一致才继续")
    print("🌟 重要特性: 后续标注优先考虑新类别")
    print("🎯 人类标注: 仅用于最终检验")
    print("📊 包含: 准确率 + LLM内部一致率 + 标签探索统计 + ECE + 皮尔逊检验")
    print("🌐 使用模型: 智谱AI GLM-4.5（通过阿里云DashScope调用）")
    print("⚠️  注意: 使用流式模式兼容GLM模型")
    print("=" * 80)

    # 检查环境变量
    api_key = os.getenv('DASHSCOPE_API_KEY')

    if not api_key:
        print("❌ DASHSCOPE_API_KEY 未设置")
        print("💡 请设置环境变量 DASHSCOPE_API_KEY")
        return

    # 检查数据文件
    if not all(os.path.exists(f) for f in [dataset_file, concept_file]):
        print("❌ 数据文件不存在")
        return

    try:
        # 创建单角色客户端
        print(f"\n🚀 创建单角色GLM模型客户端...")
        single_role_client = SingleRoleGLMClient(
            name="GLM-UXDesigner-PriorityNew",
            model_name="glm-4.5"
        )
    except Exception as e:
        print(f"❌ 创建GLM模型客户端失败: {e}")
        return

    # 加载数据
    print(f"\n📚 加载数据文件...")
    category_explanations = data_loader.load_categories_and_explanations(concept_file)
    labels = list(category_explanations.keys())
    dataset_with_labels = data_loader.load_dataset_with_labels(dataset_file)

    if not dataset_with_labels or not labels:
        print("❌ 数据加载失败")
        return

    print(f"\n✅ 数据加载完成:")
    print(f"  分类标签: {len(labels)} 个")
    print(f"  测试需求: {len(dataset_with_labels)} 条")

    # 执行处理
    print(f"\n🚀 开始独立投票分类处理（优先新类别）...")
    processor = SingleRoleProcessor(single_role_client)
    results = processor.process_dataset(dataset_with_labels, labels)

    # 保存和显示结果
    processor.save_results(results, output_file)
    processor.print_statistics(results)

    # 显示示例结果
    print(f"\n🔍 前3个结果的详细信息:")
    for i, data_point in enumerate(results[:3]):
        print(f"\n{'=' * 60}")
        print(f"{i + 1}. 需求: {data_point.content[:80]}...")
        print(f"   LLM内部状态: {data_point.prediction.agreement_status if data_point.prediction else 'N/A'}")
        print(f"   模型最终预测: {data_point.final_label}")
        print(f"   最终置信度: {data_point.prediction.confidence:.3f if data_point.prediction else 0.0}")
        print(f"   总投票次数: {data_point.prediction.total_attempts if data_point.prediction else 0}")

        if data_point.prediction and data_point.prediction.attempts:
            print(f"   投票历史（🌟表示新标签）:")
            for attempt in data_point.prediction.attempts:
                is_new = attempt.get("is_new_label", False)
                star = "🌟" if is_new else ""
                print(f"     第{attempt['attempt_number']}次: {attempt['predicted_label']}{star} "
                      f"(置信度: {attempt['confidence']:.3f})")

        if data_point.human_label:
            print(f"   人类标注: {data_point.human_label}")
            print(f"   是否一致: {'✅ 是' if data_point.prediction.match_result else '❌ 否'}")

    print(f"\n{'=' * 80}")
    print("🎉 GLM模型独立投票分类处理完成!")
    print(f"📊 结果文件: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()