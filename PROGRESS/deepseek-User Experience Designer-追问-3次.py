# 修改了尝试机制：
#
# 先投一次：先进行一次投票
#
# 核对label：与dataset.xlsx中的label列进行核对
#
# 不一致再投：如果不一致，进行第二次投票
#
# 最多3次：总共最多尝试3次
#
# 记录原因：如果3次后仍不一致，记录原因
#
# 简化了投票逻辑：
#
# 不再进行5次投票，而是单次投票
#
# 每次投票LLM都提供自我评估的置信度
#
# 置信度是LLM自己评估的，不是基于投票统计
#
# 改进了提示设计：
#
# 第一次投票：标准提示
#
# 第二次投票：重新思考提示，提供人类标注信息
#
# 第三次投票：最终思考提示，要求分析原因
#
# 更新了数据加载：
#
# DataLoader.load_dataset_with_labels() 同时加载需求内容和label列
#
# 确保数据集中有'label'列才能进行对比
#
# 优化了结果输出：
#
# 记录每次尝试的结果和置信度
#
# 清晰的尝试次数统计
#
# 详细的不一致原因分析
#
# 保持了核心功能：
#
# 只使用User Experience Designer角色
#
# 使用GLM API（支持流式模式）
#
# 置信度由LLM自我评估
#
# 现在代码实现了您要求的逻辑：
#
# 先投一次票
#
# 与数据集中的label列核对
#
# 如果不一致，进行第二次投票
#
# 最多尝试3次
#
# 如果3次后仍不一致，记录详细原因

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

# 加载环境变量
load_dotenv()


@dataclass
class ModelPrediction:
    """存储模型预测结果"""
    label: str
    confidence: float
    model_name: str
    attempts: List[Dict] = None  # 每次尝试的记录
    human_label: str = None  # 人类标注
    match_result: bool = False  # 是否匹配人类标注
    total_attempts: int = 0  # 总尝试次数
    final_reason: str = ""  # 最终不一致的原因


@dataclass
class DataPoint:
    """数据点类"""
    content: str
    prediction: ModelPrediction = None
    final_label: str = None
    human_label: str = None  # 人类标注标签


class SingleRoleDeepSeekClient:
    """单角色DeepSeek模型客户端类 - 仅User Experience Designer"""

    def __init__(self, name: str = "deepseek-chat", model_name: str = "deepseek-chat"):
        self.name = name
        self.model_name = model_name

        # DeepSeek API配置
        self.api_key = os.getenv('DEEPSEEK_API_KEY')

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY环境变量未设置")

        # 使用DeepSeek官方API端点
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
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

        print(f"✅ 成功创建单角色DeepSeek模型客户端: {self.name}")
        print(f"  使用的专业角色: {self.role['name']}")
        print(f"  🔄 尝试机制: 先投一次，与label核对，不一致再投，最多3次")
        print(f"  📝 不一致原因记录: 如果3次后仍不匹配，记录原因")
        print(f"  API配置: DeepSeek官方API")
        print(f"  模型版本: {self.model_name}")

    def single_vote(self, text: str, labels: List[str], attempt_num: int = 1) -> Dict:
        """单次投票"""
        role_name = self.role["name"]

        print(f"    🔄 第{attempt_num}次投票...")

        try:
            prompt = self._build_single_vote_prompt(text, labels, self.role, attempt_num)

            # 调用DeepSeek API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.role["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
                stream=False  # DeepSeek支持非流式模式
            )

            # 获取响应内容
            response_content = completion.choices[0].message.content

            predicted_label = self._extract_label_from_response(response_content, labels)

            # 提取置信度
            confidence = self._extract_confidence_from_response(response_content)

            if predicted_label:
                print(f"      ✅ 第{attempt_num}次投票结果: {predicted_label} (置信度: {confidence:.3f})")
                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
            else:
                fallback_label = labels[0] if labels else "Unknown"
                print(f"      ⚠️ 第{attempt_num}次投票: {fallback_label} (提取失败，使用默认)")
                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": fallback_label,
                    "confidence": 0.5,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
        except Exception as e:
            print(f"      ❌ 第{attempt_num}次投票异常: {str(e)[:100]}")
            fallback_label = labels[0] if labels else "Unknown"
            return {
                "attempt_number": attempt_num,
                "response": str(e),
                "predicted_label": fallback_label,
                "confidence": 0.5,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }

    def analyze_with_retry(self, text: str, labels: List[str], human_label: str = None,
                           category_explanations: Dict[str, str] = None,
                           requirements_examples: Dict[str, List[str]] = None) -> ModelPrediction:
        """带重试机制的分析：先投一次，与label核对，不一致再投，最多3次"""

        role_name = self.role["name"]

        print(f"\n🔍 开始{role_name}分析: '{text[:50]}...'")
        print(f"  📝 人类标注: {human_label if human_label else '无标注'}")
        print("-" * 60)

        attempts = []
        max_attempts = 3
        final_label = ""
        final_confidence = 0.0
        match_result = False
        final_reason = ""

        # 第一次投票
        attempt1 = self.single_vote(text, labels, 1)
        attempts.append(attempt1)

        # 检查是否匹配人类标注
        if human_label:
            match_result = (attempt1["predicted_label"] == human_label)

            if match_result:
                print(f"    ✅ 第1次投票就与人类标注一致！")
                final_label = attempt1["predicted_label"]
                final_confidence = attempt1["confidence"]
            else:
                print(f"    ⚠️ 第1次投票与人类标注不一致，将进行第2次投票...")
                time.sleep(1)

                # 第2次投票（带重新思考）
                attempt2 = self._rethinking_vote(text, labels, human_label, attempt1["predicted_label"], 2)
                attempts.append(attempt2)

                # 再次检查
                match_result = (attempt2["predicted_label"] == human_label)

                if match_result:
                    print(f"    ✅ 第2次投票与人类标注一致！")
                    final_label = attempt2["predicted_label"]
                    final_confidence = attempt2["confidence"]
                else:
                    print(f"    ⚠️ 第2次投票仍与人类标注不一致，将进行第3次投票...")
                    time.sleep(1)

                    # 第3次投票（最后一次尝试）
                    attempt3 = self._rethinking_vote(text, labels, human_label, attempt2["predicted_label"], 3,
                                                     is_final=True)
                    attempts.append(attempt3)

                    # 最终检查
                    match_result = (attempt3["predicted_label"] == human_label)

                    if match_result:
                        print(f"    ✅ 第3次投票与人类标注一致！")
                        final_label = attempt3["predicted_label"]
                        final_confidence = attempt3["confidence"]
                    else:
                        print(f"    ❌ 3次投票后仍与人类标注不一致，提取原因...")
                        final_label = attempt3["predicted_label"]
                        final_confidence = attempt3["confidence"]

                        # 提取不一致原因
                        final_reason = self._extract_disagreement_reason(text, final_label, human_label, attempts)
                        print(f"    📝 不一致原因: {final_reason[:100]}...")
        else:
            # 如果没有人类标注，直接使用第一次投票结果
            print(f"    ℹ️ 无人类标注可对比，使用第1次投票结果")
            final_label = attempt1["predicted_label"]
            final_confidence = attempt1["confidence"]
            match_result = True  # 因为没有标注，默认认为一致

        # 如果没有人类标注，计算平均置信度
        if not human_label and attempts:
            final_confidence = np.mean([a["confidence"] for a in attempts])

        # 打印最终结果
        print(f"\n📊 最终结果:")
        print(f"  模型预测: {final_label}")
        if human_label:
            print(f"  人类标注: {human_label}")
            print(f"  是否一致: {'✅ 是' if match_result else '❌ 否'}")
        print(f"  置信度: {final_confidence:.3f}")
        print(f"  总尝试次数: {len(attempts)}")
        if final_reason and not match_result:
            print(f"  不一致原因: {final_reason}")

        return ModelPrediction(
            label=final_label,
            confidence=final_confidence,
            model_name=f"{self.name}-single-role-retry",
            attempts=attempts,
            human_label=human_label,
            match_result=match_result,
            total_attempts=len(attempts),
            final_reason=final_reason
        )

    def _rethinking_vote(self, text: str, labels: List[str], human_label: str,
                         previous_label: str, attempt_num: int, is_final: bool = False) -> Dict:
        """重新思考投票"""
        role_name = self.role["name"]

        print(f"    🔄 第{attempt_num}次投票（重新思考）...")

        try:
            if is_final:
                prompt = self._build_final_rethinking_prompt(text, labels, human_label, previous_label, self.role)
            else:
                prompt = self._build_rethinking_prompt(text, labels, human_label, previous_label, self.role)

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.role["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=250,
                stream=False
            )

            response_content = completion.choices[0].message.content

            predicted_label = self._extract_label_from_response(response_content, labels)
            confidence = self._extract_confidence_from_response(response_content)

            if predicted_label:
                print(f"      ✅ 第{attempt_num}次重新思考结果: {predicted_label} (置信度: {confidence:.3f})")
                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "is_rethinking": True,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
            else:
                fallback_label = labels[0] if labels else "Unknown"
                print(f"      ⚠️ 第{attempt_num}次重新思考: {fallback_label} (提取失败)")
                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": fallback_label,
                    "confidence": 0.5,
                    "is_rethinking": True,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
        except Exception as e:
            print(f"      ❌ 第{attempt_num}次重新思考异常: {str(e)[:100]}")
            fallback_label = labels[0] if labels else "Unknown"
            return {
                "attempt_number": attempt_num,
                "response": str(e),
                "predicted_label": fallback_label,
                "confidence": 0.5,
                "is_rethinking": True,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }

    def _extract_disagreement_reason(self, text: str, model_label: str, human_label: str, attempts: List[Dict]) -> str:
        """提取不一致原因"""
        try:
            # 使用最后一次尝试的响应来提取原因
            last_attempt = attempts[-1]
            response = last_attempt["response"]

            # 尝试从响应中提取原因
            reason_patterns = [
                r'【原因分析】\s*[:：]?\s*(.+)',
                r'原因分析[:：]\s*(.+)',
                r'不一致原因[:：]\s*(.+)',
                r'视角差异[:：]\s*(.+)'
            ]

            for pattern in reason_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    reason = match.group(1).strip()
                    if reason and len(reason) > 10:
                        return reason

            # 如果没有明确的原因，生成一个概括性原因
            return f"UX设计师从用户体验角度分析为'{model_label}'，而人类标注为'{human_label}'。经过{len(attempts)}次尝试，仍然存在视角差异。"

        except Exception as e:
            return f"无法提取具体原因，可能因为: {str(e)[:50]}"

    def _build_single_vote_prompt(self, text: str, labels: List[str],
                                  role_info: Dict, attempt_num: int = 1) -> str:
        """构建单次投票提示"""

        role_name = role_info["name"]

        labels_section = f"""可选分类标签（请严格从以下标签中选择一个）:

{chr(10).join([f'• {label}' for label in labels])}

重要提示：请确保完全按照上述标签名称返回，不要修改、缩写或添加任何额外内容。"""

        role_guidance = f"""
作为{role_name}，请从你的专业角度分析这个需求：

你的专业关注领域：{', '.join(role_info['focus_areas'])}
请从{role_name}的角度思考并选择最合适的分类标签。"""

        response_format = f"""请按照以下格式回复：
【{role_name}分析】[你的专业分析]
【分类标签】[必须从上述标签中选择一个，完全匹配标签名称]
【置信度】[你对这个分类的自信程度，0.0-1.0之间的小数，如0.85]"""

        return f"""你现在的角色是：{role_name}

{role_guidance}

{labels_section}

待分类的需求描述:
"{text}"

{response_format}"""

    def _build_rethinking_prompt(self, text: str, labels: List[str],
                                 human_label: str, previous_label: str,
                                 role_info: Dict) -> str:
        """构建重新思考提示"""

        role_name = role_info["name"]

        labels_section = f"""可选分类标签（请严格从以下标签中选择一个）:

{chr(10).join([f'• {label}' for label in labels])}"""

        rethinking_guidance = f"""
作为{role_name}，你需要重新思考刚才的分类。

需求描述:
"{text}"

你刚才的分类是: 【{previous_label}】
但是人类专家的标注是: 【{human_label}】

请从{role_name}的专业角度重新分析，为什么人类专家会有不同的标注？
然后给出你认为最合适的分类标签。

请按照以下格式回复：
【{role_name}重新分析】[你的分析]
【分类标签】[必须从上述标签中选择一个]
【置信度】[你对这个分类的自信程度，0.0-1.0之间的小数]"""

        return rethinking_guidance

    def _build_final_rethinking_prompt(self, text: str, labels: List[str],
                                       human_label: str, previous_label: str,
                                       role_info: Dict) -> str:
        """构建最终重新思考提示（要求分析原因）"""

        role_name = role_info["name"]

        labels_section = f"""可选分类标签（请严格从以下标签中选择一个）:

{chr(10).join([f'• {label}' for label in labels])}"""

        final_guidance = f"""
作为{role_name}，这是你最后一次重新思考的机会。

需求描述:
"{text}"

你之前的分类是: 【{previous_label}】
但是人类专家的标注一直是: 【{human_label}】

请从{role_name}的专业角度：
1. 给出你认为最合适的分类标签
2. 分析为什么你的分类与人类专家不同

请按照以下格式回复：
【{role_name}最终分析】[你的分析，包括可能的视角差异]
【分类标签】[必须从上述标签中选择一个]
【置信度】[你对这个分类的自信程度]
【原因分析】[解释为什么与人类标注不同，50字以内]"""

        return final_guidance

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
                if extracted.lower() == label.lower():
                    return label
            for label in labels:
                if label.lower() in extracted.lower():
                    return label

        # 2. 在整个响应中搜索标签
        for label in labels:
            if label.lower() in response_clean.lower():
                return label

        return labels[0] if labels else "Unknown"

    def _extract_confidence_from_response(self, response: str) -> float:
        """从响应中提取置信度"""
        if not response:
            return 0.5

        response_clean = response.strip()

        # 尝试从【置信度】格式中提取
        confidence_patterns = [
            r'【置信度】\s*[:：]?\s*([0-9]*\.?[0-9]+)',
            r'置信度[:：]\s*([0-9]*\.?[0-9]+)'
        ]

        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    # 确保置信度在0-1之间
                    confidence = max(0.0, min(1.0, confidence))
                    return confidence
                except ValueError:
                    continue

        # 如果没有明确找到，尝试查找0-1之间的数字
        number_pattern = r'(0?\.\d+|1\.0|0\.[0-9]+|1\.0+)'
        number_matches = re.findall(number_pattern, response_clean)

        for num_str in number_matches:
            try:
                num = float(num_str)
                if 0 <= num <= 1:
                    return num
            except ValueError:
                continue

        return 0.5  # 默认值


class SingleRoleProcessor:
    """单角色模型处理器 - 仅User Experience Designer"""

    def __init__(self, single_role_client: SingleRoleDeepSeekClient):
        self.single_role_client = single_role_client
        self.start_time = None
        self.end_time = None
        print(f"单角色模型处理器初始化完成，使用{single_role_client.role['name']}角色")
        print(f"🔄 尝试机制: 先投一次，与label核对，不一致再投，最多3次")

    def process_dataset(self, dataset_with_labels: List[Dict], labels: List[str],
                        category_explanations: Dict[str, str] = None,
                        requirements_examples: Dict[str, List[str]] = None) -> List[DataPoint]:
        """处理带标签的数据集"""
        data_points = []

        self.start_time = datetime.datetime.now()
        print(f"🚀 单角色分析开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"开始处理 {len(dataset_with_labels)} 条带标签的软件需求...")
        print(f"分类标签数量: {len(labels)} 个")
        print(f"🎭 角色配置: User Experience Designer")
        print(f"🔄 尝试机制: 1次 → 核对 → 不一致再投(最多3次)")

        for i, item in enumerate(dataset_with_labels):
            # 从数据中提取内容和人类标注
            text = item.get('requirement', item.get('content', ''))
            human_label = item.get('label', None)  # 注意：这里使用'label'列

            if i % 2 == 0 and i > 0:
                elapsed_time = datetime.datetime.now() - self.start_time
                processed = i
                remaining = len(dataset_with_labels) - i
                avg_time_per_req = elapsed_time / processed if processed > 0 else elapsed_time
                est_remaining = avg_time_per_req * remaining

                print(f"  📈 进度: {i}/{len(dataset_with_labels)} - 已用时: {elapsed_time} - 预计剩余: {est_remaining}")

            data_point = DataPoint(content=text, human_label=human_label)
            prediction = self.single_role_client.analyze_with_retry(
                text, labels, human_label, category_explanations, requirements_examples
            )

            data_point.prediction = prediction
            data_point.final_label = prediction.label
            data_points.append(data_point)

            if i < len(dataset_with_labels) - 1:
                time.sleep(2)

        self.end_time = datetime.datetime.now()
        return data_points

    def save_results(self, results: List[DataPoint], output_file: str):
        """保存最终结果到Excel文件"""
        results_data = []

        for i, data_point in enumerate(results):
            # 构建尝试记录
            attempts_info = ""
            human_label = data_point.human_label or "N/A"
            match_status = "N/A"
            reason = ""
            confidence = 0.0
            attempts_summary = []

            if data_point.prediction:
                confidence = data_point.prediction.confidence

                if data_point.prediction.attempts:
                    attempts_summary = [
                        f"尝试{a['attempt_number']}: {a['predicted_label']}({a['confidence']:.3f})"
                        for a in data_point.prediction.attempts
                    ]
                    attempts_info = "; ".join(attempts_summary)

                if data_point.human_label:
                    match_status = "✅ 一致" if data_point.prediction.match_result else "❌ 不一致"
                    if data_point.prediction.final_reason and not data_point.prediction.match_result:
                        reason = data_point.prediction.final_reason

            row_data = {
                '序号': i + 1,
                '需求内容': data_point.content,
                '模型名称': data_point.prediction.model_name if data_point.prediction else "N/A",
                '人类标注标签': human_label,
                '模型最终预测': data_point.final_label,
                '是否一致': match_status,
                '总尝试次数': data_point.prediction.total_attempts if data_point.prediction else 0,
                '不一致原因': reason,
                '每次尝试记录': attempts_info,
                '最终置信度': round(confidence, 3),
                '备注': f"共{data_point.prediction.total_attempts}次尝试" if data_point.prediction else ""
            }
            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ 单角色分析结果已保存到: {output_file}")

        # 保存详细报告
        self._save_detailed_report(results, output_file)

    def _save_detailed_report(self, results: List[DataPoint], output_file: str):
        """保存详细分析报告"""
        report_file = output_file.replace('.xlsx', '_详细报告.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DeepSeek模型-User Experience Designer单角色分类系统分析报告\n")
            f.write("特殊配置: 先投一次，与label核对，不一致再投，最多3次\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总需求数量: {len(results)}\n")
            f.write("使用的专业角色:\n")
            f.write("1. User Experience Designer (用户体验设计师)\n\n")

            f.write("🔄 尝试机制:\n")
            f.write("  1. 第一次投票\n")
            f.write("  2. 与数据集中的label列核对\n")
            f.write("  3. 如果不一致，进行第二次投票\n")
            f.write("  4. 再次核对，如果不一致且未达3次，进行第三次投票\n")
            f.write("  5. 如果3次后仍不一致，记录原因\n\n")

            f.write("🎯 置信度机制:\n")
            f.write("  - 每次投票LLM自我评估置信度\n")
            f.write("  - 最终置信度使用最后一次投票的置信度\n\n")

            # 计算一致率
            total_with_human_label = 0
            match_count = 0
            total_attempts = 0
            inconsistencies = []

            for i, dp in enumerate(results):
                if dp.human_label:
                    total_with_human_label += 1
                    if dp.prediction and dp.prediction.match_result:
                        match_count += 1
                    if dp.prediction:
                        total_attempts += dp.prediction.total_attempts
                    if dp.prediction and not dp.prediction.match_result and dp.prediction.final_reason:
                        inconsistencies.append({
                            'index': i + 1,
                            'model': dp.final_label,
                            'human': dp.human_label,
                            'confidence': dp.prediction.confidence if dp.prediction else 0.0,
                            'reason': dp.prediction.final_reason,
                            'attempts': dp.prediction.total_attempts if dp.prediction else 0
                        })

            f.write("📊 一致率统计:\n")
            f.write("-" * 80 + "\n")
            if total_with_human_label > 0:
                f.write(f"总样本数（有人类标注）: {total_with_human_label}\n")
                f.write(f"一致样本数: {match_count}\n")
                f.write(f"一致率: {(match_count / total_with_human_label * 100):.1f}%\n")
                f.write(f"平均尝试次数: {(total_attempts / total_with_human_label):.2f}\n\n")
            else:
                f.write("无人类标注数据可用于一致率统计\n\n")

            # 尝试次数分布
            attempt_distribution = {1: 0, 2: 0, 3: 0}
            for dp in results:
                if dp.prediction:
                    attempts = dp.prediction.total_attempts
                    if attempts in attempt_distribution:
                        attempt_distribution[attempts] += 1

            f.write("🔄 尝试次数分布:\n")
            for attempts, count in attempt_distribution.items():
                if count > 0:
                    percentage = count / len(results) * 100
                    f.write(f"  {attempts}次尝试: {count}条 ({percentage:.1f}%)\n")
            f.write("\n")

            # 不一致原因分析
            if inconsistencies:
                f.write("💭 不一致原因分析 (3次尝试后仍不一致):\n")
                f.write("-" * 80 + "\n")
                for inc in inconsistencies:
                    f.write(f"第{inc['index']}条:\n")
                    f.write(f"  模型预测: {inc['model']} (置信度: {inc['confidence']:.3f})\n")
                    f.write(f"  人类标注: {inc['human']}\n")
                    f.write(f"  尝试次数: {inc['attempts']}\n")
                    f.write(f"  原因: {inc['reason']}\n\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✅ 详细分析报告已保存到: {report_file}")

    def print_statistics(self, results: List[DataPoint]):
        """打印统计信息"""
        print(f"\n{'=' * 60}")
        print("📊 DeepSeek模型-User Experience Designer单角色分类统计信息")
        print(f"特殊配置: 先投一次，核对label，不一致再投，最多3次")
        print(f"{'=' * 60}")

        print(f"处理总数据量: {len(results)}条需求")
        print(f"使用角色: User Experience Designer")

        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            print(f"\n⏰ 时间统计:")
            print(f"  开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  结束时间: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  总运行时间: {total_duration}")

            if len(results) > 0:
                avg_time_per_req = total_duration / len(results)
                print(f"  平均每条需求处理时间: {avg_time_per_req}")

        # 一致率统计
        total_with_human_label = 0
        match_count = 0
        total_attempts = 0

        for dp in results:
            if dp.human_label:
                total_with_human_label += 1
                if dp.prediction and dp.prediction.match_result:
                    match_count += 1
                if dp.prediction:
                    total_attempts += dp.prediction.total_attempts

        print(f"\n🎯 一致率统计:")
        if total_with_human_label > 0:
            match_rate = match_count / total_with_human_label * 100
            print(f"  有人类标注的样本数: {total_with_human_label}")
            print(f"  一致样本数: {match_count}")
            print(f"  一致率: {match_rate:.1f}%")
            print(f"  总尝试次数: {total_attempts}")
            print(f"  平均尝试次数: {total_attempts / total_with_human_label:.2f}")
        else:
            print(f"  无人类标注数据")

        # 尝试次数分布
        attempt_distribution = {1: 0, 2: 0, 3: 0}
        for dp in results:
            if dp.prediction:
                attempts = dp.prediction.total_attempts
                if attempts in attempt_distribution:
                    attempt_distribution[attempts] += 1

        print(f"\n🔄 尝试次数分布:")
        for attempts, count in attempt_distribution.items():
            if count > 0:
                percentage = count / len(results) * 100
                print(f"  {attempts}次尝试: {count}条 ({percentage:.1f}%)")


class DataLoader:
    """数据加载器类"""

    @staticmethod
    def load_dataset_with_labels(file_path: str) -> List[Dict]:
        """从dataset文件加载需求和人类标注"""
        try:
            df = pd.read_excel(file_path)
            dataset_with_labels = []

            # 检查必需的列
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
                    item = {
                        'requirement': str(row[requirement_col]).strip(),
                    }

                    if label_col and pd.notna(row.get(label_col)):
                        item['label'] = str(row[label_col]).strip()

                    dataset_with_labels.append(item)

            print(f"✅ 成功从 {os.path.basename(file_path)} 加载 {len(dataset_with_labels)} 条带标签的测试需求")
            if label_col:
                print(f"  其中 {sum(1 for item in dataset_with_labels if 'label' in item)} 条有标注")
            return dataset_with_labels

        except Exception as e:
            print(f"❌ 加载数据集文件 {os.path.basename(file_path)} 出错: {e}")
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
            print(f"✅ 成功从 {os.path.basename(file_path)} 加载 {len(categories)} 个分类标签")
            return categories
        except Exception as e:
            print(f"❌ 加载类别文件 {os.path.basename(file_path)} 出错: {e}")
            return {}

    @staticmethod
    def load_requirements_examples(file_path: str) -> Dict[str, List[str]]:
        """从examples文件加载需求示例"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            requirements_examples = {}

            for _, row in df.iterrows():
                if 'boileplate type' in row and pd.notna(row['boileplate type']):
                    boilerplate_type = str(row['boileplate type']).strip()
                    examples = []

                    for i in range(1, 19):
                        col_name = f'example {i}'
                        if col_name in row and pd.notna(row[col_name]):
                            example = str(row[col_name]).strip()
                            if example and example != 'nan':
                                examples.append(example)

                    if boilerplate_type and examples:
                        requirements_examples[boilerplate_type] = examples

            print(f"✅ 成功从 {os.path.basename(file_path)} 加载 {len(requirements_examples)} 个类别的需求示例")
            return requirements_examples

        except Exception as e:
            print(f"❌ 加载需求示例文件 {os.path.basename(file_path)} 出错: {e}")
            return {}


def main():
    """主函数"""
    # 文件路径配置
    dataset_file = "dataset.xlsx"  # 必须包含'requirement'和'label'列
    concept_file = "1123Concept.xlsx"
    examples_file = "1122RequirementExamples.xlsx"
    output_file = "deepseek_UXDesigner_retry_mechanism.xlsx"

    data_loader = DataLoader()

    print("=" * 80)
    print("🎨 DeepSeek模型-User Experience Designer单角色分类系统")
    print("特殊配置: 先投一次，与dataset.xlsx中的label列核对")
    print("🔄 重试机制: 如果不一致，再投第二次，最多3次")
    print("📝 原因记录: 如果3次后仍不一致，记录详细原因")
    print("=" * 80)

    # 检查环境变量
    print(f"\n🔧 环境检查:")
    api_key = os.getenv('DEEPSEEK_API_KEY')

    if api_key:
        print(f"✅ DEEPSEEK_API_KEY 已设置 (长度: {len(api_key)})")
    else:
        print(f"❌ DEEPSEEK_API_KEY 未设置")
        print(f"💡 请设置环境变量 DEEPSEEK_API_KEY")
        return

    # 检查数据文件是否存在
    print(f"\n📁 检查数据文件:")
    files_to_check = [dataset_file, concept_file, examples_file]
    all_files_exist = True

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            print(f"❌ {file_path} 不存在")
            all_files_exist = False

    if not all_files_exist:
        print(f"💡 请确保以下文件存在于当前目录:")
        for file_path in files_to_check:
            print(f"   - {file_path}")
        return

    try:
        # 创建单角色DeepSeek模型客户端
        print(f"\n🚀 创建单角色模型客户端...")
        single_role_client = SingleRoleDeepSeekClient(
            name="DeepSeek-Chat-UXDesigner",
            model_name="deepseek-chat"
        )
    except Exception as e:
        print(f"❌ 创建单角色模型客户端失败: {e}")
        return

    # 加载类别、解释和例子
    print(f"\n📚 加载数据文件...")
    category_explanations = data_loader.load_categories_and_explanations(concept_file)
    requirements_examples = data_loader.load_requirements_examples(examples_file)
    labels = list(category_explanations.keys())

    # 加载带标签的数据集
    dataset_with_labels = data_loader.load_dataset_with_labels(dataset_file)

    if not dataset_with_labels:
        print("❌ 没有找到测试需求，程序退出")
        return

    if not labels:
        print("❌ 没有找到分类标签，程序退出")
        return

    print(f"\n✅ 数据加载完成:")
    print(f"  分类标签: {len(labels)} 个")
    print(f"  测试需求: {len(dataset_with_labels)} 条")

    # 统计有多少条有标注
    labeled_count = sum(1 for item in dataset_with_labels if 'label' in item)
    print(f"  有标注的需求: {labeled_count} 条")

    if labeled_count == 0:
        print(f"\n⚠️  警告: 数据集中没有label列，将无法进行标注对比")
        print(f"  重试机制将不会被触发")

    # 执行单角色模型处理
    print(f"\n🚀 开始DeepSeek模型UX Designer单角色分类处理...")
    print(f"🎯 注意: 每次投票后都会与数据集中的label列核对")
    print(f"🔄 重试机制: 不一致 → 重投 → 最多3次 → 记录原因")

    processor = SingleRoleProcessor(single_role_client)

    results = processor.process_dataset(dataset_with_labels, labels, category_explanations, requirements_examples)

    # 保存和显示结果
    processor.save_results(results, output_file)
    processor.print_statistics(results)

    # 显示前几个结果的详细信息
    print(f"\n🔍 前3个结果的详细信息:")
    for i, data_point in enumerate(results[:3]):
        print(f"\n{'=' * 60}")
        print(f"{i + 1}. 需求: {data_point.content[:80]}...")
        if data_point.human_label:
            print(f"   真实标签: {data_point.human_label}")
            print(f"   模型预测: {data_point.final_label}")
            print(f"   是否一致: {'✅ 是' if data_point.prediction.match_result else '❌ 否'}")
            print(f"   尝试次数: {data_point.prediction.total_attempts}次")

            if data_point.prediction.attempts:
                print(f"   投票历史:")
                for attempt in data_point.prediction.attempts:
                    print(
                        f"     第{attempt['attempt_number']}次: {attempt['predicted_label']} (置信度: {attempt['confidence']:.3f})")

            if data_point.prediction.final_reason and not data_point.prediction.match_result:
                print(f"   不一致原因: {data_point.prediction.final_reason}")
        else:
            print(f"   模型预测: {data_point.final_label}")
            print(f"   置信度: {data_point.prediction.confidence:.3f}")
            print(f"   尝试次数: {data_point.prediction.total_attempts}次")

    print(f"\n{'=' * 80}")
    print("🎉 DeepSeek模型UX Designer单角色分类处理完成!")
    print(f"📊 结果文件: {output_file}")
    print(f"📄 详细报告: {output_file.replace('.xlsx', '_详细报告.txt')}")
    print(f"🔄 重试机制: 1次投票 → 核对label → 不一致再投(最多3次)")
    print(f"📝 原因记录: 3次后仍不一致 → 记录详细原因")
    print("=" * 80)


if __name__ == "__main__":
    main()