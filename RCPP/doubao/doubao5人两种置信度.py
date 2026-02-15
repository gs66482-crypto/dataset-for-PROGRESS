import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark
import time
import re
import datetime
from collections import Counter

# 加载环境变量
load_dotenv()


@dataclass
class ModelPrediction:
    """存储模型预测结果"""
    label: str
    confidence: float
    model_name: str
    votes: List[str] = None
    vote_counts: Dict[str, int] = None
    role_votes: Dict[str, str] = None
    calculation_details: Dict[str, Any] = None


@dataclass
class DataPoint:
    """数据点类"""
    content: str
    prediction: ModelPrediction = None
    final_label: str = None


class ResearchPaperConfidenceCalculator:
    """完全按照研究论文的方法计算置信度"""

    def __init__(self):
        """
        完全按照论文公式：
        C_base = V_top / N
        B_gap = min(0.2, (V_top - V_second) × 0.1)
        C_final = min(1.0, C_base + B_gap)

        其中：
        N = 5 (五种角色投票)
        K = 0.1 (缩放因子)
        U = 0.2 (Gap上限)
        """
        self.K = 0.1  # 缩放因子
        self.U = 0.2  # Gap上限

    def calculate_confidence(self, vote_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        按照研究论文的方法计算置信度
        返回包含计算过程的字典
        """
        if not vote_counts:
            return {
                'final_confidence': 0.0,
                'c_base': 0.0,
                'b_gap': 0.0,
                'v_top': 0,
                'v_second': 0,
                'total_votes': 5,
                'gap': 0
            }

        N = 5  # 总票数固定为5
        sorted_counts = sorted(vote_counts.values(), reverse=True)

        V_top = sorted_counts[0]
        V_second = sorted_counts[1] if len(sorted_counts) > 1 else 0

        # 1. 计算基础置信度 C_base (公式1)
        C_base = V_top / N

        # 2. 计算Gap Bonus B_gap (公式2)
        gap = V_top - V_second
        B_gap = min(self.U, gap * self.K)

        # 3. 计算最终置信度 C_final (公式3)
        C_final = min(1.0, C_base + B_gap)

        return {
            'final_confidence': C_final,
            'c_base': C_base,
            'b_gap': B_gap,
            'v_top': V_top,
            'v_second': V_second,
            'total_votes': N,
            'gap': gap
        }


class MultiRoleDoubaoClient:
    """多角色豆包模型客户端类 - 基于研究论文的五种专业角色"""

    def __init__(self, name: str = "doubao-seed-1-6-250615", model_name: str = "doubao-seed-1-6-250615"):
        self.name = name
        self.model_name = model_name
        self.confidence_calculator = ResearchPaperConfidenceCalculator()

        # 初始化豆包客户端
        api_key = os.getenv('DOUBAO_API_KEY')
        if not api_key:
            raise ValueError("DOUBAO_API_KEY环境变量未设置")

        self.client = Ark(
            api_key=api_key,
            timeout=1800,
        )

        # 定义五种专业角色及其系统提示
        self.roles = {
            "product_owner": {
                "name": "Product Owner",
                "system_prompt": """你是一位产品负责人(Product Owner)，专注于最大化产品价值。你从商业目标、市场需求、投资回报率(ROI)和产品战略角度分析需求。你关注需求如何支持业务目标、满足用户需求、创造商业价值，并确定需求的优先级。你的分析侧重于"为什么"要开发这个功能以及它对产品愿景的贡献。""",
                "focus_areas": ["业务价值", "市场需求", "ROI", "产品战略", "优先级"]
            },
            "business_analyst": {
                "name": "Business Analyst",
                "system_prompt": """你是一位业务分析师(Business Analyst)，专注于需求分析、流程优化和解决方案设计。你从业务流程、功能规格、需求完整性和一致性角度分析需求。你关注需求的清晰性、可测试性、与现有系统的一致性，以及如何最好地满足业务目标。你的分析侧重于"什么"需要被构建以及如何满足业务需求。""",
                "focus_areas": ["流程分析", "功能规格", "需求一致性", "解决方案设计"]
            },
            "system_architect": {
                "name": "System Architect",
                "system_prompt": """你是一位系统架构师(System Architect)，专注于技术可行性、系统设计和架构约束。你从技术实现、可扩展性、性能、安全性、集成难度和维护成本角度分析需求。你关注需求的技术影响、架构决策、技术债务和长期可持续性。你的分析侧重于"如何"在技术上实现这个需求。""",
                "focus_areas": ["技术架构", "可扩展性", "性能", "安全性", "集成"]
            },
            "user_experience_designer": {
                "name": "User Experience Designer",
                "system_prompt": """你是一位用户体验设计师(User Experience Designer)，专注于用户交互、界面设计和用户体验优化。你从用户旅程、交互设计、可用性、可访问性和情感化设计角度分析需求。你关注需求如何影响用户满意度、易用性、学习曲线和整体用户体验。你的分析侧重于需求如何被"用户感知和使用"。""",
                "focus_areas": ["用户体验", "交互设计", "可用性", "可访问性", "用户满意度"]
            },
            "software_tester": {
                "name": "Software Tester",
                "system_prompt": """你是一位软件测试工程师(Software Tester)，专注于质量保证、测试设计和缺陷预防。你从可测试性、测试用例设计、边界条件、错误处理和质量标准角度分析需求。你关注需求如何被验证、测试覆盖度、潜在风险点和质量保证措施。你的分析侧重于如何"验证"需求是否被正确实现。""",
                "focus_areas": ["可测试性", "测试设计", "边界条件", "质量保证", "验证标准"]
            }
        }

        print(f"✅ 成功创建多角色豆包模型客户端: {self.name}")
        print(f"  使用的专业角色: {', '.join([role['name'] for role in self.roles.values()])}")
        print(f"  置信度计算方法: C_final = min(1.0, C_base + B_gap), 其中B_gap = min(0.2, (V_top - V_second) × 0.1)")

    def predict_with_multi_roles(self, text: str, labels: List[str],
                                 category_explanations: Dict[str, str] = None,
                                 requirements_examples: Dict[str, List[str]] = None) -> ModelPrediction:
        """使用五种专业角色进行投票预测 - 完全按照研究论文方法"""

        role_votes = {}  # 每个角色的投票

        print(f"\n🔍 开始多角色投票分析: '{text[:50]}...'")
        print("-" * 60)

        # 按顺序让每个角色进行投票
        for role_key, role_info in self.roles.items():
            role_name = role_info["name"]
            print(f"  👤 {role_name} 正在分析...")

            for attempt in range(3):  # 重试机制
                try:
                    prompt = self._build_role_specific_prompt(
                        text, labels, role_info,
                        category_explanations, requirements_examples
                    )

                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": role_info["system_prompt"]},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=200
                    )

                    response_content = completion.choices[0].message.content.strip()
                    predicted_label = self._extract_label_from_response(response_content, labels)

                    if predicted_label:
                        role_votes[role_name] = predicted_label
                        print(f"    ✅ {role_name}: {predicted_label}")
                        break
                    else:
                        print(f"    ⚠️ {role_name} 第{attempt + 1}次尝试失败 - 响应格式异常")
                        if attempt == 2:
                            role_votes[role_name] = labels[0] if labels else "Unknown"
                except Exception as e:
                    print(f"    ❌ {role_name} 第{attempt + 1}次尝试异常: {str(e)[:100]}")
                    if attempt == 2:
                        role_votes[role_name] = labels[0] if labels else "Unknown"
                    time.sleep(2)

            time.sleep(1)  # 角色间延迟

        # 收集所有投票
        votes = list(role_votes.values())

        # 统计投票结果
        vote_counts = {}
        for vote in votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1

        # 确定最终标签（多数投票）
        if vote_counts:
            final_label = max(vote_counts.items(), key=lambda x: x[1])[0]
        else:
            final_label = labels[0] if labels else "Unknown"

        # 完全按照研究论文的方法计算共识置信度
        confidence_details = self.confidence_calculator.calculate_confidence(vote_counts)
        final_confidence = confidence_details['final_confidence']

        # 打印计算过程
        print(f"\n📊 投票统计: {dict(vote_counts)}")
        print(f"📈 置信度计算过程（按照论文公式）:")
        print(
            f"  V_top = {confidence_details['v_top']}, V_second = {confidence_details['v_second']}, N = {confidence_details['total_votes']}")
        print(
            f"  C_base = V_top / N = {confidence_details['v_top']} / {confidence_details['total_votes']} = {confidence_details['c_base']:.2f}")
        print(
            f"  B_gap = min(0.2, (V_top - V_second) × 0.1) = min(0.2, ({confidence_details['v_top']} - {confidence_details['v_second']}) × 0.1) = {confidence_details['b_gap']:.2f}")
        print(
            f"  C_final = min(1.0, C_base + B_gap) = min(1.0, {confidence_details['c_base']:.2f} + {confidence_details['b_gap']:.2f}) = {final_confidence:.3f}")
        print(f"✅ 最终分类: {final_label}, 置信度: {final_confidence:.3f}")

        return ModelPrediction(
            label=final_label,
            confidence=final_confidence,
            model_name=f"{self.name}-multi-role",
            votes=votes,
            vote_counts=vote_counts,
            role_votes=role_votes,
            calculation_details=confidence_details
        )

    def _build_role_specific_prompt(self, text: str, labels: List[str],
                                    role_info: Dict,
                                    category_explanations: Dict[str, str] = None,
                                    requirements_examples: Dict[str, List[str]] = None) -> str:
        """为特定角色构建提示"""

        role_name = role_info["name"]

        labels_section = f"""可选分类标签（请严格从以下标签中选择一个）:

{chr(10).join([f'• {label}' for label in labels])}

重要提示：请确保完全按照上述标签名称返回，不要修改、缩写或添加任何额外内容。"""

        explanations_section = ""
        if category_explanations:
            explanations_section = "\n\n类别详细解释:\n"
            for label in labels:
                if label in category_explanations:
                    explanations_section += f"\n【{label}】\n{category_explanations[label]}\n"

        examples_section = ""
        if requirements_examples:
            examples_section = "\n\n参考示例:\n"
            for label in labels:
                if label in requirements_examples and requirements_examples[label]:
                    examples = requirements_examples[label]
                    examples_section += f"\n【{label}】的典型示例："
                    for i, example in enumerate(examples[:2], 1):
                        examples_section += f"\n  {i}. {example}"
                    examples_section += "\n"

        role_guidance = f"""
作为{role_name}，请从你的专业角度分析这个需求：

你的专业关注领域：{', '.join(role_info['focus_areas'])}
请从{role_name}的角度思考并选择最合适的分类标签。"""

        target_section = f"""
待分类的需求描述:
"{text}"

请按照以下格式回复：
【{role_name}分析】[你的专业分析]
【分类标签】[必须从上述标签中选择一个，完全匹配标签名称]"""

        return f"""你现在的角色是：{role_name}

{role_guidance}

{labels_section}{explanations_section}{examples_section}{target_section}"""

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


class MultiRoleProcessor:
    """多角色模型处理器"""

    def __init__(self, multi_role_client: MultiRoleDoubaoClient):
        self.multi_role_client = multi_role_client
        self.start_time = None
        self.end_time = None
        print(f"多角色模型处理器初始化完成，使用{len(multi_role_client.roles)}种专业角色")

    def process_dataset(self, dataset: List[str], labels: List[str],
                        category_explanations: Dict[str, str] = None,
                        requirements_examples: Dict[str, List[str]] = None) -> List[DataPoint]:
        """使用多角色方法处理整个数据集"""
        data_points = []

        self.start_time = datetime.datetime.now()
        print(f"🚀 多角色分析开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"开始处理 {len(dataset)} 条软件需求，使用5种专业角色...")
        print(f"分类标签数量: {len(labels)} 个")

        print(f"🎭 专业角色配置:")
        for role_key, role_info in self.multi_role_client.roles.items():
            print(f"  • {role_info['name']}")

        for i, text in enumerate(dataset):
            if i % 2 == 0 and i > 0:
                elapsed_time = datetime.datetime.now() - self.start_time
                processed = i
                remaining = len(dataset) - i
                avg_time_per_req = elapsed_time / processed if processed > 0 else elapsed_time
                est_remaining = avg_time_per_req * remaining

                print(f"  📈 进度: {i}/{len(dataset)} - 已用时: {elapsed_time} - 预计剩余: {est_remaining}")

            data_point = DataPoint(content=text)
            prediction = self.multi_role_client.predict_with_multi_roles(
                text, labels, category_explanations, requirements_examples
            )

            data_point.prediction = prediction
            data_point.final_label = prediction.label
            data_points.append(data_point)

            if i < len(dataset) - 1:
                time.sleep(2)

        self.end_time = datetime.datetime.now()
        return data_points

    def save_results(self, results: List[DataPoint], output_file: str):
        """保存最终结果到Excel文件"""
        results_data = []

        for i, data_point in enumerate(results):
            # 构建角色投票明细
            role_votes_detail = ""
            if data_point.prediction and data_point.prediction.role_votes:
                role_votes_detail = "; ".join(
                    [f"{role}:{vote}" for role, vote in data_point.prediction.role_votes.items()])

            # 投票分布
            vote_distribution = ""
            if data_point.prediction and data_point.prediction.vote_counts:
                vote_distribution = ", ".join([f"{k}:{v}" for k, v in data_point.prediction.vote_counts.items()])

            # 置信度计算详情
            c_base = b_gap = 0.0
            v_top = v_second = 0
            if data_point.prediction and data_point.prediction.calculation_details:
                details = data_point.prediction.calculation_details
                c_base = details.get('c_base', 0.0)
                b_gap = details.get('b_gap', 0.0)
                v_top = details.get('v_top', 0)
                v_second = details.get('v_second', 0)

            row_data = {
                '序号': i + 1,
                '需求内容': data_point.content,
                '模型名称': data_point.prediction.model_name if data_point.prediction else "N/A",
                'Product Owner投票': data_point.prediction.role_votes.get('Product Owner',
                                                                          'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                'Business Analyst投票': data_point.prediction.role_votes.get('Business Analyst',
                                                                             'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                'System Architect投票': data_point.prediction.role_votes.get('System Architect',
                                                                             'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                'User Experience Designer投票': data_point.prediction.role_votes.get('User Experience Designer',
                                                                                     'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                'Software Tester投票': data_point.prediction.role_votes.get('Software Tester',
                                                                            'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                '角色投票明细': role_votes_detail,
                '投票分布': vote_distribution,
                '最高票数(V_top)': v_top,
                '次高票数(V_second)': v_second,
                '总票数(N)': 5,
                '基础置信度(C_base)': round(c_base, 3),
                '差距奖励(B_gap)': round(b_gap, 3),
                '最终置信度(C_final)': data_point.prediction.confidence if data_point.prediction else 0.0,
                '最终分类标签': data_point.final_label,
                '置信度计算公式': f"min(1.0, {c_base:.3f} + {b_gap:.3f}) = {data_point.prediction.confidence:.3f}" if data_point.prediction else "N/A"
            }
            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ 多角色分析结果已保存到: {output_file}")

        # 保存详细报告
        self._save_detailed_report(results, output_file)

    def _save_detailed_report(self, results: List[DataPoint], output_file: str):
        """保存详细分析报告"""
        report_file = output_file.replace('.xlsx', '_详细报告.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("豆包模型-五人专业角色分类系统分析报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总需求数量: {len(results)}\n")
            f.write("使用的五种专业角色:\n")
            f.write("1. Product Owner (产品负责人)\n")
            f.write("2. Business Analyst (业务分析师)\n")
            f.write("3. System Architect (系统架构师)\n")
            f.write("4. User Experience Designer (用户体验设计师)\n")
            f.write("5. Software Tester (软件测试工程师)\n\n")

            f.write("置信度计算公式（基于研究论文）:\n")
            f.write("C_base = V_top / N  (其中N=5)\n")
            f.write("B_gap = min(0.2, (V_top - V_second) × 0.1)\n")
            f.write("C_final = min(1.0, C_base + B_gap)\n\n")

            # 置信度统计
            confidences = [dp.prediction.confidence for dp in results if dp.prediction]
            if confidences:
                f.write("📊 置信度统计:\n")
                f.write("-" * 80 + "\n")
                f.write(f"平均置信度: {np.mean(confidences):.3f}\n")
                f.write(f"置信度标准差: {np.std(confidences):.3f}\n")
                f.write(f"最低置信度: {np.min(confidences):.3f}\n")
                f.write(f"最高置信度: {np.max(confidences):.3f}\n\n")

                # 置信度分布
                confidence_bins = {
                    "0.9-1.0": 0, "0.8-0.9": 0, "0.7-0.8": 0,
                    "0.6-0.7": 0, "0.5-0.6": 0, "0.4-0.5": 0,
                    "0.3-0.4": 0, "0.2-0.3": 0, "0.0-0.2": 0
                }

                for conf in confidences:
                    if conf >= 0.9:
                        confidence_bins["0.9-1.0"] += 1
                    elif conf >= 0.8:
                        confidence_bins["0.8-0.9"] += 1
                    elif conf >= 0.7:
                        confidence_bins["0.7-0.8"] += 1
                    elif conf >= 0.6:
                        confidence_bins["0.6-0.7"] += 1
                    elif conf >= 0.5:
                        confidence_bins["0.5-0.6"] += 1
                    elif conf >= 0.4:
                        confidence_bins["0.4-0.5"] += 1
                    elif conf >= 0.3:
                        confidence_bins["0.3-0.4"] += 1
                    elif conf >= 0.2:
                        confidence_bins["0.2-0.3"] += 1
                    else:
                        confidence_bins["0.0-0.2"] += 1

                f.write("置信度分布:\n")
                for bin_range, count in confidence_bins.items():
                    if count > 0:
                        percentage = count / len(confidences) * 100
                        f.write(f"  {bin_range}: {count}条 ({percentage:.1f}%)\n")

                # 投票一致性统计
                unanimous = sum(1 for dp in results if dp.prediction and dp.prediction.vote_counts and max(
                    dp.prediction.vote_counts.values()) == 5)
                majority_4 = sum(1 for dp in results if dp.prediction and dp.prediction.vote_counts and max(
                    dp.prediction.vote_counts.values()) == 4)
                majority_3 = sum(1 for dp in results if dp.prediction and dp.prediction.vote_counts and max(
                    dp.prediction.vote_counts.values()) == 3)

                f.write(f"\n投票一致性统计:\n")
                f.write(f"  全票一致(5:0): {unanimous}条 ({(unanimous / len(results) * 100):.1f}%)\n")
                f.write(f"  绝对多数(4:1): {majority_4}条 ({(majority_4 / len(results) * 100):.1f}%)\n")
                f.write(f"  相对多数(3:2): {majority_3}条 ({(majority_3 / len(results) * 100):.1f}%)\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✅ 详细分析报告已保存到: {report_file}")

    def print_statistics(self, results: List[DataPoint]):
        """打印统计信息"""
        print(f"\n{'=' * 60}")
        print("📊 豆包模型五人专业角色分类统计信息")
        print(f"{'=' * 60}")

        print(f"处理总数据量: {len(results)}条需求")
        print(f"使用角色数量: {len(self.multi_role_client.roles)}个专业角色")

        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            print(f"\n⏰ 时间统计:")
            print(f"  开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  结束时间: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  总运行时间: {total_duration}")

            if len(results) > 0:
                avg_time_per_req = total_duration / len(results)
                print(f"  平均每条需求处理时间: {avg_time_per_req}")

        # 最终标签分布
        final_labels = [dp.final_label for dp in results if dp.final_label]
        if final_labels:
            label_counts = {}
            for label in final_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            print(f"\n🏷️ 最终分类分布:")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(results) * 100
                print(f"  {label}: {count}条 ({percentage:.1f}%)")

        # 置信度统计
        confidences = [dp.prediction.confidence for dp in results if dp.prediction]
        if confidences:
            print(f"\n🎯 置信度统计（按照论文公式计算）:")
            print(f"  平均置信度: {np.mean(confidences):.3f}")
            print(f"  置信度标准差: {np.std(confidences):.3f}")

            # 计算投票一致性统计
            unanimous_count = sum(1 for dp in results if dp.prediction and dp.prediction.vote_counts and max(
                dp.prediction.vote_counts.values()) == 5)
            high_confidence_count = sum(1 for dp in results if dp.prediction and dp.prediction.confidence >= 0.8)
            medium_confidence_count = sum(
                1 for dp in results if dp.prediction and 0.6 <= dp.prediction.confidence < 0.8)
            low_confidence_count = sum(1 for dp in results if dp.prediction and dp.prediction.confidence < 0.6)

            print(f"\n📈 置信度分布:")
            print(f"  高置信度(≥0.8): {high_confidence_count}条 ({(high_confidence_count / len(results) * 100):.1f}%)")
            print(
                f"  中置信度(0.6-0.8): {medium_confidence_count}条 ({(medium_confidence_count / len(results) * 100):.1f}%)")
            print(f"  低置信度(<0.6): {low_confidence_count}条 ({(low_confidence_count / len(results) * 100):.1f}%)")
            print(f"  全票一致(5:0): {unanimous_count}条 ({(unanimous_count / len(results) * 100):.1f}%)")


class DataLoader:
    """数据加载器类"""

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

    @staticmethod
    def load_test_requirements(file_path: str) -> List[str]:
        """从dataset文件加载需求"""
        try:
            df = pd.read_excel(file_path)
            requirements = []

            if 'requirement' in df.columns:
                requirements = [
                    str(row['requirement']).strip()
                    for _, row in df.iterrows()
                    if pd.notna(row['requirement']) and str(row['requirement']).strip()
                ]
            else:
                first_col = df.columns[0]
                requirements = [
                    str(row[first_col]).strip()
                    for _, row in df.iterrows()
                    if pd.notna(row[first_col]) and str(row[first_col]).strip()
                ]

            print(f"✅ 成功从 {os.path.basename(file_path)} 加载 {len(requirements)} 条测试需求")
            return requirements
        except Exception as e:
            print(f"❌ 加载测试需求文件 {os.path.basename(file_path)} 出错: {e}")
            return []


def main():
    """主函数 - 使用多角色方法"""
    # 文件路径配置
    dataset_file = "dataset.xlsx"
    concept_file = "1123Concept.xlsx"
    examples_file = "1122RequirementExamples.xlsx"
    output_file = "doubao_5人2置信度.xlsx"

    data_loader = DataLoader()

    print("=" * 60)
    print("🎭 豆包模型-五人专业角色软件需求分类系统")
    print("基于研究论文方法: 五种专业角色投票 + 双置信度计算")
    print("置信度公式: C_final = min(1.0, C_base + B_gap)")
    print("其中: C_base = V_top/N, B_gap = min(0.2, (V_top - V_second)×0.1)")
    print("=" * 60)

    print("\n📁 正在加载数据...")
    print(f"  数据集文件: {dataset_file}")
    print(f"  概念文件: {concept_file}")
    print(f"  示例文件: {examples_file}")
    print(f"  输出文件: {output_file}")

    # 创建多角色豆包模型客户端
    try:
        multi_role_client = MultiRoleDoubaoClient(
            name="doubao-seed-1-6-250615-5人角色",
            model_name="doubao-seed-1-6-250615"
        )
    except Exception as e:
        print(f"❌ 创建多角色模型客户端失败: {e}")
        return

    # 加载类别、解释和例子
    category_explanations = data_loader.load_categories_and_explanations(concept_file)
    requirements_examples = data_loader.load_requirements_examples(examples_file)
    labels = list(category_explanations.keys())
    test_requirements = data_loader.load_test_requirements(dataset_file)

    if not test_requirements:
        print("❌ 没有找到测试需求，程序退出")
        return

    if not labels:
        print("❌ 没有找到分类标签，程序退出")
        return

    print(f"\n✅ 数据加载完成:")
    print(f"  分类标签: {len(labels)} 个")
    print(f"  测试需求: {len(test_requirements)} 条")

    slash_labels = [label for label in labels if '/' in label]
    if slash_labels:
        print(f"  🔍 包含'/'的标签: {', '.join(slash_labels)}")

    # 执行多角色模型处理
    print(f"\n🚀 开始豆包模型五人角色分类处理...")
    processor = MultiRoleProcessor(multi_role_client)
    results = processor.process_dataset(test_requirements, labels, category_explanations, requirements_examples)

    # 保存和显示结果
    processor.save_results(results, output_file)
    processor.print_statistics(results)

    # 显示前几个结果的详细信息
    print(f"\n🔍 前3个结果的详细信息:")
    for i, data_point in enumerate(results[:3]):
        print(f"\n{'=' * 60}")
        print(f"{i + 1}. 需求: {data_point.content[:100]}...")
        if data_point.prediction:
            print(f"   最终分类: {data_point.final_label}")
            print(f"   最终置信度: {data_point.prediction.confidence:.3f}")

            if data_point.prediction.role_votes:
                print(f"\n   各角色投票:")
                for role, vote in data_point.prediction.role_votes.items():
                    print(f"     {role}: {vote}")

            if data_point.prediction.calculation_details:
                details = data_point.prediction.calculation_details
                print(f"   置信度计算过程:")
                print(f"     V_top = {details['v_top']}, V_second = {details['v_second']}")
                print(f"     C_base = {details['v_top']}/5 = {details['c_base']:.2f}")
                print(f"     B_gap = min(0.2, ({details['v_top']}-{details['v_second']})×0.1) = {details['b_gap']:.2f}")
                print(
                    f"     C_final = min(1.0, {details['c_base']:.2f} + {details['b_gap']:.2f}) = {details['final_confidence']:.3f}")

    print(f"\n{'=' * 60}")
    print("🎉 豆包模型五人角色分类处理完成!")
    print(f"📊 结果文件: {output_file}")
    print(f"📄 详细报告: {output_file.replace('.xlsx', '_详细报告.txt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()