import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
import time
import re
import random

# 加载环境变量
load_dotenv()


@dataclass
class ModelPrediction:
    """存储模型预测结果"""
    label: str
    confidence: float  # 大模型自评置信度
    model_name: str


@dataclass
class DataPoint:
    """数据点类"""
    content: str
    true_label: str  # 人工标注标签
    prediction: ModelPrediction = None


class OpenAIProxyClient:
    """OpenAI代理客户端类 - 使用OpenAI兼容接口"""

    def __init__(self, name: str = "gpt-4.1-nano", model_name: str = "gpt-4.1-nano"):
        self.name = name
        self.model_name = model_name

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY环境变量未设置")

        self.client = OpenAI(
            base_url='https://api.openai-proxy.org/v1',
            api_key=api_key,
        )

    def predict(self, text: str, labels: List[str],
                category_explanations: Dict[str, str] = None,
                requirements_examples: Dict[str, List[str]] = None) -> ModelPrediction:
        """对文本进行分类预测 - 语言学者角色，单次判断，使用大模型自评置信度"""

        print(f"  语言学者角色分析需求: {text[:50]}...")

        for attempt in range(3):  # 重试机制
            try:
                # 随机打乱类别顺序（为每个需求独立打乱）
                shuffled_labels, shuffled_explanations, shuffled_examples = self._shuffle_categories(
                    labels, category_explanations, requirements_examples
                )

                prompt = self._build_classification_prompt(text, shuffled_labels,
                                                           shuffled_explanations, shuffled_examples)

                # 使用OpenAI兼容接口调用API
                completion = self.client.chat.completions.create(
                    model=self.model_name,  # 使用 gpt-4.1-nano
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一位资深的语言学者，专门研究软件需求的语言结构和表达方式。请从语言学角度分析软件需求语句的结构、用词、语法和语义特征，并给出最准确的分类和你的置信度评分。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,  # 较低温度以获得更确定性的回答
                    max_tokens=100  # 增加token数以获取置信度信息
                )

                response_content = completion.choices[0].message.content.strip()

                # 提取标签和置信度（注意：需要将打乱的标签映射回原始标签）
                label, confidence = self._extract_label_and_confidence(response_content, shuffled_labels, labels)

                if label:
                    print(f"  分类结果: {label}")
                    print(f"  大模型自评置信度: {confidence:.3f}")

                    return ModelPrediction(
                        label=label,
                        confidence=confidence,
                        model_name=self.name
                    )
                else:
                    print(f"  第{attempt + 1}次尝试失败 - 响应: {response_content}")
                    time.sleep(1)

            except Exception as e:
                print(f"  调用异常: {str(e)}")
                time.sleep(2)

        print("  所有尝试都失败了，使用默认分类")
        return ModelPrediction(
            label=labels[0] if labels else "Unknown",
            confidence=0.1,  # 失败时的默认置信度
            model_name=self.name
        )

    def _shuffle_categories(self, labels: List[str],
                            category_explanations: Dict[str, str] = None,
                            requirements_examples: Dict[str, List[str]] = None) -> Tuple:
        """随机打乱类别顺序，确保公平性"""

        # 创建标签的副本
        labels_copy = labels.copy()

        # 随机打乱标签顺序
        random.shuffle(labels_copy)

        # 根据打乱的标签顺序重新组织解释和示例
        shuffled_explanations = None
        if category_explanations:
            shuffled_explanations = {}
            for label in labels_copy:
                if label in category_explanations:
                    shuffled_explanations[label] = category_explanations[label]

        shuffled_examples = None
        if requirements_examples:
            shuffled_examples = {}
            for label in labels_copy:
                if label in requirements_examples:
                    shuffled_examples[label] = requirements_examples[label]

        print(f"    类别顺序已随机打乱，新顺序: {labels_copy}")
        return labels_copy, shuffled_explanations, shuffled_examples

    def _extract_label_and_confidence(self, response: str, shuffled_labels: List[str],
                                      original_labels: List[str]) -> Tuple[str, float]:
        """从响应中提取标签和大模型自评置信度"""
        response_clean = response.strip()

        # 默认值
        confidence = 0.5
        extracted_label = None

        # 1. 先尝试提取置信度
        confidence_patterns = [
            r'置信度[：:]\s*(\d+\.?\d*)',
            r'confidence[：:]\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*分',
            r'(\d+\.?\d*)/1',
            r'(\d+\.?\d*)/10',
            r'(\d+\.?\d*)%'
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, response_clean.lower())
            if match:
                try:
                    conf_value = float(match.group(1))
                    # 根据格式调整置信度范围
                    if pattern.endswith('%'):
                        confidence = conf_value / 100.0
                    elif pattern.endswith('/10'):
                        confidence = conf_value / 10.0
                    else:
                        # 假设是0-1范围，但如果值大于1，调整范围
                        if conf_value > 1 and conf_value <= 10:
                            confidence = conf_value / 10.0
                        elif conf_value > 10 and conf_value <= 100:
                            confidence = conf_value / 100.0
                        else:
                            confidence = min(1.0, max(0.0, conf_value))
                    break
                except ValueError:
                    continue

        # 2. 提取标签（从打乱的标签中查找）
        response_for_label = response_clean.strip().strip('.,!?;:"')

        # 精确匹配（在打乱的标签中查找）
        if response_for_label in shuffled_labels:
            extracted_label = response_for_label

        # 部分匹配（在打乱的标签中查找）
        if not extracted_label:
            for label in shuffled_labels:
                if label.lower() in response_for_label.lower():
                    extracted_label = label
                    break

        # 检查是否包含标签的关键词（使用原始标签映射）
        if not extracted_label:
            label_keywords = {
                'Data constraint': ['data', 'constraint'],
                'Action constraint': ['action', 'constraint'],
                'Object constraint': ['object', 'constraint'],
                'Calculation': ['calculate', 'calculation'],
                'Trigger': ['trigger', 'when'],
                'System reaction': ['system', 'reaction'],
                'Conditional': ['if', 'condition'],
                'Timing': ['time', 'when'],
                'Exception': ['exception', 'error']
            }

            # 先在打乱的标签中查找
            for label in shuffled_labels:
                if label in label_keywords:
                    keywords = label_keywords[label]
                    if all(keyword in response_for_label.lower() for keyword in keywords):
                        extracted_label = label
                        break

        # 默认返回第一个原始标签（不是打乱的第一个）
        if not extracted_label:
            extracted_label = original_labels[0] if original_labels else "Unknown"

        # 确保返回的标签在原始标签列表中（如果不在，映射到最相似的）
        if extracted_label not in original_labels:
            # 尝试找到最相似的标签
            for original_label in original_labels:
                if original_label.lower() == extracted_label.lower():
                    extracted_label = original_label
                    break
                elif original_label.lower() in extracted_label.lower():
                    extracted_label = original_label
                    break

        return extracted_label, confidence

    def _build_classification_prompt(self, text: str, labels: List[str],
                                     category_explanations: Dict[str, str] = None,
                                     requirements_examples: Dict[str, List[str]] = None) -> str:
        """构建分类提示 - 语言学者角色，要求提供置信度"""

        # 构建清晰的分类标签部分（使用打乱的顺序）
        labels_section = f"""🏷️ 可选分类标签（请从以下标签中选择一个）:
{', '.join(labels)}"""

        # 构建分类解释部分（使用打乱的顺序）
        explanations_section = ""
        if category_explanations:
            explanations_section = "\n\n📚 类别语言学特征解释:\n"
            for label in labels:
                if label in category_explanations:
                    explanations_section += f"""
【{label}】
{category_explanations[label]}\n"""

        # 构建需求例子部分（使用打乱的顺序）
        examples_section = ""
        if requirements_examples:
            examples_section = "\n\n📋 各类别语言学参考示例:\n"
            for label in labels:
                if label in requirements_examples and requirements_examples[label]:
                    examples = requirements_examples[label]
                    examples_section += f"""
【{label}】的典型表达："""
                    for i, example in enumerate(examples[:3], 1):  # 只显示前3个示例
                        examples_section += f"""
  {i}. {example}"""
                    examples_section += "\n"

        # 待分类的需求部分
        target_section = f"""

🔍 待分析的软件需求:
"{text}"

💡 语言学分析要求:
1. 从语言学角度分析这个需求的语法结构、用词特点和表达方式
2. 参考上述类别解释和示例，进行语言学对比
3. 选择最合适的分类标签
4. 对你的分类判断给出一个置信度评分（0-1之间，1表示完全确定）

📝 请严格按照以下格式回复：
标签: [你的分类标签]
置信度: [你的置信度评分，0-1之间的小数]

例如：
标签: Conditional
置信度: 0.85"""

        return f"""你是一位资深的语言学者，专门分析软件需求的语言特征。

{labels_section}{explanations_section}{examples_section}{target_section}"""


class DataLoader:
    """数据加载器类"""

    @staticmethod
    def get_file_path(filename: str) -> str:
        """获取文件路径，支持相对路径和绝对路径"""
        # 先检查当前目录
        if os.path.exists(filename):
            return filename

        # 检查上一级目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        parent_path = os.path.join(parent_dir, filename)
        if os.path.exists(parent_path):
            return parent_path

        return filename  # 返回原路径，让错误在读取时抛出

    @staticmethod
    def load_dataset(file_path: str) -> Tuple[List[str], List[str]]:
        """从dataset.xlsx加载需要标注的语句和人工标注"""
        try:
            file_path = DataLoader.get_file_path(file_path)
            df = pd.read_excel(file_path)

            # 查找requirement和label列
            requirement_col = None
            label_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'requirement' in col_lower or '需求' in col_lower:
                    requirement_col = col
                elif 'label' in col_lower or '标签' in col_lower:
                    label_col = col

            if not requirement_col:
                # 如果没有找到，使用第一列
                requirement_col = df.columns[0]

            if not label_col and len(df.columns) > 1:
                # 尝试使用第二列作为label
                label_col = df.columns[1]

            requirements = []
            true_labels = []

            for _, row in df.iterrows():
                if pd.notna(row[requirement_col]):
                    requirements.append(str(row[requirement_col]).strip())

                    if label_col and pd.notna(row.get(label_col, None)):
                        true_labels.append(str(row[label_col]).strip())
                    else:
                        true_labels.append("Unknown")

            print(f"成功加载 {len(requirements)} 条需求，{len(true_labels)} 个标签")
            return requirements, true_labels

        except Exception as e:
            print(f"加载数据集文件出错: {e}")
            return [], []

    @staticmethod
    def load_categories_and_explanations(file_path: str) -> Tuple[Dict[str, str], List[str]]:
        """从1123Concept文件加载类别和解释，返回类别解释和排序后的标签列表"""
        try:
            file_path = DataLoader.get_file_path(file_path)
            df = pd.read_excel(file_path)

            # 查找category和explanation列
            category_col = None
            explanation_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'category' in col_lower or '类别' in col_lower:
                    category_col = col
                elif 'explanation' in col_lower or '解释' in col_lower or '说明' in col_lower:
                    explanation_col = col

            if not category_col:
                category_col = df.columns[0]
            if not explanation_col and len(df.columns) > 1:
                explanation_col = df.columns[1]

            category_explanations = {}
            original_labels = []

            for _, row in df.iterrows():
                if pd.notna(row[category_col]):
                    category = str(row[category_col]).strip()
                    explanation = ""
                    if explanation_col and pd.notna(row.get(explanation_col, None)):
                        explanation = str(row[explanation_col]).strip()
                    category_explanations[category] = explanation
                    original_labels.append(category)

            print(f"成功加载 {len(category_explanations)} 个类别解释")
            return category_explanations, original_labels

        except Exception as e:
            print(f"加载类别解释文件出错: {e}")
            return {}, []

    @staticmethod
    def load_requirements_examples(file_path: str) -> Dict[str, List[str]]:
        """从1122RequirementExamples文件加载需求示例"""
        try:
            file_path = DataLoader.get_file_path(file_path)
            df = pd.read_excel(file_path)

            # 查找boileplate type列
            type_col = None
            for col in df.columns:
                col_lower = col.lower()
                if 'boileplate' in col_lower or 'type' in col_lower or '类别' in col_lower:
                    type_col = col
                    break

            if not type_col:
                type_col = df.columns[0]

            # 查找example列
            example_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if 'example' in col_lower and col != type_col:
                    example_cols.append(col)

            # 按example 1, example 2...排序
            example_cols.sort()

            requirements_examples = {}

            for _, row in df.iterrows():
                if pd.notna(row[type_col]):
                    boilerplate_type = str(row[type_col]).strip()
                    examples = []

                    for col in example_cols:
                        if col in row and pd.notna(row[col]):
                            example = str(row[col]).strip()
                            if example and example.lower() != 'nan':
                                examples.append(example)

                    if boilerplate_type and examples:
                        requirements_examples[boilerplate_type] = examples

            print(f"成功加载 {len(requirements_examples)} 个类别的需求示例")
            return requirements_examples

        except Exception as e:
            print(f"加载需求示例文件出错: {e}")
            return {}


class OpenAIProxyProcessor:
    """OpenAI代理模型处理器 - 语言学者角色"""

    def __init__(self, openai_client: OpenAIProxyClient):
        """初始化OpenAI代理模型处理器"""
        self.openai_client = openai_client
        print(f"OpenAI代理语言学者处理器初始化完成，使用模型: {openai_client.name}")
        print("模式: 语言学者角色，单次判断，使用大模型自评置信度")
        print("注意: 每个需求的类别顺序都会随机打乱，确保公平性")

    def process_dataset(self, dataset: List[str], true_labels: List[str], labels: List[str],
                        category_explanations: Dict[str, str] = None,
                        requirements_examples: Dict[str, List[str]] = None) -> List[DataPoint]:
        """处理整个数据集 - 单次判断模式"""
        data_points = []

        print(f"开始处理 {len(dataset)} 条软件需求，使用语言学者角色单次判断...")
        print(f"使用分类解释: {len(category_explanations) if category_explanations else 0} 个")
        if requirements_examples:
            total_examples = sum(len(examples) for examples in requirements_examples.values())
            print(f"使用需求示例: {total_examples} 条")

        for i, (text, true_label) in enumerate(zip(dataset, true_labels)):
            if i % 10 == 0:  # 每10条显示一次进度
                print(f"  进度: {i}/{len(dataset)} ({i / len(dataset) * 100:.1f}%)")

            # 创建数据点
            data_point = DataPoint(content=text, true_label=true_label)

            # 使用OpenAI代理模型进行预测（单次判断）
            prediction = self.openai_client.predict(text, labels, category_explanations, requirements_examples)
            data_point.prediction = prediction

            data_points.append(data_point)

            # 添加适当的延迟以避免速率限制
            if i < len(dataset) - 1:
                time.sleep(0.5)  # 每0.5秒处理一个请求

        return data_points

    def save_results(self, results: List[DataPoint], output_file: str):
        """保存最终结果到Excel文件"""
        results_data = []

        # 统计准确率
        correct_count = 0
        total_comparable = 0

        for i, data_point in enumerate(results):
            # 检查是否有可比较的真实标签
            is_comparable = data_point.true_label != "Unknown" and data_point.prediction

            if is_comparable:
                total_comparable += 1
                if data_point.prediction.label == data_point.true_label:
                    correct_count += 1

            row_data = {
                '序号': i + 1,
                '需求内容': data_point.content,
                '人工标注标签': data_point.true_label,
                '模型名称': data_point.prediction.model_name if data_point.prediction else "N/A",
                '预测标签': data_point.prediction.label if data_point.prediction else "N/A",
                '大模型自评置信度': data_point.prediction.confidence if data_point.prediction else 0.0,
                '是否正确': ("是" if data_point.prediction.label == data_point.true_label
                             else "否" if is_comparable else "未知")
            }
            results_data.append(row_data)

        df = pd.DataFrame(results_data)

        # 添加统计信息
        if total_comparable > 0:
            accuracy = correct_count / total_comparable * 100
            stats_df = pd.DataFrame([{
                '序号': '统计',
                '需求内容': f"总数量: {len(results)}, 可比较数量: {total_comparable}, 准确率: {accuracy:.2f}%",
                '人工标注标签': '',
                '模型名称': '',
                '预测标签': '',
                '大模型自评置信度': '',
                '是否正确': ''
            }])
            df = pd.concat([df, stats_df], ignore_index=True)

        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ 最终结果已保存到: {output_file}")

        if total_comparable > 0:
            print(f"📊 准确率统计: {accuracy:.2f}% ({correct_count}/{total_comparable})")

    def print_statistics(self, results: List[DataPoint]):
        """打印统计信息"""
        print(f"\n📊 OpenAI代理语言学者模型统计信息:")
        print(f"处理总数据量: {len(results)}")
        print(f"使用模型: {self.openai_client.name}")
        print(f"判断模式: 单次判断，使用大模型自评置信度")
        print(f"公平性: 每个需求的类别顺序都已随机打乱")

        # 计算准确率
        correct_count = 0
        total_comparable = 0

        for dp in results:
            if dp.true_label != "Unknown" and dp.prediction:
                total_comparable += 1
                if dp.prediction.label == dp.true_label:
                    correct_count += 1

        if total_comparable > 0:
            accuracy = correct_count / total_comparable * 100
            print(f"\n准确率统计:")
            print(f"  可比较样本数: {total_comparable}")
            print(f"  正确分类数: {correct_count}")
            print(f"  准确率: {accuracy:.2f}%")

        # 预测标签分布
        predicted_labels = [dp.prediction.label for dp in results if dp.prediction]
        if predicted_labels:
            label_counts = {}
            for label in predicted_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            print(f"\n预测分类分布:")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(predicted_labels) * 100
                print(f"  {label}: {count} 条 ({percentage:.1f}%)")

        # 置信度统计（大模型自评）
        confidences = [dp.prediction.confidence for dp in results if dp.prediction]
        if confidences:
            print(f"\n大模型自评置信度统计:")
            print(f"  平均置信度: {np.mean(confidences):.3f}")
            print(f"  置信度标准差: {np.std(confidences):.3f}")
            print(f"  最低置信度: {np.min(confidences):.3f}")
            print(f"  最高置信度: {np.max(confidences):.3f}")

            # 置信度分布
            high_conf = sum(1 for c in confidences if c >= 0.8)
            medium_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
            low_conf = sum(1 for c in confidences if c < 0.5)

            print(f"  高置信度(≥0.8): {high_conf} 条 ({high_conf / len(confidences) * 100:.1f}%)")
            print(f"  中置信度(0.5-0.8): {medium_conf} 条 ({medium_conf / len(confidences) * 100:.1f}%)")
            print(f"  低置信度(<0.5): {low_conf} 条 ({low_conf / len(confidences) * 100:.1f}%)")


def main():
    """主函数"""
    # 文件路径配置
    dataset_file = "dataset.xlsx"
    concept_file = "1123Concept.xlsx"
    examples_file = "1122RequirementExamples.xlsx"
    output_file = "gpt4.1_nano语言学者1次分类结果_随机顺序.xlsx"

    data_loader = DataLoader()

    print("=" * 60)
    print("GPT-4.1-Nano 语言学者软件需求分类系统（随机顺序版）")
    print("=" * 60)
    print("模式: 语言学者角色，单次判断，使用大模型自评置信度")
    print("模型: GPT-4.1-Nano (通过OpenAI代理)")
    print("特点: 每个需求的类别顺序都会随机打乱，消除顺序偏差")
    print("=" * 60)

    # 创建OpenAI代理模型客户端 - 改为使用 gpt-4.1-nano
    try:
        openai_client = OpenAIProxyClient(
            name="gpt-4.1-nano",  # 修改这里
            model_name="gpt-4.1-nano"  # 修改这里
        )
        print(f"✅ 成功创建OpenAI代理模型客户端: {openai_client.name}")
        print(f"   API代理地址: {openai_client.client.base_url}")
    except Exception as e:
        print(f"❌ 创建OpenAI代理模型客户端失败: {e}")
        return

    # 加载数据
    print("\n📂 正在加载数据文件...")

    # 加载数据集
    test_requirements, true_labels = data_loader.load_dataset(dataset_file)

    if not test_requirements:
        print("❌ 没有找到测试需求，程序退出")
        return

    # 加载类别、解释和例子
    category_explanations, original_labels = data_loader.load_categories_and_explanations(concept_file)
    requirements_examples = data_loader.load_requirements_examples(examples_file)

    # 使用原始标签列表
    labels = original_labels

    if not labels:
        print("❌ 没有找到分类标签，程序退出")
        return

    print(f"\n✅ 数据加载完成:")
    print(f"测试需求数量: {len(test_requirements)} 条")
    print(f"人工标注数量: {len([l for l in true_labels if l != 'Unknown'])} 条")
    print(f"可用分类标签: {len(labels)} 个")
    if requirements_examples:
        total_examples = sum(len(examples) for examples in requirements_examples.values())
        print(f"需求示例数量: {total_examples} 条")
    print(f"原始标签顺序: {labels}")

    # 执行OpenAI代理模型处理
    print(f"\n🚀 开始语言学者分析（GPT-4.1-Nano，单次判断模式，随机顺序）...")
    processor = OpenAIProxyProcessor(openai_client)
    results = processor.process_dataset(test_requirements, true_labels, labels,
                                        category_explanations, requirements_examples)

    # 保存和显示结果
    processor.save_results(results, output_file)
    processor.print_statistics(results)

    # 显示示例分析
    print(f"\n🔍 前5个结果的详细信息:")
    for i, data_point in enumerate(results[:5]):
        print(f"\n{i + 1}. 需求: {data_point.content[:60]}...")
        print(f"   人工标注: {data_point.true_label}")
        if data_point.prediction:
            print(f"   预测标签: {data_point.prediction.label}")
            print(f"   大模型自评置信度: {data_point.prediction.confidence:.3f}")
            print(f"   是否正确: {'是' if data_point.prediction.label == data_point.true_label else '否'}")

    print(f"\n🎉 GPT-4.1-Nano 语言学者分析完成!")
    print(f"📄 结果文件: {output_file}")
    print(f"💡 注意: 为了消除顺序偏差，每个需求的类别顺序都进行了随机打乱")
    print(f"📱 使用的平台: OpenAI代理 (api.openai-proxy.org)")
    print(f"🤖 使用的模型: GPT-4.1-Nano")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    main()