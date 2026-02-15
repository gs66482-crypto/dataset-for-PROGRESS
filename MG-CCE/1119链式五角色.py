import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from volcenginesdkarkruntime import Ark
import time
import random
from collections import Counter

# 加载环境变量
load_dotenv()


@dataclass
class ModelPrediction:
    """存储模型预测结果"""
    label: str
    confidence: float
    model_name: str
    normalized_rank: float = None
    vote_details: Dict[str, int] = None  # 新增：投票详情


@dataclass
class DataPoint:
    """数据点类"""
    content: str
    predictions: List[ModelPrediction] = None
    final_label: str = None

    def __post_init__(self):
        if self.predictions is None:
            self.predictions = []


class LLMClient:
    """LLM客户端类 - 带有角色身份的投票"""

    def __init__(self, name: str, model_name: str, method_code: str, role: str = None):
        self.name = name
        self.model_name = model_name
        self.method_code = method_code
        self.role = role  # 新增：角色身份
        self.client = self._create_client()

        # ERNIE模型速率限制配置
        self.is_ernie = "ERNIE" in name.upper()
        self.last_request_time = 0
        self.min_request_interval = 2.0

    def _create_client(self):
        """根据method列创建客户端"""
        # 豆包模型特殊处理
        if "Ark(api_key=" in self.method_code:
            api_key_env = self.method_code.split('os.getenv("')[1].split('")')[0]
            return Ark(api_key=os.getenv(api_key_env))

        # 其他模型使用OpenAI兼容接口
        else:
            # 解析api_key环境变量名
            api_key_env = None
            if 'api_key=os.getenv(' in self.method_code:
                api_key_part = self.method_code.split('api_key=os.getenv(')[1].split(')')[0]
                api_key_env = api_key_part.strip('"\'')

            # 解析base_url环境变量名
            base_url_env = None
            if 'base_url=os.getenv(' in self.method_code:
                base_url_part = self.method_code.split('base_url=os.getenv(')[1].split(')')[0]
                base_url_env = base_url_part.strip('"\'')

            api_key = os.getenv(api_key_env) if api_key_env else None
            base_url = os.getenv(base_url_env) if base_url_env else None

            if not api_key:
                raise ValueError(f"未找到API密钥环境变量: {api_key_env}")

            return OpenAI(api_key=api_key, base_url=base_url)

    def _rate_limit_delay(self):
        """ERNIE模型速率限制延迟"""
        if self.is_ernie:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last_request
                print(f"  ERNIE速率限制: 等待 {sleep_time:.2f} 秒")
                time.sleep(sleep_time)
            self.last_request_time = time.time()

    def predict_with_role_voting(self, text: str, labels: List[str],
                                 category_explanations: Dict[str, str] = None,
                                 requirements_examples: Dict[str, List[str]] = None) -> ModelPrediction:
        """基于角色身份的5次投票预测"""

        # ERNIE模型速率限制
        self._rate_limit_delay()

        # 定义角色提示词
        role_prompts = self._create_role_prompts(category_explanations)

        # 进行5次不同角色的投票
        votes = []
        vote_details = {}  # 记录每次投票的详细信息

        print(f"  {self.name} 开始5次角色投票...")

        for i, (role_key, role_info) in enumerate(list(role_prompts.items())[:5]):  # 只取前5个角色
            for attempt in range(3):  # 重试机制
                try:
                    role_name = role_info['name']
                    role_prompt = role_info['prompt']

                    prediction = self._single_prediction_with_role(
                        text, labels, role_name, role_prompt, f"vote_{i + 1}"
                    )

                    if prediction and prediction != "分析失败":
                        votes.append(prediction)
                        vote_details[f"{role_name}_第{i + 1}次"] = prediction
                        print(f"    {role_name}投票: {prediction}")
                        break
                    else:
                        print(f"    {role_name}第{attempt + 1}次尝试失败")
                        time.sleep(1)
                except Exception as e:
                    print(f"    {role_name}投票异常: {str(e)}")
                    time.sleep(2)

            # 投票间延迟
            time.sleep(0.5)

        # 基于投票结果计算置信度
        if votes:
            confidence = self._calculate_confidence_from_votes(votes)
            vote_counts = Counter(votes)
            final_label = vote_counts.most_common(1)[0][0]

            print(f"  投票结果: {dict(vote_counts)}")
            print(f"  最终分类: {final_label}")
            print(f"  置信度: {confidence:.3f}")

            return ModelPrediction(
                label=final_label,
                confidence=confidence,
                model_name=f"{self.name}({self.role})" if self.role else self.name,
                vote_details=dict(vote_counts)  # 保存投票详情
            )
        else:
            print("  所有投票都失败了")
            return ModelPrediction(
                label=labels[0] if labels else "Unknown",
                confidence=0.1,
                model_name=f"{self.name}({self.role})" if self.role else self.name,
                vote_details={}
            )

    def _create_role_prompts(self, category_explanations: Dict[str, str]) -> Dict:
        """创建角色提示词"""
        explanations = ""
        if category_explanations:
            for label, explanation in category_explanations.items():
                explanations += f"- {label}: {explanation}\n"

        role_prompts = {
            'product_owner': {
                'name': '产品负责人',
                'prompt': f"""作为产品负责人，您需要从产品价值和商业目标的角度分析需求。请基于以下分类标准：

{explanations}

请从产品路线图和市场需求的角度，分析以下需求描述语句，判断其最合适的类别。

请直接回复类别名称，不要添加其他内容。"""
            },
            'business_analyst': {
                'name': '业务分析师',
                'prompt': f"""作为业务分析师，您需要从业务流程和功能需求的角度分析需求。请基于以下分类标准：

{explanations}

请从业务流程优化和功能实现的角度，分析以下需求描述语句，判断其最合适的类别。

请直接回复类别名称，不要添加其他内容。"""
            },
            'system_architect': {
                'name': '系统架构师',
                'prompt': f"""作为系统架构师，您需要从技术架构和系统设计的角度分析需求。请基于以下分类标准：

{explanations}

请从技术可行性、系统扩展性和架构影响的角度，分析以下需求描述语句，判断其最合适的类别。

请直接回复类别名称，不要添加其他内容。"""
            },
            'ux_designer': {
                'name': '用户体验设计师',
                'prompt': f"""作为用户体验设计师，您需要从用户交互和界面设计的角度分析需求。请基于以下分类标准：

{explanations}

请从用户体验、界面设计和用户满意度的角度，分析以下需求描述语句，判断其最合适的类别。

请直接回复类别名称，不要添加其他内容。"""
            },
            'software_tester': {
                'name': '软件测试人员',
                'prompt': f"""作为软件测试人员，您需要从测试可行性和质量保证的角度分析需求。请基于以下分类标准：

{explanations}

请从测试覆盖度、质量要求和验证方法的角度，分析以下需求描述语句，判断其最合适的类别。

请直接回复类别名称，不要添加其他内容。"""
            }
        }

        return role_prompts

    def _single_prediction_with_role(self, text: str, labels: List[str],
                                     role_name: str, role_prompt: str, vote_name: str) -> str:
        """单次角色投票预测"""
        try:
            full_prompt = role_prompt + f'\n请分析以下需求描述语句："{text}"'

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=10
            )

            result = completion.choices[0].message.content.strip()
            cleaned_result = self._extract_label_from_response(result, labels)
            return cleaned_result

        except Exception as e:
            print(f"    {vote_name}({role_name}) 预测失败: {str(e)}")
            return "分析失败"

    def _calculate_confidence_from_votes(self, votes: List[str]) -> float:
        """基于投票结果计算置信度"""
        if not votes or len(votes) < 3:  # 至少需要3次有效投票
            return 0.1

        vote_counts = Counter(votes)
        total_votes = len(votes)

        # 获取前两个最频繁的类别
        most_common = vote_counts.most_common(2)

        if len(most_common) == 0:
            return 0.1
        elif len(most_common) == 1:
            # 只有一个类别被投票，置信度基于一致性
            top_count = most_common[0][1]
            return top_count / total_votes
        else:
            # 有两个或更多类别，计算相对优势
            top1_count, top2_count = most_common[0][1], most_common[1][1]
            advantage_ratio = (top1_count - top2_count) / total_votes
            base_confidence = top1_count / total_votes

            # 结合基础置信度和相对优势
            confidence = base_confidence * (1 + advantage_ratio)
            return min(1.0, max(0.1, confidence))

    def _extract_label_from_response(self, response: str, labels: List[str]) -> str:
        """从响应中提取标签"""
        response_clean = response.strip().strip('.,!?;:"')

        # 精确匹配
        if response_clean in labels:
            return response_clean

        # 部分匹配
        for label in labels:
            if label in response_clean:
                return label

        # 默认返回第一个标签
        return labels[0] if labels else "Unknown"


class DataLoader:
    """数据加载器类"""

    @staticmethod
    def load_categories_and_explanations(file_path: str) -> Dict[str, str]:
        """从briefConcept文件加载类别和解释"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            return {
                str(row['category']).strip(): str(row['explanation']).strip()
                for _, row in df.iterrows()
                if pd.notna(row['category'])
            }
        except Exception as e:
            print(f"加载类别文件出错: {e}")
            return {}

    @staticmethod
    def load_requirements_examples(file_path: str) -> Dict[str, List[str]]:
        """从RequirementExamples文件加载需求示例"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            requirements_examples = {}

            for _, row in df.iterrows():
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

            print(f"成功加载 {len(requirements_examples)} 个类别的需求示例")
            return requirements_examples

        except Exception as e:
            print(f"加载需求示例文件出错: {e}")
            return {}

    @staticmethod
    def load_test_requirements(file_path: str) -> List[str]:
        """从sentencesForTest文件加载测试需求"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            return [
                str(row['requirement']).strip()
                for _, row in df.iterrows()
                if 'requirement' in row and pd.notna(row['requirement'])
            ]
        except Exception as e:
            print(f"加载测试需求文件出错: {e}")
            return []

    @staticmethod
    def load_llm_clients_from_excel(file_path: str) -> List[LLMClient]:
        """从Excel文件加载LLM客户端 - 支持角色分配"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            print(f"Excel文件列名: {list(df.columns)}")

            # 尝试不同的列名组合
            possible_name_cols = ['模型名称', '模型名', 'name', 'Name', '模型']
            possible_model_cols = ['model', 'Model', '模型型号', '模型类型']
            possible_method_cols = ['method', 'Method', '调用方法', '方法']
            possible_role_cols = ['role', 'Role', '角色', '身份']  # 新增：角色列

            # 找到实际的列名
            name_col, model_col, method_col, role_col = None, None, None, None

            for col in df.columns:
                if col in possible_name_cols:
                    name_col = col
                elif col in possible_model_cols:
                    model_col = col
                elif col in possible_method_cols:
                    method_col = col
                elif col in possible_role_cols:
                    role_col = col

            if not name_col or not model_col or not method_col:
                print(f"未找到必要的列。找到的列: {list(df.columns)}")
                if len(df.columns) >= 3:
                    name_col = df.columns[0]
                    model_col = df.columns[1]
                    method_col = df.columns[2]
                    print(f"使用默认列: {name_col}, {model_col}, {method_col}")
                else:
                    return []

            clients = []

            for _, row in df.iterrows():
                name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
                model_name = str(row[model_col]).strip() if pd.notna(row[model_col]) else ""
                method_code = str(row[method_col]).strip() if pd.notna(row[method_col]) else ""
                role = str(row[role_col]).strip() if role_col and pd.notna(row[role_col]) else None

                if name and model_name and method_code:
                    try:
                        client = LLMClient(name, model_name, method_code, role)
                        clients.append(client)
                        role_info = f"({role})" if role else ""
                        print(f"✓ 成功加载模型: {name}{role_info}")
                    except Exception as e:
                        print(f"✗ 加载模型 {name} 失败: {e}")

            return clients

        except Exception as e:
            print(f"加载模型配置文件出错: {e}")
            return []


class ChainAlgorithm:
    """链式协同算法主类 - 使用角色投票"""

    def __init__(self, llm_clients: List[LLMClient]):
        """初始化链式算法"""
        self.llm_clients = llm_clients
        self.chain_length = len(llm_clients)
        print(f"链式算法初始化完成，链长度: {self.chain_length}")

    def save_intermediate_results(self, data_points: List[DataPoint], chain_step: int,
                                  client_name: str, output_prefix: str = "链式算法中间结果"):
        """保存中间结果到文件"""
        output_file = f"{output_prefix}_链节{chain_step}_{client_name}.xlsx"

        results_data = []
        for i, data_point in enumerate(data_points):
            row_data = {
                '序号': i + 1,
                '需求内容': data_point.content,
                '当前链节': chain_step,
                '当前模型': client_name
            }

            # 添加所有已完成的预测结果
            for j, pred in enumerate(data_point.predictions):
                row_data[f'模型{j + 1}_名称'] = pred.model_name
                row_data[f'模型{j + 1}_预测标签'] = pred.label
                row_data[f'模型{j + 1}_置信度'] = pred.confidence
                if pred.vote_details:
                    row_data[f'模型{j + 1}_投票详情'] = str(pred.vote_details)
                if pred.normalized_rank is not None:
                    row_data[f'模型{j + 1}_归一化排名'] = pred.normalized_rank

            if data_point.final_label:
                row_data['最终标签'] = data_point.final_label

            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✓ 链节 {chain_step} 中间结果已保存到: {output_file}")

    def process_dataset(self, dataset: List[str], labels: List[str],
                        category_explanations: Dict[str, str] = None,
                        requirements_examples: Dict[str, List[str]] = None) -> List[DataPoint]:
        """处理整个数据集 - 使用角色投票方法"""
        data_points = [DataPoint(content=text) for text in dataset]

        print(f"开始处理 {len(data_points)} 条软件需求，使用 {self.chain_length} 个模型的链式架构...")
        print(f"使用分类解释: {len(category_explanations) if category_explanations else 0} 个")

        current_data = data_points.copy()

        for i, client in enumerate(self.llm_clients, 1):
            print(f"\n链节 {i}/{self.chain_length} ({client.name}) 处理 {len(current_data)} 条数据...")

            # 当前链节处理数据
            for j, data_point in enumerate(current_data):
                if j % 5 == 0 and j > 0:  # 每5条显示一次进度
                    print(f"  进度: {j}/{len(current_data)}")

                # 使用角色投票进行预测
                prediction = client.predict_with_role_voting(
                    data_point.content, labels, category_explanations, requirements_examples
                )
                data_point.predictions.append(prediction)

            # 保存中间结果
            self.save_intermediate_results(data_points, i, client.name)

            # 数据路由
            if i < self.chain_length:
                confidences = [dp.predictions[-1].confidence for dp in current_data]

                pass_ratio = (self.chain_length - i) / self.chain_length
                min_pass_count = max(1, int(len(data_points) * 0.1))
                pass_count = max(min_pass_count, int(len(current_data) * pass_ratio))

                print(f"  传递比例: {pass_ratio:.1%} (共{pass_count}条)")

                # 按置信度从低到高排序，传递低置信度数据
                sorted_indices = np.argsort(confidences)
                next_chain_data = [current_data[idx] for idx in sorted_indices[:pass_count]]
                current_data = next_chain_data

                print(f"  传递给链节 {i + 1} 的数据量: {len(current_data)}")

                if not current_data:
                    print("  没有数据需要传递给下一个链节，提前结束链式处理")
                    break

        # 应用基于排名的集成方法
        self._rank_based_ensemble(data_points)
        return data_points

    def _rank_based_ensemble(self, data_points: List[DataPoint]):
        """基于排名的集成方法"""
        for chain_idx in range(self.chain_length):
            chain_confidences = []
            valid_data_points = []

            for data_point in data_points:
                if len(data_point.predictions) > chain_idx:
                    confidence = data_point.predictions[chain_idx].confidence
                    chain_confidences.append(confidence)
                    valid_data_points.append(data_point)

            if not chain_confidences:
                continue

            ranks = np.argsort(np.argsort(chain_confidences)) + 1
            normalized_ranks = ranks / len(chain_confidences)

            for idx, data_point in enumerate(valid_data_points):
                data_point.predictions[chain_idx].normalized_rank = normalized_ranks[idx]

        # 最终决策：选择具有最高归一化排名的预测
        for data_point in data_points:
            if not data_point.predictions:
                continue

            best_rank = -1
            best_label = None

            for prediction in data_point.predictions:
                if prediction.normalized_rank is not None and prediction.normalized_rank > best_rank:
                    best_rank = prediction.normalized_rank
                    best_label = prediction.label

            if best_label:
                data_point.final_label = best_label
            else:
                self._weighted_voting_fallback(data_point)

    def _weighted_voting_fallback(self, data_point: DataPoint):
        """回退方法：加权投票"""
        predictions = {}
        for pred in data_point.predictions:
            if pred.label not in predictions:
                predictions[pred.label] = 0
            predictions[pred.label] += pred.confidence

        if predictions:
            data_point.final_label = max(predictions.items(), key=lambda x: x[1])[0]

    def save_results(self, results: List[DataPoint], output_file: str):
        """保存最终结果到Excel文件"""
        results_data = []

        for i, data_point in enumerate(results):
            row_data = {
                '序号': i + 1,
                '需求内容': data_point.content,
                '最终分类标签': data_point.final_label
            }

            # 添加每个模型的预测结果
            for j, pred in enumerate(data_point.predictions):
                row_data[f'模型{j + 1}_名称'] = pred.model_name
                row_data[f'模型{j + 1}_预测标签'] = pred.label
                row_data[f'模型{j + 1}_置信度'] = pred.confidence
                if pred.vote_details:
                    row_data[f'模型{j + 1}_投票详情'] = str(pred.vote_details)
                if pred.normalized_rank is not None:
                    row_data[f'模型{j + 1}_归一化排名'] = pred.normalized_rank

            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ 最终结果已保存到: {output_file}")

    def print_statistics(self, results: List[DataPoint]):
        """打印统计信息"""
        print(f"\n📊 链式算法统计信息:")
        print(f"处理总数据量: {len(results)}")

        final_labels = [dp.final_label for dp in results if dp.final_label]
        if final_labels:
            label_counts = {}
            for label in final_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            print(f"\n最终分类分布:")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {label}: {count} 条 ({count / len(results):.1%})")


def main():
    """主函数"""
    # 文件路径配置
    brief_concept_file = "1119Concept.xlsx"
    requirement_examples_file = "1119RequirementExamples.xlsx"
    test_requirements_file = "120sentencesForTest.xlsx"
    model_config_file = "模型调用方法对比.xlsx"
    output_file = "链式算法分类结果_角色投票1119.xlsx"

    data_loader = DataLoader()

    print("正在加载数据...")
    random.seed(42)

    # 加载数据
    test_df, concept_df, examples_df = load_data_files()
    print(f"成功加载 {len(test_df)} 条测试数据")
    print(f"成功加载 {len(concept_df)} 个分类定义")

    # 学习分类概念
    category_explanations = data_loader.load_categories_and_explanations(brief_concept_file)
    requirements_examples = data_loader.load_requirements_examples(requirement_examples_file)
    labels = list(category_explanations.keys())
    test_requirements = data_loader.load_test_requirements(test_requirements_file)

    if not test_requirements:
        print("没有找到测试需求，程序退出")
        return

    if not labels:
        print("没有找到分类标签，程序退出")
        return

    # 加载LLM客户端
    llm_clients = data_loader.load_llm_clients_from_excel(model_config_file)
    if not llm_clients:
        print("没有可用的LLM客户端，程序退出")
        return

    print(f"\n✅ 数据加载完成:")
    print(f"可用分类标签: {len(labels)} 个")
    print(f"测试需求数量: {len(test_requirements)} 条")
    print(f"可用模型: {len(llm_clients)} 个")

    # 执行链式算法
    print(f"\n🚀 开始链式算法处理（角色投票方法）...")
    chain_algo = ChainAlgorithm(llm_clients)
    results = chain_algo.process_dataset(test_requirements, labels, category_explanations, requirements_examples)

    # 保存和显示结果
    chain_algo.save_results(results, output_file)
    chain_algo.print_statistics(results)

    # 显示前几个结果的详细信息
    print(f"\n🔍 前3个结果的详细信息:")
    for i, data_point in enumerate(results[:3]):
        print(f"\n{i + 1}. {data_point.content[:60]}...")
        print(f"   最终分类: {data_point.final_label}")
        for pred in data_point.predictions:
            vote_info = f" 投票: {pred.vote_details}" if pred.vote_details else ""
            rank_info = f" (排名: {pred.normalized_rank:.3f})" if pred.normalized_rank is not None else ""
            print(f"   {pred.model_name}: {pred.label} (置信度: {pred.confidence:.3f}{rank_info}{vote_info})")


def load_data_files():
    """加载数据文件"""
    test_df = pd.read_excel('120sentencesForTest.xlsx', sheet_name='Sheet1')
    concept_df = pd.read_excel('1119Concept.xlsx', sheet_name='Sheet1')
    examples_df = pd.read_excel('1119RequirementExamples.xlsx', sheet_name='Sheet1')
    return test_df, concept_df, examples_df


if __name__ == "__main__":
    main()