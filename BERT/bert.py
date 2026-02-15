import pandas as pd
import numpy as np
import torch
import os
import sys
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, \
    precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')


# 设置随机种子以确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# 1. 数据加载与预处理
def load_and_preprocess_data():
    print("1. Loading and preprocessing data...")

    # 设置文件路径 - 上一级目录
    file_path = '../1226datasetForUpload.xlsx'  # 上一级目录
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    print(f"Attempting to load file: {file_path}")

    try:
        # 读取Excel文件的Sheet2
        df = pd.read_excel(file_path,
                           sheet_name='Sheet2',  # 指定Sheet2
                           engine='openpyxl')

        print(f"Successfully loaded file: {file_path} (Sheet2)")
        print(f"File columns: {df.columns.tolist()}")

        # 检查是否有requirement和label列
        if 'requirement' not in df.columns or 'label' not in df.columns:
            print("Warning: 'requirement' or 'label' columns not found, attempting automatic matching...")

            # 尝试自动匹配列名
            for col in df.columns:
                col_lower = col.lower()
                if 'requirement' in col_lower or '需求' in col_lower or 'text' in col_lower or '内容' in col_lower:
                    df = df.rename(columns={col: 'requirement'})
                    print(f"  Renamed column '{col}' to 'requirement'")

                if 'label' in col_lower or '标签' in col_lower or '类别' in col_lower or 'category' in col_lower or '分类' in col_lower:
                    df = df.rename(columns={col: 'label'})
                    print(f"  Renamed column '{col}' to 'label'")

        # 检查是否成功重命名
        if 'requirement' not in df.columns or 'label' not in df.columns:
            print(f"Error: Required columns not found")
            print(f"Available columns: {df.columns.tolist()}")
            print("Please ensure the file contains requirement text column and label column")
            return None

    except Exception as e:
        print(f"Failed to load file: {e}")
        return None

    # 查看数据基本信息
    print(f"\nDataset size: {df.shape}")
    print(f"Data columns: {df.columns.tolist()}")
    print("\nData preview (first 5 rows):")
    print(df[['requirement', 'label']].head())
    print("\nMissing value check:")
    print(df[['requirement', 'label']].isnull().sum())

    # 数据清洗：移除requirement或label为空的行
    print(f"\nDataset size before cleaning: {df.shape}")
    df = df.dropna(subset=['requirement', 'label'])
    print(f"Dataset size after cleaning: {df.shape}")

    if len(df) == 0:
        print("Error: Dataset is empty after cleaning, please check the data file")
        return None

    return df


# 2. 创建自定义Dataset类
class RequirementDataset(Dataset):
    def __init__(self, requirements, labels, tokenizer, max_length=128):
        self.requirements = requirements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.requirements)

    def __getitem__(self, idx):
        requirement = str(self.requirements[idx])
        label = self.labels[idx]

        # 对文本进行编码
        encoding = self.tokenizer(
            requirement,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 3. 准备训练集和测试集
def prepare_datasets(df):
    print("\n2. Splitting training and test sets...")

    # 处理label列：确保标签是数值类型
    print("\nUnique values in label column:")
    unique_labels = df['label'].unique()
    print(f"Total {len(unique_labels)} unique labels:")
    for i, label in enumerate(unique_labels[:10]):  # 只显示前10个
        print(f"  {i + 1}. {label} ({type(label).__name__})")
    if len(unique_labels) > 10:
        print(f"  ... and {len(unique_labels) - 10} more labels")

    # 将label转换为数值
    if df['label'].dtype == 'object':
        # 如果是字符串标签，创建映射
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        df['label'] = df['label'].map(label_mapping)
        print(f"\nCreated label mapping (first 10):")
        for label, idx in list(label_mapping.items())[:10]:
            print(f"  '{label}' -> {idx}")
    else:
        # 如果已经是数值，确保是整数
        df['label'] = df['label'].astype(int)
        unique_labels = sorted(df['label'].unique())
        label_mapping = {idx: idx for idx in unique_labels}

    # 统计类别分布
    print("\nClass distribution:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        print(f"Class {label}: {count} samples ({percentage:.2f}%)")

    # 设置类别数量
    num_labels = len(unique_labels)
    print(f"\nTotal number of classes: {num_labels}")

    # 按8:2比例划分
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['label']  # 分层抽样保持类别分布
        )
    except ValueError as e:
        print(f"Stratified sampling failed: {e}")
        print("Using random sampling instead...")
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    return train_df, test_df, label_mapping, num_labels


# 4. 训练函数
def train_epoch(model, data_loader, optimizer, scheduler, device, batch_size):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch_idx, batch in enumerate(data_loader):
        # 将数据移到设备上
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        scheduler.step()

        # 统计
        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        # 每10个batch显示一次进度
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = correct_predictions.double() / ((batch_idx + 1) * batch_size)
            print(f"  Batch {batch_idx + 1}/{len(data_loader)}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / len(data_loader.dataset)

    return avg_loss, accuracy


# 5. 评估函数（带置信度）
def eval_model_with_confidence(model, data_loader, device):
    """评估模型，返回预测结果和置信度"""
    model.eval()
    predictions = []
    true_labels = []
    confidences = []
    all_probabilities = []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            # 计算softmax概率
            probabilities = torch.softmax(logits, dim=1)

            # 获取预测类别和置信度（最大概率）
            conf, preds = torch.max(probabilities, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)

    return avg_loss, accuracy, predictions, true_labels, confidences, all_probabilities


# 6. 计算准确率与置信度关系指标
def calculate_confidence_metrics(true_labels, predictions, confidences, n_bins=10):
    """
    计算准确率与置信度的关系指标

    参数:
    - true_labels: 真实标签
    - predictions: 预测标签
    - confidences: 置信度（最大概率）
    - n_bins: 分箱数量

    返回:
    - 包含各种指标的字典
    """

    print("\n" + "=" * 60)
    print("Accuracy vs Confidence Analysis")
    print("=" * 60)

    # 转换为numpy数组
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    confidences = np.array(confidences)

    # 1. 计算皮尔逊相关系数
    correct = (true_labels == predictions).astype(int)
    pearson_corr, pearson_p = pearsonr(confidences, correct)

    print(f"\n1. Pearson Correlation between Confidence and Accuracy:")
    print(f"   Correlation coefficient: {pearson_corr:.4f}")
    print(f"   P-value: {pearson_p:.4e}")

    if pearson_corr > 0:
        print(f"   Interpretation: Positive correlation - higher confidence tends to predict higher accuracy")
    else:
        print(f"   Interpretation: Negative correlation - higher confidence tends to predict lower accuracy")

    # 2. 计算预期校准误差（ECE）
    print(f"\n2. Expected Calibration Error (ECE) with {n_bins} bins:")

    # 创建分箱
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0  # 最大校准误差
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    bin_eces = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到置信度在此区间的样本
        in_bin_mask = (confidences >= bin_lower) & (confidences < bin_upper)
        n_in_bin = np.sum(in_bin_mask)

        if n_in_bin > 0:
            # 计算该区间内的准确性
            accuracy_in_bin = np.mean(predictions[in_bin_mask] == true_labels[in_bin_mask])
            # 计算该区间内的平均置信度
            avg_confidence_in_bin = np.mean(confidences[in_bin_mask])

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(n_in_bin)

            # 计算该区间的校准误差
            bin_ece = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            bin_eces.append(bin_ece)

            # 累积ECE
            ece += bin_ece * n_in_bin

            # 更新MCE
            mce = max(mce, bin_ece)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
            bin_eces.append(0)

    # 最终ECE
    ece = ece / len(true_labels) if len(true_labels) > 0 else 0

    print(f"   ECE: {ece:.4f}")
    print(f"   MCE (Maximum Calibration Error): {mce:.4f}")

    # 3. 按置信度分组的准确性
    print(f"\n3. Accuracy by Confidence Bins:")

    for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        n_in_bin = bin_counts[i]
        if n_in_bin > 0:
            accuracy = bin_accuracies[i]
            avg_conf = bin_confidences[i]
            bin_ece = bin_eces[i]
            print(f"   Bin {i + 1:2d} [{lower:.2f}, {upper:.2f}]: {n_in_bin:4d} samples, "
                  f"Avg Conf: {avg_conf:.3f}, Accuracy: {accuracy:.3f}, "
                  f"Bin ECE: {bin_ece:.3f}")

    # 4. 计算可靠性和绘制校准曲线
    print(f"\n4. Reliability Analysis:")

    # 整体可靠性
    overall_accuracy = np.mean(predictions == true_labels)
    overall_confidence = np.mean(confidences)
    reliability_gap = np.abs(overall_confidence - overall_accuracy)

    print(f"   Overall Accuracy: {overall_accuracy:.4f}")
    print(f"   Overall Confidence: {overall_confidence:.4f}")
    print(f"   Reliability Gap: {reliability_gap:.4f}")

    # 高置信度样本分析
    high_conf_threshold = 0.8
    high_conf_mask = confidences >= high_conf_threshold
    n_high_conf = np.sum(high_conf_mask)

    if n_high_conf > 0:
        high_conf_accuracy = np.mean(predictions[high_conf_mask] == true_labels[high_conf_mask])
        print(f"\n5. High Confidence Samples (≥{high_conf_threshold:.2f}):")
        print(f"   Number of samples: {n_high_conf} ({n_high_conf / len(true_labels) * 100:.1f}%)")
        print(f"   Accuracy: {high_conf_accuracy:.4f}")
    else:
        high_conf_accuracy = 0
        print(f"\n5. High Confidence Samples (≥{high_conf_threshold:.2f}):")
        print(f"   No samples with confidence ≥ {high_conf_threshold:.2f}")

    # 低置信度样本分析
    low_conf_threshold = 0.5
    low_conf_mask = confidences < low_conf_threshold
    n_low_conf = np.sum(low_conf_mask)

    if n_low_conf > 0:
        low_conf_accuracy = np.mean(predictions[low_conf_mask] == true_labels[low_conf_mask])
        print(f"\n6. Low Confidence Samples (<{low_conf_threshold:.2f}):")
        print(f"   Number of samples: {n_low_conf} ({n_low_conf / len(true_labels) * 100:.1f}%)")
        print(f"   Accuracy: {low_conf_accuracy:.4f}")
    else:
        low_conf_accuracy = 0
        print(f"\n6. Low Confidence Samples (<{low_conf_threshold:.2f}):")
        print(f"   No samples with confidence < {low_conf_threshold:.2f}")

    # 返回所有指标
    metrics = {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'ece': ece,
        'mce': mce,
        'overall_accuracy': overall_accuracy,
        'overall_confidence': overall_confidence,
        'reliability_gap': reliability_gap,
        'high_conf_threshold': high_conf_threshold,
        'high_conf_samples': n_high_conf,
        'high_conf_accuracy': high_conf_accuracy,
        'low_conf_threshold': low_conf_threshold,
        'low_conf_samples': n_low_conf,
        'low_conf_accuracy': low_conf_accuracy,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'bin_eces': bin_eces,
        'bin_boundaries': bin_boundaries
    }

    return metrics


# 7. 生成详细结果Excel
def generate_results_excel(test_df, predictions, true_labels, confidences,
                           label_names, performance_metrics, confidence_metrics,
                           error_matrix, filename='bert_classification_results.xlsx'):
    """
    生成包含所有结果的Excel文件

    参数:
    - test_df: 测试集DataFrame
    - predictions: 预测标签
    - true_labels: 真实标签
    - confidences: 置信度
    - label_names: 标签名称列表
    - performance_metrics: 性能指标字典
    - confidence_metrics: 置信度指标字典
    - error_matrix: 错误矩阵
    - filename: 输出文件名
    """

    print(f"\nGenerating Excel report: {filename}")

    # 创建Excel写入器
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:

        # Sheet 1: 详细预测结果
        detailed_results = pd.DataFrame({
            'Requirement': test_df['requirement'].values if 'requirement' in test_df.columns else [''] * len(test_df),
            'True_Label_Index': true_labels,
            'True_Label_Name': [label_names[t] if t < len(label_names) else str(t) for t in true_labels],
            'Predicted_Label_Index': predictions,
            'Predicted_Label_Name': [label_names[p] if p < len(label_names) else str(p) for p in predictions],
            'Confidence': confidences,
            'Is_Correct': [1 if t == p else 0 for t, p in zip(true_labels, predictions)],
            'Error_Type': [
                'Correct' if t == p else f'Misclassified as {label_names[p] if p < len(label_names) else str(p)}'
                for t, p in zip(true_labels, predictions)]
        })

        # 对错误样本排序
        detailed_results = detailed_results.sort_values(['Is_Correct', 'Confidence'], ascending=[True, False])

        # 写入Sheet1
        detailed_results.to_excel(writer, sheet_name='Detailed_Predictions', index=False)

        # Sheet 2: 总体性能指标
        overall_metrics = pd.DataFrame({
            'Metric': [
                'Overall Accuracy',
                'Weighted F1 Score',
                'Macro F1 Score',
                'Pearson Correlation (Confidence vs Accuracy)',
                'Pearson P-value',
                'Expected Calibration Error (ECE)',
                'Maximum Calibration Error (MCE)',
                'Overall Confidence',
                'Reliability Gap',
                f'High Confidence (≥{confidence_metrics["high_conf_threshold"]}) Samples',
                f'High Confidence Accuracy',
                f'Low Confidence (<{confidence_metrics["low_conf_threshold"]}) Samples',
                f'Low Confidence Accuracy'
            ],
            'Value': [
                performance_metrics.get('accuracy', 0),
                performance_metrics.get('weighted_f1', 0),
                performance_metrics.get('macro_f1', 0),
                confidence_metrics.get('pearson_correlation', 0),
                confidence_metrics.get('pearson_p_value', 0),
                confidence_metrics.get('ece', 0),
                confidence_metrics.get('mce', 0),
                confidence_metrics.get('overall_confidence', 0),
                confidence_metrics.get('reliability_gap', 0),
                confidence_metrics.get('high_conf_samples', 0),
                confidence_metrics.get('high_conf_accuracy', 0),
                confidence_metrics.get('low_conf_samples', 0),
                confidence_metrics.get('low_conf_accuracy', 0)
            ],
            'Description': [
                'Percentage of correctly classified samples',
                'F1 score weighted by class support',
                'Unweighted average of per-class F1 scores',
                'Correlation between prediction confidence and accuracy',
                'Statistical significance of the correlation',
                'Expected Calibration Error - measures how well confidence matches accuracy',
                'Maximum calibration error across all bins',
                'Average confidence across all predictions',
                'Absolute difference between overall confidence and accuracy',
                f'Number of samples with confidence ≥ {confidence_metrics["high_conf_threshold"]}',
                f'Accuracy of samples with confidence ≥ {confidence_metrics["high_conf_threshold"]}',
                f'Number of samples with confidence < {confidence_metrics["low_conf_threshold"]}',
                f'Accuracy of samples with confidence < {confidence_metrics["low_conf_threshold"]}'
            ]
        })

        overall_metrics.to_excel(writer, sheet_name='Overall_Metrics', index=False)

        # Sheet 3: 按类别性能指标
        if 'category_metrics' in performance_metrics:
            category_data = []
            for label_name, metrics in performance_metrics['category_metrics'].items():
                category_data.append({
                    'Class_Name': label_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1_Score': metrics['f1'],
                    'Support': metrics['support']
                })

            category_metrics_df = pd.DataFrame(category_data)
            category_metrics_df.to_excel(writer, sheet_name='Per_Class_Metrics', index=False)

        # Sheet 4: 置信度分箱分析
        bin_data = []
        bin_boundaries = confidence_metrics.get('bin_boundaries', [])
        bin_accuracies = confidence_metrics.get('bin_accuracies', [])
        bin_confidences = confidence_metrics.get('bin_confidences', [])
        bin_counts = confidence_metrics.get('bin_counts', [])
        bin_eces = confidence_metrics.get('bin_eces', [])

        for i in range(len(bin_boundaries) - 1):
            bin_data.append({
                'Bin_Index': i + 1,
                'Confidence_Lower': bin_boundaries[i],
                'Confidence_Upper': bin_boundaries[i + 1],
                'Sample_Count': bin_counts[i] if i < len(bin_counts) else 0,
                'Average_Confidence': bin_confidences[i] if i < len(bin_confidences) else 0,
                'Accuracy': bin_accuracies[i] if i < len(bin_accuracies) else 0,
                'Bin_ECE': bin_eces[i] if i < len(bin_eces) else 0,
                'Confidence_Accuracy_Gap': abs(bin_confidences[i] - bin_accuracies[i]) if i < len(
                    bin_confidences) and i < len(bin_accuracies) else 0
            })

        bin_analysis_df = pd.DataFrame(bin_data)
        bin_analysis_df.to_excel(writer, sheet_name='Confidence_Bin_Analysis', index=False)

        # Sheet 5: 错误分析矩阵
        if error_matrix:
            error_data = []
            for true_label, pred_dict in error_matrix.items():
                for pred_label, count in pred_dict.items():
                    error_data.append({
                        'True_Label': true_label,
                        'Predicted_Label': pred_label,
                        'Count': count
                    })

            error_matrix_df = pd.DataFrame(error_data)
            # 按错误数量排序
            error_matrix_df = error_matrix_df.sort_values('Count', ascending=False)
            error_matrix_df.to_excel(writer, sheet_name='Error_Analysis', index=False)

        # Sheet 6: 样本级别的置信度-准确性分析
        sample_level_data = pd.DataFrame({
            'Sample_Index': range(len(true_labels)),
            'True_Label': true_labels,
            'Predicted_Label': predictions,
            'Confidence': confidences,
            'Is_Correct': [1 if t == p else 0 for t, p in zip(true_labels, predictions)],
            'Confidence_Bin': pd.cut(confidences, bins=bin_boundaries,
                                     labels=[f'Bin_{i + 1}' for i in range(len(bin_boundaries) - 1)])
        })

        sample_level_data.to_excel(writer, sheet_name='Sample_Level_Analysis', index=False)

        print(f"Excel report saved to {filename}")
        print(f"Number of sheets: {len(writer.sheets)}")

    return filename


# 8. 绘制校准曲线
def plot_calibration_curve(confidence_metrics, filename='calibration_curve.png'):
    """
    绘制校准曲线

    参数:
    - confidence_metrics: 置信度指标字典
    - filename: 输出文件名
    """

    bin_confidences = confidence_metrics.get('bin_confidences', [])
    bin_accuracies = confidence_metrics.get('bin_accuracies', [])
    bin_counts = confidence_metrics.get('bin_counts', [])

    # 过滤掉没有样本的分箱
    valid_indices = [i for i, count in enumerate(bin_counts) if count > 0]
    if len(valid_indices) == 0:
        print("No valid bins to plot calibration curve")
        return

    valid_confidences = [bin_confidences[i] for i in valid_indices]
    valid_accuracies = [bin_accuracies[i] for i in valid_indices]
    valid_counts = [bin_counts[i] for i in valid_indices]

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: 校准曲线
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.7)
    ax1.scatter(valid_confidences, valid_accuracies, s=100, c='red', alpha=0.7, label='Model Calibration')

    # 添加误差条（用样本数量加权）
    for conf, acc, count in zip(valid_confidences, valid_accuracies, valid_counts):
        # 误差条大小与样本数量成比例
        size = 5 + (count / max(valid_counts)) * 20 if max(valid_counts) > 0 else 10
        ax1.scatter(conf, acc, s=size, c='red', alpha=0.7)

    ax1.set_xlabel('Average Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # 添加ECE信息
    ece = confidence_metrics.get('ece', 0)
    ax1.text(0.05, 0.95, f'ECE = {ece:.4f}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 子图2: 置信度分布直方图
    ax2.hist(confidence_metrics.get('confidences_for_hist', [0]), bins=20,
             color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])

    # 添加统计信息
    mean_conf = confidence_metrics.get('overall_confidence', 0)
    ax2.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
                label=f'Mean Confidence = {mean_conf:.3f}')
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Calibration curve saved to {filename}")


# 9. 推理函数
def predict_requirement(requirement_text, model, tokenizer, device, label_names):
    """Predict classification for a single requirement"""
    model.eval()

    # 编码文本
    encoding = tokenizer(
        requirement_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        _, predicted_class = torch.max(logits, dim=1)

    # 获取概率和类别名称
    probs = probabilities.cpu().numpy()[0]
    predicted_idx = predicted_class.item()

    if predicted_idx < len(label_names):
        predicted_label = label_names[predicted_idx]
    else:
        predicted_label = str(predicted_idx)

    return {
        'label': predicted_label,
        'confidence': float(probs[predicted_idx]),
        'probabilities': {label_names[i] if i < len(label_names) else str(i): float(probs[i])
                          for i in range(len(probs))}
    }


# 10. 性能分析函数
def analyze_performance(true_labels, predictions, label_names):
    """Analyze model performance"""

    print("\n" + "=" * 60)
    print("Model Performance Detailed Analysis")
    print("=" * 60)

    # 计算总体指标
    accuracy = accuracy_score(true_labels, predictions)
    weighted_f1 = f1_score(true_labels, predictions, average='weighted')
    macro_f1 = f1_score(true_labels, predictions, average='macro')

    print(f"\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    # 计算每个类别的指标
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None
    )

    print("\nPerformance by Class:")
    for i, label in enumerate(label_names):
        print(f"{label}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, "
              f"F1={f1[i]:.3f}, Support={support[i]}")

    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'category_metrics': {
            label: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
            for i, label in enumerate(label_names)
        }
    }


# 11. 错误分析函数
def analyze_misclassifications(test_df, predictions, true_labels, label_names):
    """Analyze misclassified samples"""

    print("\n" + "=" * 60)
    print("Misclassification Analysis")
    print("=" * 60)

    misclassified_indices = [i for i, (p, t) in enumerate(zip(predictions, true_labels)) if p != t]

    print(f"\nTotal misclassifications: {len(misclassified_indices)}")
    print(f"Error rate: {len(misclassified_indices) / len(predictions):.3f}")

    if misclassified_indices:
        # 统计各类别的错误情况
        error_matrix = {}
        for idx in misclassified_indices:
            true_label = label_names[true_labels[idx]] if true_labels[idx] < len(label_names) else str(true_labels[idx])
            pred_label = label_names[predictions[idx]] if predictions[idx] < len(label_names) else str(predictions[idx])

            if true_label not in error_matrix:
                error_matrix[true_label] = {}
            if pred_label not in error_matrix[true_label]:
                error_matrix[true_label][pred_label] = 0
            error_matrix[true_label][pred_label] += 1

        print("\nMisclassification Matrix:")
        for true_label in sorted(error_matrix.keys()):
            print(f"\nTrue label: {true_label}")
            total_errors = sum(error_matrix[true_label].values())
            for pred_label, count in sorted(error_matrix[true_label].items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_errors * 100
                print(f"  Misclassified as {pred_label}: {count} times ({percentage:.1f}%)")

        return misclassified_indices, error_matrix
    else:
        print("No misclassified samples!")
        return [], {}


# 主函数
def main():
    # 1. 加载和预处理数据
    df = load_and_preprocess_data()
    if df is None:
        return

    # 2. 准备数据集
    train_df, test_df, label_mapping, num_labels = prepare_datasets(df)

    # 3. 加载BERT模型和分词器
    print("\n3. Loading BERT model and tokenizer...")

    # 使用中文BERT预训练模型
    model_name = 'bert-base-chinese'  # 中文BERT基础版
    print(f"Using model: {model_name}")

    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    try:
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        print("BERT model loaded successfully")
        print("Note: Classifier weights are randomly initialized and need fine-tuning")
    except Exception as e:
        print(f"Failed to load BERT model: {e}")
        return

    # 4. 创建DataLoader
    print("\n4. Creating data loaders...")

    train_dataset = RequirementDataset(
        train_df['requirement'].values,
        train_df['label'].values,
        tokenizer
    )

    test_dataset = RequirementDataset(
        test_df['requirement'].values,
        test_df['label'].values,
        tokenizer
    )

    # 设置批大小（根据数据量调整）
    if len(train_df) < 100:
        batch_size = 8  # 小数据集使用小批次
    elif len(train_df) < 1000:
        batch_size = 16
    else:
        batch_size = 32

    print(f"Batch size: {batch_size}")

    # Windows下设置num_workers为0以避免多进程问题
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows下设置为0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Windows下设置为0
    )

    # 5. 设置训练参数和优化器
    print("\n5. Setting training parameters...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model.to(device)

    # 训练参数（根据数据量调整）
    if len(train_df) < 500:
        epochs = 10  # 小数据集可能需要更多epoch
    elif len(train_df) < 5000:
        epochs = 5
    else:
        epochs = 3

    learning_rate = 2e-5
    warmup_steps = 0

    print(f"Number of epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 学习率调度器
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 6. 训练循环
    print("\n6. Starting training...")

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, batch_size
        )

        # 评估（带置信度）
        test_loss, test_acc, test_preds, test_true, test_confidences, _ = eval_model_with_confidence(
            model, test_loader, device
        )

        # 保存指标
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Training set - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Test set - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # 7. 最终评估和报告
    print("\n7. Generating detailed evaluation report...")

    # 最终的详细评估（带置信度）
    test_loss, test_acc, predictions, true_labels, confidences, all_probabilities = eval_model_with_confidence(
        model, test_loader, device
    )

    # 创建标签名称列表
    if label_mapping and isinstance(label_mapping, dict):
        # 如果有映射关系，反转映射
        idx_to_label = {v: k for k, v in label_mapping.items()}
        label_names = [str(idx_to_label[i]) for i in range(num_labels)]
    else:
        # 如果没有映射，使用数字作为标签名
        label_names = [str(i) for i in range(num_labels)]

    print(f"Label names: {label_names}")

    # 分类报告
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions,
                                target_names=label_names,
                                digits=4))

    # 8. 计算准确率与置信度关系指标
    confidence_metrics = calculate_confidence_metrics(
        true_labels, predictions, confidences, n_bins=10
    )

    # 为绘图添加置信度直方图数据
    confidence_metrics['confidences_for_hist'] = confidences

    # 9. 性能分析
    performance_metrics = analyze_performance(true_labels, predictions, label_names)

    # 10. 错误分析
    misclassified_indices, error_matrix = analyze_misclassifications(
        test_df, predictions, true_labels, label_names
    )

    # 11. 生成Excel报告
    excel_filename = generate_results_excel(
        test_df, predictions, true_labels, confidences,
        label_names, performance_metrics, confidence_metrics,
        error_matrix, filename='bert_classification_results.xlsx'
    )

    # 12. 绘制校准曲线
    plot_calibration_curve(confidence_metrics, filename='calibration_curve.png')

    # 13. 混淆矩阵 - 英文标注
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 14. 可视化训练过程 - 英文标注
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1.plot(train_losses, label='Training Loss', marker='o', linewidth=2, markersize=8)
    ax1.plot(test_losses, label='Test Loss', marker='s', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # 准确率曲线
    ax2.plot(train_accuracies, label='Training Accuracy', marker='o', linewidth=2, markersize=8)
    ax2.plot(test_accuracies, label='Test Accuracy', marker='s', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Test Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 15. 保存模型
    print("\n8. Saving model...")

    # 保存完整模型
    model_info = {
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'label_mapping': label_mapping,
        'model_config': model.config,
        'num_labels': num_labels,
        'label_names': label_names,
        'test_accuracy': test_acc,
        'confidence_metrics': confidence_metrics,
        'performance_metrics': performance_metrics
    }

    torch.save(model_info, 'bert_requirement_classifier.pth')
    print("Model saved as 'bert_requirement_classifier.pth'")

    # 16. 测试推理函数
    print("\n9. Testing inference function...")
    print("-" * 50)

    # 从测试集中选取几个例子进行预测
    test_samples = list(test_df.head(min(5, len(test_df))).iterrows())

    if len(test_samples) > 0:
        for idx, (_, row) in enumerate(test_samples):
            true_label_idx = row['label']
            requirement_text = row['requirement']

            # 获取真实标签名称
            if true_label_idx < len(label_names):
                true_label_name = label_names[true_label_idx]
            else:
                true_label_name = str(true_label_idx)

            result = predict_requirement(
                requirement_text, model, tokenizer, device, label_names
            )

            print(f"Example {idx + 1}:")
            if len(requirement_text) > 100:
                print(f"Requirement: {requirement_text[:100]}...")
            else:
                print(f"Requirement: {requirement_text}")
            print(f"True label: {true_label_name} (index: {true_label_idx})")
            print(f"Predicted label: {result['label']} (confidence: {result['confidence']:.3f})")

            # 只显示前3个类别的概率
            top_probs = dict(sorted(result['probabilities'].items(),
                                    key=lambda x: x[1], reverse=True)[:3])
            print(f"Top 3 probabilities: {top_probs}")
            print("-" * 40)
    else:
        print("Test set is empty, cannot perform inference test")

    # 17. 创建额外的可视化图表 - 类别分布图（英文）
    print("\n10. Creating additional visualizations...")

    # 类别分布条形图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 训练集类别分布
    train_counts = train_df['label'].value_counts().sort_index()
    train_percentages = train_counts / len(train_df) * 100

    bars1 = ax1.bar(range(len(train_counts)), train_counts.values, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(train_counts)))
    ax1.set_xticklabels([label_names[i] if i < len(label_names) else str(i) for i in train_counts.index], rotation=45,
                        ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # 在条形上添加数值
    for bar, count, percentage in zip(bars1, train_counts.values, train_percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{count}\n({percentage:.1f}%)',
                 ha='center', va='bottom', fontsize=9)

    # 测试集类别分布
    test_counts = test_df['label'].value_counts().sort_index()
    test_percentages = test_counts / len(test_df) * 100

    bars2 = ax2.bar(range(len(test_counts)), test_counts.values, color='darkorange', alpha=0.8)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(test_counts)))
    ax2.set_xticklabels([label_names[i] if i < len(label_names) else str(i) for i in test_counts.index], rotation=45,
                        ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # 在条形上添加数值
    for bar, count, percentage in zip(bars2, test_counts.values, test_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{count}\n({percentage:.1f}%)',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"✓ BERT fine-tuning experiment completed!")
    print(f"✓ Final test accuracy: {test_acc:.4f}")
    print(f"✓ Pearson Correlation (Confidence vs Accuracy): {confidence_metrics['pearson_correlation']:.4f}")
    print(f"✓ Expected Calibration Error (ECE): {confidence_metrics['ece']:.4f}")
    print(f"✓ Model file saved: bert_requirement_classifier.pth")
    print(f"✓ Excel report saved: {excel_filename}")
    print(f"✓ Visualization files saved:")
    print(f"    - confusion_matrix.png")
    print(f"    - training_curves.png")
    print(f"    - class_distribution.png")
    print(f"    - calibration_curve.png")
    print(f"✓ Total training samples: {len(train_df)}")
    print(f"✓ Total test samples: {len(test_df)}")
    print(f"✓ Number of classes: {num_labels}")
    print("=" * 60)


# Windows下需要添加if __name__ == '__main__'保护
if __name__ == '__main__':
    # 添加freeze_support()用于Windows多进程支持
    from multiprocessing import freeze_support

    freeze_support()
    main()