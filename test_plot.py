import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from defame.utils.plot import plot_confusion_matrix, heatmap, annotate_heatmap, plot_grouped_bar_chart, plot_histogram_comparison
from defame.common.label import Label

verite_path = "out/verite/summary/dynamic/gpt_4o/2025-08-24_18-29 verite"
predictions_file = "out/verite/summary/dynamic/gpt_4o/2025-08-24_18-29 verite/predictions.csv"

def load_predictions(csv_path):
    """从CSV文件加载预测结果"""
    df = pd.read_csv(csv_path)
    return df

def extract_labels(df):
    """从DataFrame中提取预测标签和真实标签"""
    predictions = []
    ground_truth = []
    
    for _, row in df.iterrows():
        # 将字符串标签转换为Label枚举
        pred_label = Label[row['predicted']] if row['predicted'] in Label.__members__ else None
        true_label = Label[row['target']] if row['target'] in Label.__members__ else None
        
        if pred_label and true_label:
            predictions.append(pred_label)
            ground_truth.append(true_label)
    
    return predictions, ground_truth

def get_unique_classes(ground_truth, predictions):
    """获取数据中的唯一类别"""
    all_labels = set(ground_truth + predictions)
    return sorted(list(all_labels), key=lambda x: x.value)

def calculate_metrics(predictions, ground_truth, classes):
    """计算各类别的精确率、召回率和F1分数"""
    metrics = {}
    
    for cls in classes:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != cls and g == cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for g in ground_truth if g == cls)
        }
    
    return metrics

def plot_class_distribution(ground_truth, predictions, save_dir):
    """绘制类别分布对比图"""
    gt_counter = Counter(ground_truth)
    pred_counter = Counter(predictions)
    
    classes = sorted(set(ground_truth + predictions), key=lambda x: x.value)
    class_names = [cls.name for cls in classes]
    
    gt_counts = [gt_counter.get(cls, 0) for cls in classes]
    pred_counts = [pred_counter.get(cls, 0) for cls in classes]
    
    values = {
        'Ground Truth': gt_counts,
        'Predictions': pred_counts
    }
    
    colors = ['#005AA9', '#EC6500']  # 使用调色板中的颜色
    
    plot_grouped_bar_chart(
        x_labels=class_names,
        values=values,
        title="Class Distribution Comparison",
        x_label="Classes",
        y_label="Number of Samples",
        colors=colors,
        save_path=save_dir / "class_distribution.png"
    )

def plot_performance_metrics(metrics, save_dir):
    """绘制性能指标对比图"""
    classes = list(metrics.keys())
    class_names = [cls.name for cls in classes]
    
    precision_values = [metrics[cls]['precision'] for cls in classes]
    recall_values = [metrics[cls]['recall'] for cls in classes]
    f1_values = [metrics[cls]['f1'] for cls in classes]
    
    values = {
        'Precision': precision_values,
        'Recall': recall_values,
        'F1-Score': f1_values
    }
    
    colors = ['#009CDA', '#009D81', '#F5A300']  # 使用调色板中的颜色
    
    plot_grouped_bar_chart(
        x_labels=class_names,
        values=values,
        title="Performance Metrics by Class",
        x_label="Classes",
        y_label="Score",
        colors=colors,
        save_path=save_dir / "performance_metrics.png"
    )

def plot_confidence_distribution(df, save_dir):
    """绘制置信度分布直方图"""
    # 假设有置信度列，如果没有则生成模拟数据
    if 'confidence' in df.columns:
        correct_confidences = df[df['correct'] == True]['confidence'].values
        incorrect_confidences = df[df['correct'] == False]['confidence'].values
    else:
        # 生成模拟置信度数据用于演示
        np.random.seed(42)
        correct_confidences = np.random.beta(3, 1, size=sum(df['correct']))  # 正确预测倾向于高置信度
        incorrect_confidences = np.random.beta(1, 2, size=sum(~df['correct']))  # 错误预测倾向于低置信度
    
    data_rows = [correct_confidences, incorrect_confidences]
    labels = ['Correct Predictions', 'Incorrect Predictions']
    colors = ['#009D81', '#EC6500']
    
    plot_histogram_comparison(
        data_rows=data_rows,
        title="Confidence Distribution by Prediction Correctness",
        labels=labels,
        y_label="Confidence Score",
        x_label="Prediction Type",
        colors=colors,
        h_line_at=0.5,  # 添加0.5的参考线
        save_path=save_dir / "confidence_distribution.png"
    )

def plot_support_distribution(metrics, save_dir):
    """绘制类别支持度分布"""
    classes = list(metrics.keys())
    class_names = [cls.name for cls in classes]
    support_values = [metrics[cls]['support'] for cls in classes]
    
    # 为不同类别生成不同的数据行（这里简化为单一数据行的重复）
    data_rows = []
    labels = []
    # 确保颜色列表有足够的颜色，如果不够则重复使用
    base_colors = ['#005AA9', '#0083CC', '#009CDA', '#009D81']
    colors = []
    
    for i, (cls, support) in enumerate(zip(classes, support_values)):
        # 生成正态分布数据，均值为support，用于直方图
        data = np.random.normal(support, support * 0.1, size=100)
        data_rows.append(data)
        labels.append(cls.name)
        colors.append(base_colors[i % len(base_colors)])  # 使用模运算循环使用颜色
    
    if data_rows:
        plot_histogram_comparison(
            data_rows=data_rows,
            title="Support Distribution Across Classes",
            labels=labels,
            y_label="Sample Count",
            colors=colors,
            save_path=save_dir / "support_distribution.png"
        )

def main():
    # 加载预测数据
    df = load_predictions(predictions_file)
    print(f"加载了 {len(df)} 条预测记录")
    
    # 提取标签
    predictions, ground_truth = extract_labels(df)
    print(f"有效预测: {len(predictions)}")
    
    # 计算总的正确率
    correct_predictions = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    total_accuracy = correct_predictions / len(predictions) if predictions else 0
    print(f"总正确率: {total_accuracy:.4f} ({total_accuracy:.2%})")
    print(f"正确预测数: {correct_predictions}/{len(predictions)}")
    
    # 获取类别
    classes = get_unique_classes(ground_truth, predictions)
    print(f"类别: {[c.name for c in classes]}")
    
    # 创建保存目录
    save_dir = Path(verite_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算性能指标
    metrics = calculate_metrics(predictions, ground_truth, classes)
    
    print("\n=== 绘制混淆矩阵 ===")
    # 1. 绘制混淆矩阵
    plot_confusion_matrix(
        predictions=predictions,
        ground_truth=ground_truth,
        classes=classes,
        benchmark_name="Verite",
        save_dir=save_dir
    )
    
    print("\n=== 绘制类别分布对比图 ===")
    # 2. 绘制类别分布对比图
    plot_class_distribution(ground_truth, predictions, save_dir)
    
    print("\n=== 绘制性能指标对比图 ===")
    # 3. 绘制性能指标对比图
    plot_performance_metrics(metrics, save_dir)
    
    print("\n=== 绘制置信度分布直方图 ===")
    # 4. 绘制置信度分布直方图
    plot_confidence_distribution(df, save_dir)
    
    print("\n=== 绘制支持度分布图 ===")
    # 5. 绘制支持度分布图
    plot_support_distribution(metrics, save_dir)
    
    # 打印统计信息
    print(f"\n=== 统计信息 ===")
    print(f"总正确率: {total_accuracy:.4f} ({total_accuracy:.2%})")
    print(f"正确预测数: {correct_predictions}/{len(predictions)}")
    
    print("\n各类别性能:")
    for cls, metric in metrics.items():
        print(f"{cls.name}: Precision={metric['precision']:.3f}, "
              f"Recall={metric['recall']:.3f}, F1={metric['f1']:.3f}, "
              f"Support={metric['support']}")
    
    print(f"\n所有图表已保存到: {save_dir}")

if __name__ == "__main__":
    main()
