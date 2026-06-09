#!/usr/bin/env python3
"""训练评论分析器的独立脚本"""

import sys
from pathlib import Path
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from comment_analyzer import CommentAnalyzer, evaluate_comment_analyzer


def main():
    data_dir = Path(__file__).parent / "data"
    csv_path = data_dir / "filtered_comments.csv"
    
    if not csv_path.exists():
        print(f"错误: 找不到数据文件 {csv_path}")
        return
    
    # 创建分析器
    print("初始化 BERT 分析器...")
    analyzer = CommentAnalyzer(model_name="bert-base-chinese")
    
    # 训练
    print("\n开始训练...")
    results = analyzer.train_from_existing_data(
        str(csv_path),
        model_save_dir=str(data_dir / "models" / "bert_analyzer")
    )
    
    print(f"\n训练完成!")
    print(f"  - 质量评分 MAE: {results['quality_mae']:.3f}")
    print(f"  - 类别分类 F1: {results['category_f1']:.4f}")
    
    # 测试示例
    print("\n测试示例:")
    test_comments = [
        "酒店设施完善，性价比高，wifi给力，环境不错，地理位置好",
        "房间太小了，隔音很差，晚上很吵，价格还贵",
        "早餐很丰富，服务态度好，前台很热情"
    ]
    
    for comment in test_comments:
        score, cats = analyzer.predict(comment)
        print(f"  评论: {comment[:50]}...")
        print(f"  质量分: {score:.1f} | 类别: {cats}\n")


if __name__ == "__main__":
    main()