"""评论质量分析与类别划分模块 - 基于BERT"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support
from typing import Tuple, List, Dict
import json

class CommentAnalyzer:
    """酒店评论分析器：质量评分(1-5) + 多标签类别分类
    
    使用 BERT 系列模型同时处理两个任务：
    - 质量评分：回归任务，预测 1-5 分
    - 类别分类：多标签分类，识别评论涉及的方面（服务/位置/卫生/价格/设施等）
    """
    
    # 预定义的评论类别（根据酒店评论特点）
    CATEGORIES = [
        '整体满意度',
        '前台服务',
        '客房服务',
        '退房/入住效率',
        '房间设施',
        '公共设施',
        '餐饮设施',
        '交通便利性',
        '周边配套',
        '卫生状况',
        '性价比'
    ]
    
    def __init__(self, model_name: str = "bert-base-chinese", device: str = None):
        """初始化分析器
        
        Args:
            model_name: 预训练模型名称，可选：
                - "bert-base-chinese" (基础)
                - "hfl/rbt3" (更小更快)
                - "hfl/chinese-roberta-wwm-ext" (效果更好)
            device: 运行设备 ('cpu' 或 'cuda')
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        
        # 加载基础模型（将在 train/finetune 时替换）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 质量评分模型 (回归头)
        self.quality_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1,  # 回归任务输出1个值
            problem_type="regression"
        ).to(device)
        
        # 类别分类模型 (多标签，输出10个二分类)
        self.category_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.CATEGORIES),
            problem_type="multi_label_classification"
        ).to(device)
        
        self.is_trained = False
        
    def predict(self, comment: str) -> Tuple[float, List[str]]:
        """预测单个评论的质量分和类别
        
        Args:
            comment: 评论文本
            
        Returns:
            (quality_score, categories): 质量分(1-5) 和 类别列表
        """
        if not self.is_trained:
            # 使用启发式规则作为fallback
            return self._heuristic_predict(comment)
        
        inputs = self.tokenizer(
            comment, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        # 质量评分预测
        with torch.no_grad():
            quality_output = self.quality_model(**inputs)
            quality_score = quality_output.logits.item()
            # 限制到 [1, 5] 范围
            quality_score = np.clip(quality_score, 1, 5)
        
        # 类别预测
        with torch.no_grad():
            category_output = self.category_model(**inputs)
            probs = torch.sigmoid(category_output.logits).cpu().numpy()[0]
            # 阈值 0.5 决定是否属于该类
            categories = [
                self.CATEGORIES[i] for i, p in enumerate(probs) if p > 0.5
            ]
        
        return quality_score, categories
    
    def predict_batch(self, comments: List[str]) -> Tuple[np.ndarray, List[List[str]]]:
        """批量预测
        
        Returns:
            (quality_scores, categories_list)
        """
        if not comments:
            return np.array([]), []
        
        inputs = self.tokenizer(
            comments, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            quality_output = self.quality_model(**inputs)
            quality_scores = np.clip(quality_output.logits.cpu().numpy().flatten(), 1, 5)
        
        with torch.no_grad():
            category_output = self.category_model(**inputs)
            probs = torch.sigmoid(category_output.logits).cpu().numpy()
            categories_list = [
                [self.CATEGORIES[i] for i, p in enumerate(prob) if p > 0.5]
                for prob in probs
            ]
        
        return quality_scores, categories_list
    
    def train_from_existing_data(self, csv_path: str, model_save_dir: str = "models/bert_analyzer"):
        """使用现有的 filtered_comments.csv 训练模型
        
        数据中已有 quality_score 和 categories 字段作为训练标签
        
        Args:
            csv_path: filtered_comments.csv 路径
            model_save_dir: 模型保存目录
        """
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from sklearn.model_selection import train_test_split
        
        # 加载数据
        df = pd.read_csv(csv_path)
        print(f"加载 {len(df)} 条评论用于训练")
        
        # 解析 categories (可能是字符串如 "['服务','位置']" 或 "服务,位置")
        def parse_categories(cat_str):
            if pd.isna(cat_str):
                return []
            if isinstance(cat_str, str):
                # 处理 JSON 数组格式
                if cat_str.startswith('['):
                    import ast
                    try:
                        return ast.literal_eval(cat_str)
                    except:
                        return [c.strip() for c in cat_str.strip('[]').split(',')]
                # 处理逗号分隔
                return [c.strip() for c in cat_str.split(',')]
            return []
        
        df['categories_list'] = df['categories'].apply(parse_categories)
        
        # 过滤有效数据
        df_valid = df[df['quality_score'].notna() & (df['quality_score'] > 0)]
        df_valid = df_valid[df_valid['categories_list'].apply(len) > 0]
        print(f"有效训练数据: {len(df_valid)} 条")
        
        # 准备训练数据
        texts = df_valid['comment'].tolist()
        quality_labels = df_valid['quality_score'].values.astype(np.float32)
        
        # 多标签编码
        category_labels = np.zeros((len(df_valid), len(self.CATEGORIES)))
        for i, cats in enumerate(df_valid['categories_list']):
            for cat in cats:
                if cat in self.CATEGORIES:
                    category_labels[i, self.CATEGORIES.index(cat)] = 1
        
        # 划分训练集/验证集
        X_train, X_val, y_q_train, y_q_val, y_c_train, y_c_val = train_test_split(
            texts, quality_labels, category_labels, test_size=0.2, random_state=42
        )
        
        # 创建 Dataset
        class CommentDataset(Dataset):
            def __init__(self, texts, quality_labels, category_labels, tokenizer, max_len=512):
                self.texts = texts
                self.quality_labels = quality_labels
                self.category_labels = category_labels
                self.tokenizer = tokenizer
                self.max_len = max_len
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'quality_label': torch.tensor(self.quality_labels[idx], dtype=torch.float32),
                    'category_label': torch.tensor(self.category_labels[idx], dtype=torch.float32)
                }
        
        train_dataset = CommentDataset(X_train, y_q_train, y_c_train, self.tokenizer)
        val_dataset = CommentDataset(X_val, y_q_val, y_c_val, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 优化器
        optimizer = optim.AdamW(
            list(self.quality_model.parameters()) + list(self.category_model.parameters()),
            lr=2e-5
        )
        
        quality_criterion = torch.nn.MSELoss()
        category_criterion = torch.nn.BCEWithLogitsLoss()
        
        # 训练循环
        best_val_loss = float('inf')
        Path(model_save_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch in range(5):  # 小数据集 3-5 轮即可
            print(f"\nEpoch {epoch+1}/5 开始...", flush=True)
            self.quality_model.train()
            self.category_model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                quality_label = batch['quality_label'].to(self.device)
                category_label = batch['category_label'].to(self.device)
                
                # 质量评分
                quality_output = self.quality_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                quality_loss = quality_criterion(
                    quality_output.logits.squeeze(), quality_label
                )
                
                # 类别分类
                category_output = self.category_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                category_loss = category_criterion(
                    category_output.logits, category_label
                )
                
                loss = quality_loss + category_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
            
            # 验证
            self.quality_model.eval()
            self.category_model.eval()
            val_q_loss, val_c_loss = 0, 0
            all_pred_q, all_true_q = [], []
            all_pred_c, all_true_c = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    quality_label = batch['quality_label'].to(self.device)
                    category_label = batch['category_label'].to(self.device)
                    
                    quality_output = self.quality_model(input_ids, attention_mask)
                    q_loss = quality_criterion(quality_output.logits.squeeze(), quality_label)
                    val_q_loss += q_loss.item()
                    
                    category_output = self.category_model(input_ids, attention_mask)
                    c_loss = category_criterion(category_output.logits, category_label)
                    val_c_loss += c_loss.item()
                    
                    all_pred_q.extend(quality_output.logits.squeeze().cpu().numpy())
                    all_true_q.extend(quality_label.cpu().numpy())
                    all_pred_c.extend(torch.sigmoid(category_output.logits).cpu().numpy())
                    all_true_c.extend(category_label.cpu().numpy())
            
            val_q_loss /= len(val_loader)
            val_c_loss /= len(val_loader)
            
            # 计算评估指标
            q_mae = mean_absolute_error(all_true_q, all_pred_q)
            # 类别指标 (micro average)
            pred_c_binary = (np.array(all_pred_c) > 0.5).astype(int)
            true_c_binary = np.array(all_true_c).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_c_binary, pred_c_binary, average='micro'
            )
            
            print(f"  Val: Q_Loss={val_q_loss:.4f}, Q_MAE={q_mae:.3f}, "
                  f"C_F1={f1:.4f}")
            
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                # 保存模型
                self.quality_model.save_pretrained(f"{model_save_dir}/quality_model")
                self.category_model.save_pretrained(f"{model_save_dir}/category_model")
                self.tokenizer.save_pretrained(f"{model_save_dir}")
                print(f"  模型已保存: {model_save_dir}")
        
        self.is_trained = True
        return {
            'quality_mae': q_mae,
            'category_f1': f1,
            'num_train_samples': len(X_train)
        }
    
    def _heuristic_predict(self, comment: str) -> Tuple[float, List[str]]:
        """启发式规则 fallback（无训练时使用）"""
        quality_score = 3.5  # 默认中等
        
        # 简单关键词评分
        positive_words = ['好', '不错', '满意', '推荐', '喜欢', '干净', '舒适', '热情']
        negative_words = ['差', '不好', '失望', '糟糕', '脏', '贵', '吵', '垃圾']
        
        pos_count = sum(1 for w in positive_words if w in comment)
        neg_count = sum(1 for w in negative_words if w in comment)
        
        if pos_count > neg_count:
            quality_score = 4.0 + min(0.5, pos_count * 0.1)
        elif neg_count > pos_count:
            quality_score = 2.5 - min(0.5, neg_count * 0.1)
        
        quality_score = np.clip(quality_score, 1, 5)
        
        # 类别检测
        categories = []
        category_keywords = {
            '整体满意度': ['满意', '体验', '推荐', '喜欢', '整体', '综合'],
            '前台服务': ['前台', '办理', '入住', '退房', '效率', '热情'],
            '客房服务': ['客房', '打扫', '清洁', '送物', '阿姨'],
            '退房/入住效率': ['入住', '退房', '办理', '等待', '排队'],
            '房间设施': ['房间', '设施', 'wifi', '空调', '冰箱', '电视', '床'],
            '公共设施': ['公共', '大堂', '电梯', '健身房', '泳池'],
            '餐饮设施': ['早餐', '餐厅', '自助', '吃', '食物', '餐饮'],
            '交通便利性': ['交通', '地铁', '公交', '打车', '便利'],
            '周边配套': ['周边', '逛街', '商场', '吃饭', '配套'],
            '卫生状况': ['卫生', '干净', '整洁', '异味', '脏'],
            '性价比': ['性价比', '价格', '便宜', '贵', '值']
        }
        
        for cat, keywords in category_keywords.items():
            if any(kw in comment.lower() for kw in keywords):
                categories.append(cat)
        
        if not categories:
            categories = ['设施']  # 默认
        
        return quality_score, categories
    
    def load_pretrained(self, model_dir: str):
        """加载预训练模型"""
        self.quality_model = AutoModelForSequenceClassification.from_pretrained(
            f"{model_dir}/quality_model"
        ).to(self.device)
        self.category_model = AutoModelForSequenceClassification.from_pretrained(
            f"{model_dir}/category_model"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.is_trained = True
        print(f"模型加载完成: {model_dir}")


# 评估函数
def evaluate_comment_analyzer(analyzer: CommentAnalyzer, test_csv_path: str) -> Dict:
    """评估分析器性能
    
    Returns:
        {
            'quality_mae': float,
            'quality_rmse': float,
            'quality_acc_within_1': float,  # 预测误差在1分内的比例
            'category_precision': float,
            'category_recall': float,
            'category_f1': float,
            'sample_count': int
        }
    """
    df = pd.read_csv(test_csv_path)
    
    # 解析实际标签
    def parse_categories(cat_str):
        if pd.isna(cat_str) or cat_str == '[]' or cat_str == '':
            return []
        import re
        cat_str = str(cat_str)
        # 格式1: "['服务','位置']"
        if cat_str.startswith('['):
            cats = re.findall(r"['\"]([^'\"]+)['\"]", cat_str)
            return cats if cats else []
        # 格式2: "服务,位置"
        else:
            return [c.strip() for c in cat_str.split(',') if c.strip()]
        
    # 取前100条用于快速评估
    test_df = df.head(100)
    
    predictions = []
    true_quality = []
    true_categories = []
    
    for _, row in test_df.iterrows():
        if pd.isna(row['comment']):
            continue
        pred_score, pred_cats = analyzer.predict(row['comment'])
        predictions.append((pred_score, pred_cats))
        true_quality.append(row['quality_score'] if pd.notna(row['quality_score']) else 3.0)
        true_categories.append(parse_categories(row['categories']))
    
    pred_quality = [p[0] for p in predictions]
    pred_categories = [p[1] for p in predictions]
    
    # 质量评分评估
    mae = mean_absolute_error(true_quality, pred_quality)
    rmse = np.sqrt(np.mean((np.array(true_quality) - np.array(pred_quality))**2))
    acc_within_1 = np.mean([abs(t - p) <= 1 for t, p in zip(true_quality, pred_quality)])
    
    # 类别评估 (多标签)
    from sklearn.preprocessing import MultiLabelBinarizer
    
    mlb = MultiLabelBinarizer(classes=analyzer.CATEGORIES)
    true_bin = mlb.fit_transform(true_categories)
    pred_bin = mlb.transform(pred_categories)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_bin, pred_bin, average='micro'
    )
    
    return {
        'quality_mae': mae,
        'quality_rmse': rmse,
        'quality_acc_within_1': acc_within_1,
        'category_precision': precision,
        'category_recall': recall,
        'category_f1': f1,
        'sample_count': len(predictions)
    }
    