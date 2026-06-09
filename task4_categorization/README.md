# Task 4: Review Category Mining and Multi-label Categorization

本目录对应方向 4：类别划分是否可以通过数据挖掘方式实现，而不是完全依赖人工预定义类别；以及如何处理一条评论归属多个类别的问题。

## 目录结构

```text
tfidf_kmeans.ipynb
bert_embedding_topics.ipynb
bert_kmeans_multilabel.ipynb
tfidf_result/
bert_kmeans_result/
bert_bertopic_result/
```

## 1. TF-IDF + KMeans

Notebook：

```text
tfidf_kmeans.ipynb
```

结果目录：

```text
tfidf_result/
```

主要输出：

```text
guangzhou_garden_hotel_tfidf_kmeans_clusters.csv
guangzhou_garden_hotel_tfidf_kmeans_cluster_keywords.csv
guangzhou_garden_hotel_tfidf_kmeans_representatives.csv
```

说明：

- 使用 TF-IDF 表示评论文本；
- 使用 KMeans 进行无监督聚类；
- 输出每条评论的聚类结果、每个聚类的关键词和代表性评论；
- 本实验中 TF-IDF KMeans 使用 `k=13`。

结论：

- TF-IDF + KMeans 可作为可解释的传统文本聚类基线；
- 关键词结果直观，便于理解每个簇的大致主题；
- 缺点是容易受到表层词汇影响，同义表达或语义相近内容可能被拆到不同簇。

## 2. BERT Embedding + KMeans

Notebook：

```text
bert_embedding_topics.ipynb
```

结果目录：

```text
bert_kmeans_result/
```

主要输出：

```text
guangzhou_garden_hotel_bert_kmeans_clusters.csv
guangzhou_garden_hotel_bert_kmeans_metrics.csv
guangzhou_garden_hotel_bert_kmeans_representatives.csv
```

说明：

- 使用 BERT / SentenceTransformer 生成评论语义向量；
- 使用 KMeans 在语义向量空间中聚类；
- 使用聚类指标和可解释性共同选择类别数；
- 本实验中 BERT KMeans 使用 `k=5`。

聚类类别包括：

```text
花园特色、文化底蕴与空间体验
餐饮早餐、景观与基础住宿体验
前台服务、舒适住宿与升级好评
老牌五星综合长评体验
问题反馈与负面复杂体验
```

结论：

- BERT KMeans 是本方向的主要推荐方案；
- 相比 TF-IDF，它更能捕捉语义相似性，减少同义表达带来的类别碎片化；
- 它得到的是“语义场景类别”，与原有人工业务类别并不完全相同。

## 3. BERT KMeans Multi-label Extension

Notebook：

```text
bert_kmeans_multilabel.ipynb
```

结果目录：

```text
bert_kmeans_result/
```

主要输出：

```text
guangzhou_garden_hotel_bert_kmeans_multilabel.csv
guangzhou_garden_hotel_bert_kmeans_multilabel_summary.csv
```

说明：

KMeans 默认只能给每条评论分配一个主类别。为了解决“一条评论归属多个类别”的问题，本实验基于 BERT 向量和聚类中心相似度进行多标签扩展：

1. 保留 KMeans 的主簇作为 `primary_category`；
2. 计算每条评论向量与所有聚类中心的 cosine similarity；
3. 选取除主簇外相似度最高的两个簇作为 secondary categories；
4. 输出 top-3 类别、top-3 相似度，以及主类别和第一副类别之间的相似度差距。

关键字段：

```text
primary_category
secondary_category_1
secondary_category_2
primary_similarity
secondary_similarity_1
secondary_similarity_2
primary_secondary_gap_1
top3_categories
top3_similarities
```

结论：

- 酒店评论天然包含多个关注点，如服务、房间、早餐、交通、价格等；
- 通过 Top-N 聚类中心相似度，可以避免把一条评论强行压到单一类别；
- `primary_secondary_gap_1` 较小说明很多评论在多个语义类别之间都有较高相关性，支持多标签处理的必要性。

## 4. BERT Embedding + BERTopic

Notebook：

```text
bert_embedding_topics.ipynb
```

结果目录：

```text
bert_bertopic_result/
```

主要输出：

```text
guangzhou_garden_hotel_bertopic_topics.csv
guangzhou_garden_hotel_bertopic_topic_info.csv
guangzhou_garden_hotel_bertopic_keywords.csv
guangzhou_garden_hotel_bertopic_representatives.csv
```

说明：

- 使用 BERTopic 进行细粒度主题发现；
- 输出每条评论的 topic、topic 信息、关键词和代表性评论；
- 本实验结果包含 18 个正式 topic 和一个离群 topic `-1`。

结论：

- BERTopic 适合发现更细粒度的主题，例如节日活动、具体员工服务、会员权益、房型套餐等；
- 它可以作为 BERT KMeans 的补充，用于发现更细的业务洞察；
- 但当前结果中离群 topic 和大主题占比较高，因此不作为主类别体系。

## 5. 总体结论

本方向最终建议：

1. **TF-IDF + KMeans** 作为传统可解释基线；
2. **BERT Embedding + KMeans** 作为主要的无监督类别挖掘方案；
3. **BERT KMeans Top-N 相似度** 用于解决一条评论归属多个类别的问题；
4. **BERTopic** 作为细粒度主题发现补充。

与原有“人工预定义类别 + LLM 分类”相比，聚类方法并不是简单替代关系：

- 原有人工类别更偏业务维度，稳定且可控；
- 无监督聚类更偏数据驱动，能发现自然语义主题；
- 两者可以结合使用：人工类别用于业务口径，聚类结果用于发现新主题和辅助优化类别体系。

## 6. 注意事项

本目录只包含 notebook 和实验结果文件，不包含：

- 原始大体量 embedding `.npy` 文件；
- BERTopic 模型目录；
- 原始评论数据文件。

如需复现实验，请将项目数据文件放到 notebook 中配置的数据路径下，并根据服务器环境安装对应依赖。
