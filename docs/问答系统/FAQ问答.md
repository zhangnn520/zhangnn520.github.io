# FAQ问答资料汇总

# **一、专栏推荐**

## 1.1、[如何搭建基于文本语义的智能问答系统？](https://www.zhihu.com/question/555696715/answer/2736787407)

问答系统大概划分为5个类型：FAQ-Bot、MRC-Bot、KG-Bot、Task-Bot和闲聊。

| 机器人类型 | 知识库结构                                                   | 核心技术          | 落地难度 |
| ---------- | ------------------------------------------------------------ | ----------------- | -------- |
| FAQ-Bot    | {问题:答案}                                                  | 信息检索          | 低       |
| MRC-Bot    | 文档                                                         | 信息检索+机器阅读 | 中       |
| KG-Bot     | [知识三元组](https://www.zhihu.com/search?q=知识三元组&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2736787407}) | 知识图谱构建/检索 | 高       |
| Task-Bot   | 槽位/对话策略                                                | 对话状态跟踪/管理 | 高       |
| Chat-Bot   | {寒暄语:回复}                                                | 信息检索          | 低       |

​		在上述五种问答系统中比较常用且落地难度较小的就是FAQ。而要搭建FAQ-Bot最快的方式就是通过ES库来构建，基于ES可以快速构建检索型的智能问答系统，包括“输入联想”，“相似问题检索排序”，“拼音/首字母混合检索”等常见功能。传统的ES仅支持“字面”匹配（BM25算法），最新的ES也已经支持“语义”匹配，所以可以通过深度学习模型提取问题的语义特征（例如sentence-bert），然后存入ES中。这样用户的query就可以与问题库进行“字面”匹配+“语义”匹配了。

![img](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\faq_es.png)



### 1.1.1、[基于FAQ智能问答之es的调教](https://zhuanlan.zhihu.com/p/347957917)

​		文章主要讲解了es的安装和使用方法，文章中描述mysql与es数据集需要同时发生变化时，可以使用阿里的[canal](https://github.com/alibaba/canal)。文章也介绍了es的检索方式，支持自定义词典等。

![img](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\cancal.png)

### 1.1.2、[基于FAQ智能问答之召回篇](https://zhuanlan.zhihu.com/p/349993294)  

​		本文主要介绍了es召回，语义召回和双路召回三种方式。

​		（1）基于es召回。它本质就是基于BM25的召回，也是基于字面的关键词进行召回，但不是语义的召回。无法解决以下句子的检索。[es操作说明书](https://www.elastic.co/guide/en/elasticsearch/reference/7.16/index-modules-similarity.html)

​			知识库内问题：“可以免运费吗？”

​			用户问题：“你们还包邮？”

​		这两个问题没有一个关键词相同，但是其实是语义一致的。这就需要基于语义的召回来补充关键词召回。

​		（2）基于语义召回。它通常是基于embedding的召回。操作步骤如下：首先训练sentence embedding模型，然后将知识库中的问题都预先计算出embedding向量。在线上预测阶段，对于每个query同样先计算出embedding，再到知识库中检索出相近的embedding所属的问题。

​		（3）双路召回。字面召回 + 语义召回” 的双路召回结构，如下图。

![img](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\双路召回.png)

### 1.1.3、[基于FAQ智能问答之精排篇](https://zhuanlan.zhihu.com/p/352316559)

​			给定一个用户的问题q，和一批召回的问题d，对每个d进行相关性的打分计算，并根据相关性进行评价。同时根据top1打分的不同，客户端执行不同的操作：(1) 如果top1的评分"很高"，则直接将答案返回给用户 (2) 如果top1的评分"较高"，则进行问题的推荐 (3) 如果top1的评分"较低"，则提示无法回答。



# 二、开源项目

## 2.1、paddle [多路召回项目](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/semantic-search/Multi_Recall.md)

本项目提供了低成本搭建端到端两路召回语义检索系统的能力。用户只需要处理好自己的业务数据，就可以使用本项目预置的两路召回语义检索系统模型(召回模型、排序模型)快速搭建一个针对自己业务数据的检索系统，并可以提供 Web 化产品服务。

![img](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\paddle双路召回.png)

![img](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\paddlnlp多路召回与语义检索对比.png)



## 2.2、[政务问答系统](https://github.com/PaddlePaddle/PaddleNLP/tree/635b272b64485640f4a5e3fe6c79b0846abc6c84/applications/question_answering/supervised_qa/faq_system)

### 2.2.1、系统架构

![image-20230807142843236](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\政务faq系统架构.png)

## 2.2.2、系统特色

- 低门槛
  - 手把手搭建检索式 FAQ System
  - 无需相似 Query-Query Pair 标注数据也能构建 FAQ System
- 效果好
  - 业界领先的检索预训练模型: RocketQA Dual Encoder
  - 针对无标注数据场景的领先解决方案: 检索预训练模型 + 增强的无监督语义索引微调
- 性能快
  1. 基于 Paddle Inference 快速抽取向量
  2. 基于 Milvus 快速查询和高性能建库
  3. 基于 Paddle Serving 高性能部署

### 2.2.3、效果评估

| 模型                   | Recall@1   | Recall@10  |
| ---------------------- | ---------- | ---------- |
| ERNIE1.0 + SimCSE      | 68.068     | 85.686     |
| RocketQA               | 81.381     | 96.997     |
| RocketQA + SimCSE      | 83.283     | 97.297     |
| RocketQA + SimCSE + WR | **83.584** | **97.497** |



## 2.3、[金融问答系统](https://github.com/PaddlePaddle/PaddleNLP/tree/635b272b64485640f4a5e3fe6c79b0846abc6c84/applications/question_answering/supervised_qa/faq_finance)

### 2.3.1、配套视频链接

​			https://aistudio.baidu.com/aistudio/projectdetail/3882519

### 2.3.2、问答流程

![img](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\金融问答系统)

​		如上图所示，问答的流程分为两部分，第一部分是管理员/工程师流程，第二部分就是用户使用流程，在模型的层面，需要离线的准备数据集，训练模型，然后把训练好的模型部署上线。另外，就是线上搭建问答检索引擎，第一步把收集好的语料数据，利用训练好的模型抽取问题的向量，然后把向量插入到近似向量检索引擎中，构建语义索引库，这部分做完了之后，就可以使用这个问答服务了，但是用户输入了Query之后，发生了什么呢？第一步就是线上服务会接收Query后，对数据进行处理，并抽取用户Query的向量，然后在ANN查询模块进行检索匹配相近的问题，最终选取Top10条数据，返回给线上服务，线上服务经过一定的处理，把最终的答案呈现给用户。

### 2.3.3、模型优化流程

![img](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\金融问答优化流程.png)

### 2.3.4、模型 WR 策略

|      策略      |             举例             |     解释     |
| :------------: | :--------------------------: | :----------: |
|      原句      |   企业养老保险自己怎么办理   |      -       |
| WR策略（Yes）  | 企业养老老保险自己怎么么办理 | 语义改变较小 |
|  随机插入(No)  |  无企业养老保险自己怎么办理  | 语义改变较大 |
| 随机删除（No） |    企业养保险自己怎么办理    | 语义改变较大 |

​		上表是WR策略跟其他策略的简单比较，其中WR策略对原句的语义改变很小，但是改变了句子的长度，破除了SimCSE句子长度相等的假设。WR策略起源于ESimCSE的论文，有兴趣可以从论文里了解其原理。

### 2.4、[多场景对话机器人](https://github.com/charlesXu86/Chatbot_CN)

![](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\Chatbot_CN00.png)

### ![](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\wechatter_1130.png)



### 2.5、[FAQ智能问答系统](https://github.com/wzzzd/FAQ_system)

### 2.5.1、知乎文章

​	https://zhuanlan.zhihu.com/p/602337390

### 2.5.2、系统框架

​		系统支持有监督和无监督语义表征训练，也同样支持精排和粗排召回。上述功能均可在配置文件中进行配置。但是目前系统最大的缺点就是系统推理速度很慢，即使使用gpu效果依旧很慢。

![image-20230807144212020](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\FAQ智能问答系统架构.png)

### 2.5.3、查询流程

```text
输入query文本 -> 分词 -> 召回（ES） -> 粗序（PreRank） -> 精排（Rank） -> result
```



### 2.5.4、性能指标

​		在100个样本的测试集中，分别统计top1、top3、top10的召回结果准确率。

![img](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\召回指标.png)

## 2.6、[简易问答系统demo:faq-qa-sys](https://github.com/lerry-lee/faq-qa-sys)

一个简单的FAQ问答系统实现。基于检索和排序的两阶段框架，检索阶段基于Elasticsearch检索引擎、排序阶段基于语义匹配深度学习模型。后端基于SpringBoot系列框架。

### 2.6.1、系统示意图

![](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\faq-qa-sys流程.png)

### 2.6.2、系统架构图

![](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\faq-qa-sys.png)

### 2.6.3、对话流程图

![](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\对话流程.png)

### 2.6.4、多轮对话设计

![](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\多轮树.png)





## 2.7、BEFAQ

​		BEFAQ(BERT-based Embedding Frequently Asked Question)** 开源项目是好好住面向多领域FAQ集合的问答系统框架。我们将Sentence BERT模型应用到FAQ问答系统中。开发者可以使用BEFAQ系统快速构建和定制适用于特定业务场景的FAQ问答系统。

### 2.7.1、BEFAQ优点


（1）使用了Elasticsearch、Faiss、Annoy 作为召回引擎

（2）使用了Sentence BERT 语义向量（Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks）

（3）对同义问题有很好的支持

（4）支持多领域语料（保证了召回的数据是对应领域的，即使是同样的问题，也可以得到不同的答案。）

（5）提供了根据当前输入提示联想问题（suggest）功能的接口



2.7.2系统框架

![BEFAQ 框架](C:\Users\Administrator\Downloads\zhangnn520.github.io\imgs\BEFAQ 框架.png)

## 三、经典算法





## 四、字面和向量库：



### 4.1、miluvs

​	（1）[milvus_v1.1.1-cpu使用说明书](https://milvus.io/docs/v1.1.1/milvus_docker-cpu.md)

​	（2）[milvus开源的数据库](https://mp.weixin.qq.com/s?__biz=MzAwOTU4NzM5Ng==&mid=2455772667&idx=1&sn=ead133d70c53c2dbe1483921c1597cc9&chksm=8cc9e250bbbe6b460ccc5a9faa5274ddf48861efd5484fd5b6445661c9fe06ad632732b463c2&scene=178&cur_album_id=3006183510551822339#rd)

### 4.2、faiss

​	（1）[faiss开源的向量索引库](https://mp.weixin.qq.com/s?__biz=MzAwOTU4NzM5Ng==&mid=2455772648&idx=1&sn=6bfb1c6ee9b7ca5e21fc76583e5d4680&chksm=8cc9e243bbbe6b55b4c207d997bafe75f364aa21cbcf4f7efb09ba39460441dbcc67e2e224f3&scene=178&cur_album_id=3006183510551822339#rd)

​	（2）

### 4.3、elasticsearch

​	(1）[使用向量字段进行文本相似度搜索](https://www.elastic.co/cn/blog/text-similarity-search-with-vectors-in-elasticsearch)

## 五、数据清洗：

​       (1) 去除重复项。检查数据集中是否有重复的问题或答案，如果有，只保留一份。

​       (2) 格式一致。确保所有的问题和答案都是以相同的格式储存的，规范数据集格式一致。

​      (3) 去除无关项。如果数据集中包含了与你的项目无关的问题或答案，你应该将它们去除。

​      (4) 语言清洗。去除文本中的拼写错误，语法错误，使用统一的词汇和表达方式。

​     (5) 数据填充。如果某些重要的问题或答案缺失了，你可能需要填充它们。你可以使用一些预设的值，或者是使用一些算法来预测它们。

​     (6) 数据分割。对于FAQ数据集，可能需要将问题和答案分开，存储在不同的列或者数据结构中。

## 六、数据集

（1）[m3e数据集链接](https://pan.baidu.com/s/1KHOWZ7OM9_BrWFyVT7c6xg?pwd=x609 ) 

（2）[FAQ数据集调研](https://zhuanlan.zhihu.com/p/83211462)



## 七、技术方案：

​	（1）第一版faq就用paddle的simcse+milvus+rocketqa_crossencoder，后续上线版本主要对粗排进行处理，增加bm25等，不能用单一的召回方式。
​	（2）第二版采用m3e+milvius2.x+多路召回+rocketqa或许其他精排方法，可以尝试一下es数据库，也可以对召回的问答对基于multilinggual-e5-large用于检索qa的相关性提高召回的精准度。
​	（3）第三版就是llm+faq，第三版应为测试和研究版本，用于对第一版或第二版进行词向量部分进行改造，使用大模型的词向量进行处理。



## 八、主要参考资料：

​	1、[如何搭建基于文本语义的智能问答系统？](https://www.zhihu.com/question/555696715/answer/2736787407)

​	2、https://huggingface.co/moka-ai/m3e-large

​	3、https://milvus.io/docs/v1.1.1/milvus_docker-cpu.md

​	4、https://www.zhihu.com/question/555696715/answer/2736787407

​	5、https://zhuanlan.zhihu.com/p/347957917

​	6、https://zhuanlan.zhihu.com/p/352316559

​	7、https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/semantic-search/Multi_Recall.md

​	8、[paddle-nlp faq_system](https://github.com/PaddlePaddle/PaddleNLP/tree/635b272b64485640f4a5e3fe6c79b0846abc6c84/applications/question_answering/supervised_qa/faq_system)

​	9、[paddle-nlp _faq_finance](https://github.com/PaddlePaddle/PaddleNLP/tree/635b272b64485640f4a5e3fe6c79b0846abc6c84/applications/question_answering/supervised_qa/faq_finance)

​	10、https://github.com/wzzzd/FAQ_system

​	11、https://github.com/lerry-lee/faq-qa-sys

​	12、https://www.elastic.co/cn/blog/text-similarity-search-with-vectors-in-elasticsearch