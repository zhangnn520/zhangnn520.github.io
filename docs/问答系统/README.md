---
sort: 
---

# 对话系统

问答系统大概划分为5个类型，主要根据任务形式和知识库里数据的存储结构。

(1) **FAQ-Bot**: 基于常见问答对的问答，这也是运用最为广泛的[智能问答技术](https://www.zhihu.com/search?q=智能问答技术&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2736787407})。抽象出来是一个[信息检索](https://www.zhihu.com/search?q=信息检索&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2736787407})的问题，给定用户的问题，在由{问答：答案}组成的知识库中检索相似的问题，最后将与用户相似问法问题的答案作为结果返回给用户。

(2) **MRC-Bot**: 基于机器阅读的智能问答，一般运用在[开放域](https://www.zhihu.com/search?q=开放域&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2736787407})的问答中。给定用户的问题，具体分成召回和机器阅读两个阶段，先从知识库中检索出可能存在答案的文档，再针对文档做机器阅读确定答案。在实际落地中也很有前景，相比FAQ-Bot用户不需要耗费很大力气构建知识库，只需要上传产品文档即可。但是目前机器阅读的准确性还不够，效果不稳定，还不能直接将机器阅读的结果作为答案返回给用户。

(3)**KG-Bot**: 基于知识图谱的问答，一般用于解答属性型的问题，比如“北京的市长是谁”。给定用户的问题，需要先解析成知识图谱查询语句，再到知识图谱中检索答案。这种问答一般回答的准确率非常高，但是能回答的问题也非常局限，同时构建知识图谱非常耗费人力。

(4)**Task-Bot**: 任务型对话，是面向特定场景的多轮对话，比如“查天气”，“订机票”。"Task oriented dialogue"在学术和工业界都已经有了很深入的研究，分成[pipeline](https://www.zhihu.com/search?q=pipeline&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2736787407})和end-to-end两种思路。在实地落地过程中，难得是如何让用户自主的灵活配置一个任务型对话场景，训练语料可能只有一两条，如何让模型能学到这个槽位？

(5)**Chat-Bot**: 闲聊对话，一般用于提高机器人的趣味性，比如“你是谁？”，“你是机器人吗？”等。在学术上一般基于end-to-end的方案，可以支持多轮，但是回复结果不可控。所以在实际落地中还是会转换成FAQ-Bot，预先构建一个寒暄库，转换成检索的任务。

| 机器人类型 | 知识库结构                                                   | 核心技术          | 落地难度 |
| ---------- | ------------------------------------------------------------ | ----------------- | -------- |
| FAQ-Bot    | {问题:答案}                                                  | 信息检索          | 低       |
| MRC-Bot    | 文档                                                         | 信息检索+机器阅读 | 中       |
| KG-Bot     | [知识三元组](https://www.zhihu.com/search?q=知识三元组&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2736787407}) | 知识图谱构建/检索 | 高       |
| Task-Bot   | 槽位/对话策略                                                | 对话状态跟踪/管理 | 高       |
| Chat-Bot   | {寒暄语:回复}                                                | 信息检索          | 低       |

一般作为一个商业化的智能问答系统一般上面的各种bot都会有，通过中控来做类型识别和分发。

链接：https://www.zhihu.com/question/555696715/answer/2736787407