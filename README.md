# 大模型与法律 LLM + RAG

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

## 简介

本项目是一个基于大模型的法律问答系统，LangChain 作为开发框架， DeepSeek 作为 LLM 大模型，智谱的 Embedding-3 作为 Embedding 模型。

该项目旨在展现 LangChain 和 RAG（Retrieval Augmented Generation） 的使用，可作为大模型应用的 HelloWorld 示例应用。

---

项目依赖：

- LangChain：构建大模型应用框架
- DeepSeek：深度求索大模型
- 智谱 AI：词嵌入 Embedding 模型
- Chroma：向量数据库


## TODO
- [x] 基于 LangChain 实现 LLM+RAG+法律对话系统
- [ ] 数据处理：格式化，删除空格、分隔符等
- [ ] 增量更新：支持增量更新知识库
- [ ] 开发交互式应用，如 Web 或 App
- [ ] 校验RAG效果

## 项目结构
```
.
├── README.md   # 项目说明文档
├── chroma      # 向量存储目录
├── law_data    # 知识库目录
├── main.py     # 主程序
├── requirements.txt # 依赖
├── vector.py   # 基于 Chroma 的向量类
└── zhipuai_embedding.py # 智谱 AI 的 Embedding 类
```

## 前置准备

克隆项目
```shell
git clone https://github.com/xiaolinstar/llm-law.git
# git clone git@github.com:xiaolinstar/llm-law.git
```

进入目录
```shell
cd llm-law
```

安装依赖

```shell
# 1. 创建 conda 环境
conda create -n llm-law python=3.10
# 2. 激活 conda 环境
conda activate llm-law
# 3. 安装项目依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

DeepSeek 和 智谱 AI 的 API 密钥需要在环境变量中配置。

一次性的：
```shell
export OPENAI_API_KEY="deepseek-api-xxx"
export ZHIPUAI_API_KEY="zhipu-api-xxx"
```

永久有效的：
```shell
echo "export OPENAI_API_KEY=\"deepseek-api-xxx\"" >> ~/.bashrc
echo "export ZHIPUAI_API_KEY=\"zhipu-api-xxx\"" >> ~/.bashrc
```

`.bashrc` 根据你的 shell 情况可以是 `.zshrc`或其他。

## 快速开始

运行主程序

```shell
python main.py
```

修改问题：`main.py` 中的 `questions` 变量，代码第 50 行

```
questions = ["根据未成年人保护法，在学校幼儿园和其他未成年人集中活动的公共场所吸烟饮酒应该受到什么处罚？",
             "2024年，某市某体育公司举办大型体育活动，有上千名群众参加，因为天气恶劣，主办方没有做好安全工作导致超过20名群众死亡，根据刑法，其主管人员犯了什么罪，将会受到何种刑事处罚，刑期如何？"]
```

## 自定义知识库
RAG 基于向量检索，需要有知识库。

知识库的目录为 `./law_data`，本项目代码中支持 PDF 和 Markdown 格式。

重要参数可在 `main.py` 中修改：

```
# 重要参数，可自行修改
# DeepSeek LLM
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# Vectordb
DATA_DIR = "./law_data"
PERSIST_DIR = "./chroma"
CHUNK_SIZE = 20
OVERLAP = 5
```

如果您在体验之后，进一步尝试：

1. 修改参数（CHUNK_SIZE 或 OVERLAP）
2. 更新知识库

进行以下操作即可：

1. 修改 `main.py` 中的参数
2. 更新 `./law_data` 目录下的文件
3. 删除 `./chroma` 目录下的所有文件

然后重新运行 `main.py` 即可。

## Reference

[1]. LangChain https://github.com/langchain-ai/langchain
[2]. 动手学习大模型 https://datawhalechina.github.io/llm-universe/#/ 
[3]. LangChain 自定义 智谱 AI Embedding类 https://github.com/datawhalechina/llm-universe/blob/main/notebook/C3%20%E6%90%AD%E5%BB%BA%E7%9F%A5%E8%AF%86%E5%BA%93/zhipuai_embedding.py
[4]. DeepSeek API 文档 https://api-docs.deepseek.com/zh-cn/
[5]. 智谱大模型API Keys https://bigmodel.cn/usercenter/proj-mgmt/apikeys
[6]. 面向法律场景的大模型 RAG 检索增强解决方案 https://developer.aliyun.com/article/1649317?spm=5176.2020520130.0.0.67bd697b9cRzjc&msctype=pmsg&mscareaid=cn&mscsiteid=cn&mscmsgid=7630125021100903556&yunge_info=pmsg___7630125021100903556&tlog=yuekan_2501
[7]. Anaconda https://anaconda.org/anaconda/conda

## 联系作者

1. 在issues中提问
2. 联系邮箱 :email: xing.xiaolin@foxmail.com

<!-- links -->

[contributors-shield]: https://img.shields.io/github/contributors/xiaolinstar/llm-law.svg?style=flat-square
[contributors-url]: https://github.com/xiaolinstar/llm-law/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/xiaolinstar/llm-law.svg?style=flat-square
[forks-url]: https://github.com/xiaolinstar/llm-law/network/members
[stars-shield]: https://img.shields.io/github/stars/xiaolinstar/llm-law.svg?style=flat-square
[stars-url]: https://github.com/xiaolinstar/llm-law/stargazers
[issues-shield]: https://img.shields.io/github/issues/xiaolinstar/llm-law.svg?style=flat-square
[issues-url]: https://github.com/xiaolinstar/llm-law/issues