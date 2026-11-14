---
title: Harness 的来源与大模型的本质缺陷
date: 2026-11-14
tags:
  - LLM
  - Harness
layout: post
---

# Harness 的来源与大模型的本质缺陷

### 1.大模型的本质缺陷:

上下文有限；幻觉；依赖于训练数据的分布；安全性问题；无确定性能力；慢；无法对齐人类的多样性；

对应的解决方式是：压缩；重试；校验工具；确定性工具或格式约束；性能优化；

依赖于训练数据的分布分为：域外数据无法处理，即不可能提前知道业务知识；自然语言不可能完全无歧义；

### 2.**被包装成Harness的技术：**


| 模块                | 常见技术                                  |
| ------------------- | ----------------------------------------- |
| Agent Loop          | Python / TypeScript、AsyncIO、状态机      |
| Tool Runtime        | Function Calling、MCP、API、Shell         |
| Context Engineering | Prompt Template、RAG、Context Compression |
| Memory & State      | Redis、PostgreSQL、向量数据库、Checkpoint |
| Guardrails          | RBAC、Sandbox、Docker、Policy Engine      |
| Evaluation          | pytest、LLM-as-a-Judge、自动化测试        |

#### 2.1 上下文有限：

RAG；记忆机制；AGENT\.md约束；Dreaming机制（Anthropic）

#### 2.2 幻觉：

使用确定性工具：用编译器、linter、类型检查、单元测试，sandbox等；

#### 2.3 安全性：

权限确认、白名单、二次确认；沙箱执行；

其他：训练的时候提前加入包括格式对齐、价值观对齐、工具安全知识的对齐，降低不安全输出的概率，对于不安全操作的判断，但是这个不是Harness的内容；

#### 2.4 不确定性输出：

结构化输出；JSON Schema；

MCP：作为中转服务，通过转换自然语言为格式化数据，降低不确定性

#### 2.5 依赖于训练数据的分布：

MCP：由于对私域数据和私有工具的一无所知，需要用MCP进行

#### 2.6 模型的固化问题（不可测试时训练）：

Skill：项目的隐形规定、最佳实践、踩过的坑，导致经验无法复用，所以需要有Skill方便地引入经验；（模型不可能在没有环境信息的情况下做出完美决策）；

对于非常专业的没有见过的任务，必须增加数据进行训练；

#### 2.7 效率低下：

Skill：本质上是降低模型的多余输出，减少无谓尝试；

Sub\-Agent：提高速度；



还有一个基本的问题，大模型只会吐token，而负责整个系统的循环，即发起请求、接收调用、真正去执行、处理失败、把结果回填、再次请求。这个循环的机械部分才是 Harness 的。

而 Evaluation 是在大模型无法完全跟不同人类偏好对齐的前提下产生的问题；

### 3.更多更难任务下的问题

无限多的任务和无限多的领域知识；

无限大的scope，超出本身可能获知知识的领域（如隐藏偏好、人类知识）；

跟真实世界的交互；

由于慢必须加速；





### 4.当前所谓的前沿方案：

1.来自Anthropic，已经极大地解决了Prompt注入的问题；

2.deepseek-harness: 一些工程上的小改进；都是把工程上的东西具象化了。仅仅针对上下文有限，就给出了自动压缩、子 Agent 隔离、Ralph 循环、大结果落盘、Skills 按需加载、跨会话记忆（需要通过插件如Mem0、Honcho）的各种手段。针对慢的问题，给出了PTC模式、异步执行提升了调用效率；可回放提升了用户体验；Agent Teams增加了并行；但是本质上没有解决任何重要的问题。

3.pi-agent也是大同小异，解决思路跟上面差不多；这里贴一下deepseek-harness和pi-agent的对比

| 大模型缺陷       | Pi Agent                                         | dsh                                                |
| ---------------- | ------------------------------------------------ | -------------------------------------------------- |
| 上下文有限       | 自动压缩、Skills、RAG、会话分支                  | 自动压缩、子 Agent 隔离、Ralph 循环、大结果落盘    |
| 幻觉             | 工具读取、代码测试、MCP、结果验证                | 工具查证、测试、搜索、Hooks 复核                   |
| 训练数据分布依赖 | 多模型、MCP、RAG、Skills；**不解决底层泛化能力** | 多模型、子 Agent、外部知识；**不解决底层泛化能力** |
| 安全性问题       | 扩展权限控制、外部沙箱；核心安全隔离较弱         | 权限审批、四种沙箱后端、最小权限                   |
| 无确定性能力     | Codemode、程序化工具调用、Extensions             | PTC、Workflow、状态机、结构化输出                  |
| 慢               | Codemode、并发调用、多模型选择                   | PTC、并行执行、后台任务、辅助模型路由              |
| 人类多样性       | Extensions、Skills、提示词、模型定制             | Profile、Preset、Patch、Hooks、运行时纠偏          |



从根本上讲，目前的agent最严峻的问题在于大模型的确定性不足、幻觉问题、低效率与输出不确定性，需要程序兜底。而从根本上幻觉问题不可解决，低效率与边际成本不为0，所以Agent必然依赖于确定性程序提供稳定性保障。


#### References

When to build an agent https://cdn\.openai\.com/business\-guides\-and\-resources/a\-practical\-guide\-to\-building\-agents\.pdf

Pi\-Agent https://pi\.dev/

Deer\-Flow https://github\.com/bytedance/deer\-flow

Sandbox https://github.com/agent-infra/sandbox/

DeepSeek-Harness https://www.datacamp.com/blog/what-is-deepseek-harness

Smolagents 入门级Agent https://github\.com/huggingface/smolagents

