# 多智能体旅行规划系统

基于 LangGraph 多智能体编排，融合本地攻略 RAG 与高德地图服务的智能旅行规划系统。


## ✨ 功能特点

- 🤖 **AI驱动的旅行规划**: 基于LangGraph框架的多智能体工作流,智能生成详细的多日旅程
- 🗺️ **高德地图集成**: 通过MCP协议接入高德地图服务,支持景点搜索、路线规划、天气查询
- 🧠 **智能工具调用**: Agent自动调用高德地图MCP工具,获取实时POI、路线和天气信息
- 📚 **本地攻略 RAG**: 对 Markdown 旅行攻略做切分、查询改写、向量召回和重排，再注入行程 Planner
- 🎨 **现代化前端**: Vue3 + TypeScript + Vite,响应式设计,流畅的用户体验
- 📱 **完整功能**: 包含住宿、交通、餐饮和景点游览时间推荐

## 🏗️ 技术栈

### 后端
- **框架**: LangGraph (基于StateGraph的多智能体工作流)
- **API**: FastAPI
- **MCP工具**: amap-mcp-server (高德地图)
- **RAG**: Markdown 切分、Query Rewrite、向量召回、噪声预过滤、规则重排/可选 qwen3-rerank、缓存、关键词兜底
- **LLM**: 支持多种LLM提供商(OpenAI, DeepSeek等)

### 前端
- **框架**: Vue 3 + TypeScript
- **构建工具**: Vite
- **UI组件库**: Ant Design Vue
- **地图服务**: 高德地图 JavaScript API
- **HTTP客户端**: Axios

## 📁 项目结构

```
langgraph-multi-agent-trip-rag/
├── backend/                    # 后端服务
│   ├── app/
│   │   ├── workflows/         # LangGraph工作流实现
│   │   │   ├── trip_planner_graph.py
│   │   │   ├── trip_planner_state.py
│   │   │   └── train.py
│   │   ├── rag/               # RAG 检索链路
│   │   │   ├── vector_db.py
│   │   │   ├── retriever.py
│   │   │   └── fallback_candidates.py
│   │   ├── agents/            # LangChain智能体定义
│   │   │   ├── langgraph_agents.py
│   │   │   ├── old_helloagent_planner_agent.py
│   │   │   └── __init__.py
│   │   ├── api/               # FastAPI路由
│   │   │   ├── main.py
│   │   │   └── routes/
│   │   │       ├── trip.py
│   │   │       ├── map.py
│   │   │       └── poi.py
│   │   ├── services/          # 服务层
│   │   │   ├── amap_service.py
│   │   │   ├── llm_service.py
│   │   │   └── unsplash_service.py
│   │   ├── models/            # 数据模型
│   │   │   └── schemas.py
│   │   ├── tools/             # 工具定义
│   │   │   ├── amap_mcp_tools.py
│   │   │   └── __init__.py
│   │   └── config.py          # 配置管理
│   ├── data/                  # 本地旅行攻略 Markdown
│   ├── scripts/               # RAG 入库与调试脚本
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
├── frontend/                   # 前端应用
│   ├── src/
│   │   ├── components/        # Vue组件
│   │   ├── services/          # API服务
│   │   ├── types/             # TypeScript类型
│   │   └── views/             # 页面视图
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 🚀 快速开始

### 前提条件

- Python 3.10+
- Node.js 18+
- 高德地图API密钥 (Web服务API和Web端(JS API))
- LLM API密钥 (OpenAI/DeepSeek等)

### 后端安装

1. 进入后端目录
```bash
cd backend
```

2. 创建虚拟环境
```bash
python -m venv venv
venv\Scripts\activate
# Mac: source venv/bin/activate 
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件,填入你的API密钥
```

5. 启动后端服务
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

```

### 前端安装

1. 进入前端目录
```bash
cd frontend
```

2. 安装依赖
```bash
npm install
```

3. 配置环境变量
```bash
# 创建.env文件, 填入高德地图Web API Key 和 Web端JS API Key
cp .env.example .env
```

4. 启动开发服务器
```bash
npm run dev
```

5. 打开浏览器访问 `http://localhost:5173`

## 📝 使用指南

1. 在首页填写旅行信息:
   - 目的地城市
   - 旅行日期和天数
   - 交通方式偏好
   - 住宿偏好
   - 旅行风格标签

2. 点击"生成旅行计划"按钮

3. 系统将:
   - 调用LangGraph工作流生成初步计划
   - 并行检索本地旅行攻略 RAG 上下文
   - Agent自动调用高德地图MCP工具搜索景点
   - Agent获取天气信息和路线规划
   - 整合所有信息生成完整行程

4. 查看结果:
   - 每日详细行程
   - 景点信息与地图标记
   - 交通路线规划
   - 天气预报
   - 餐饮推荐

## 🔧 核心实现

### RAG 检索链路

旅行请求进入 LangGraph 后，会新增一条 `retrieve_rag_context` 分支，与景点、天气、酒店三路并行执行：

1. 用规则或 LLM 把目的地、偏好、旅行风格和额外要求改写成检索 query。
2. 对 `backend/data/*.md` 攻略按标题切分。
3. 优先使用 ChromaDB 向量召回；未安装 ChromaDB 时使用内置本地 n-gram 向量索引。
4. 先做噪声预过滤，再对候选片段做目的地过滤和轻量重排；配置 `RERANK_PROVIDER=dashscope` 后可选 qwen3-rerank 精排。
5. 把命中的攻略片段注入 Planner 提示词，并限制模型优先采用攻略中的真实景点、餐厅和酒店名称。

默认不需要外部 embedding API。安装 ChromaDB 后会自动升级为 MiniLM 本地 embedding 向量库；也可以把 `EMBEDDING_PROVIDER` 改为 `openai`，使用兼容 OpenAI 的 embedding 接口。

如需显式构建 ChromaDB，或调试某城市的 RAG 召回：

```bash
cd backend
python scripts/ingest_data.py
python scripts/debug_rag_retrieval.py 北京
```

### LangGraph工作流集成

```python
from langgraph.graph import StateGraph, END
from app.workflows.trip_planner_graph import TripPlannerWorkflow
from app.models.schemas import TripRequest

# 创建旅行规划工作流
workflow = TripPlannerWorkflow()

# 创建旅行请求
request = TripRequest(
    city="北京",
    travel_days=3,
    transportation="地铁",
    accommodation="经济型酒店",
    preferences=["历史文化", "公园"],
    start_date="2024-10-01",
    end_date="2024-10-03",
    free_text_input="希望行程轻松一些"
)

# 执行工作流生成旅行计划
trip_plan = workflow.plan_trip(request)
print(f"生成 {len(trip_plan.days)} 天行程计划")
```

### MCP工具调用

工作流中的智能体可以自动调用以下高德地图MCP工具:
- `maps_text_search`: 搜索景点POI
- `maps_weather`: 查询天气
- `maps_direction_walking_by_address`: 步行路线规划
- `maps_direction_driving_by_address`: 驾车路线规划
- `maps_direction_transit_integrated_by_address`: 公共交通路线规划

## 📄 API文档

启动后端服务后,访问 `http://localhost:8000/docs` 查看完整的API文档。

主要端点:
- `POST /api/trip/plan` - 生成旅行计划
- `GET /api/map/poi` - 搜索POI
- `GET /api/map/weather` - 查询天气
- `POST /api/map/route` - 规划路线

## 🤝 贡献指南

欢迎提交Pull Request或Issue!

## 📜 开源协议

CC BY-NC-SA 4.0

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - 多智能体工作流框架
- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用开发框架
- [高德地图开放平台](https://lbs.amap.com/) - 地图服务
- [amap-mcp-server](https://github.com/sugarforever/amap-mcp-server) - 高德地图MCP服务器

---

**多智能体旅行规划系统** - 让旅行计划变得简单而智能

