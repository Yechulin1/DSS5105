# 快捷提问功能说明

## 功能概述
在Q&A界面添加了快捷提问气泡，用户可以一键点击快速提问，根据用户身份（租客/房东）显示不同的问题。

## 功能特点

### 1. **智能身份识别**
- 系统自动识别用户身份（租客/房东）
- 根据身份显示相关的快捷问题

### 2. **租客快捷问题** 🏡
当用户身份为"租客"时，显示以下8个问题：

| 按钮显示 | 完整问题 |
|---------|---------|
| 💰 Monthly Rent | What is the monthly rent amount? |
| 📅 Lease Duration | What is the lease duration? |
| 🏦 Security Deposit | What is the security deposit amount? |
| 📆 Payment Due | When is the rent due each month? |
| 🔧 Maintenance | What are my maintenance responsibilities? |
| 🐕 Pet Policy | What is the pet policy? |
| 🚪 Termination | What are the termination conditions? |
| 💡 Utilities | Who pays for utilities? |

### 3. **房东快捷问题** 🏢
当用户身份为"房东"时，显示以下8个问题：

| 按钮显示 | 完整问题 |
|---------|---------|
| 💵 Payment Terms | What are the tenant's payment obligations? |
| ⚠️ Late Penalties | What are the late payment penalties? |
| 🏗️ Maintenance | What are my maintenance obligations as landlord? |
| 🔑 Access Rights | What are the property access rights? |
| 🔄 Renewal Terms | What are the lease renewal terms? |
| ⛔ Restrictions | What are the tenant's restrictions? |
| 📋 Eviction | What are the eviction conditions? |
| 🛡️ Liability | What are my liability protections? |

## 使用方式

### 步骤1：加载合同
- 在"Upload"标签页上传合同PDF
- 或从侧边栏"Recent Files"加载已处理的合同

### 步骤2：进入Q&A界面
- 点击"💬 Q&A"标签页

### 步骤3：查看快捷问题
- 在对话开始前，会看到8个彩色按钮（2行×4列布局）
- 每个按钮显示：**emoji图标 + 简短英文标签**
- 例如："💰 Monthly Rent"、"📅 Lease Duration"
- 鼠标悬停在按钮上可以看到完整问题

### 步骤4：一键提问
- 点击任意按钮，系统自动提交完整问题
- AI会立即分析合同并返回答案
- 答案会附带参考来源（页码和文本片段）

### 步骤5：查看所有问题
- 点击"📝 View all quick questions"展开器
- 可以查看所有8个问题的完整文本和简短标签

## 显示逻辑

### 显示条件
快捷问题气泡只在以下情况下显示：
- ✅ 已加载合同文件
- ✅ 对话历史为空（首次进入或清空对话后）

### 隐藏条件
快捷问题气泡会在以下情况下隐藏：
- ❌ 已开始对话（有消息历史）
- ❌ 未加载合同文件

## 技术实现

### 前端实现
1. **状态管理**：
   - 添加 `pending_question` session state 存储待处理问题
   - 在点击按钮时设置待处理问题并重新加载页面

2. **UI布局**：
   - 使用 `st.columns(4)` 创建4列布局
   - 每行显示4个问题按钮
   - 使用emoji图标美化按钮

3. **CSS样式**：
   - 自定义按钮样式（圆角、渐变色、阴影）
   - 添加悬停效果（上移、阴影增强）

### 问题处理流程
```
用户点击按钮
    ↓
设置 pending_question
    ↓
页面重新加载 (st.rerun)
    ↓
检测到 pending_question
    ↓
提交问题到 RAG 系统
    ↓
获取AI答案
    ↓
保存到对话历史
    ↓
清除 pending_question
    ↓
页面重新加载显示对话
```

## 用户体验优化

1. **视觉反馈**：
   - 按钮使用渐变色背景
   - 悬停时按钮上移+阴影增强
   - emoji图标直观易懂

2. **信息展示**：
   - 按钮 tooltip 显示完整问题
   - 展开器查看所有问题列表
   - 提示文字引导用户操作

3. **智能隐藏**：
   - 开始对话后自动隐藏快捷问题
   - 避免界面杂乱
   - 清空对话后重新显示

## 代码位置

### 主要修改文件
- **frontend.py**
  - 第 29-59 行：CSS样式定义
  - 第 72-74 行：session state 初始化
  - 第 355-420 行：快捷问题界面实现

### 相关功能
- **身份管理**：`backend.py` - UserManager 类
- **RAG问答**：`langchain_rag_system.py` - AdvancedContractRAG 类

## 未来扩展建议

1. **自定义问题**：
   - 允许用户添加自己的常用问题
   - 保存用户的问题收藏夹

2. **智能推荐**：
   - 根据合同类型推荐相关问题
   - 基于用户历史问题推荐

3. **多语言支持**：
   - 中文问题选项
   - 双语切换功能

4. **问题分类**：
   - 财务类、法律类、维护类等分类
   - 分类标签过滤

5. **热门问题统计**：
   - 统计最常被点击的问题
   - 动态调整问题顺序

## 测试建议

### 测试场景1：租客身份
1. 注册/登录账号
2. 选择"租客"身份
3. 上传租赁合同
4. 进入Q&A界面
5. 验证显示租客相关的8个问题
6. 点击任意问题，验证能获得正确答案

### 测试场景2：房东身份
1. 注册/登录账号
2. 选择"房东"身份
3. 上传租赁合同
4. 进入Q&A界面
5. 验证显示房东相关的8个问题
6. 点击任意问题，验证能获得正确答案

### 测试场景3：切换对话
1. 使用快捷问题提问
2. 验证对话历史正确显示
3. 点击"Clear Chat"清空对话
4. 验证快捷问题重新显示

## 总结
快捷提问功能大大提升了用户体验，让用户可以快速获取合同关键信息，无需手动输入问题。根据用户身份智能推荐相关问题，使得系统更加人性化和专业化。
