# DEFAME Benchmark 评估系统详细说明

## 概览

DEFAME 是一个用于事实核查的评估框架，支持多个基准测试数据集（AVeriTeC、MOCHEG、VERITE、ClaimReview2024、FEVER 等）。该系统采用高度并行化的设计，能够高效地处理大规模的事实核查任务。

## 系统架构

### 核心组件

1. **评估引擎** (`defame/eval/evaluate.py`)
   - 主要的评估入口点
   - 负责协调整个评估流程
   - 管理实验配置和结果保存

2. **基准测试抽象** (`defame/eval/benchmark.py`)
   - 定义了统一的基准测试接口
   - 支持多种数据集格式
   - 提供标准化的数据加载和预处理

3. **并行处理系统** (`defame/helpers/parallelization/`)
   - **Pool** - 工作池管理器
   - **Worker** - 工作进程
   - **Task** - 任务封装

4. **事实核查器** (`defame/fact_checker.py`)
   - 执行具体的事实核查逻辑
   - 支持多种推理策略
   - 集成多种工具和模型

## 评估流程详解

### 1. 初始化阶段

```python
def evaluate(
    llm: str,                      # 使用的语言模型
    benchmark_name: str,           # 基准测试名称
    tools_config: dict,            # 工具配置
    experiment_name: str = None,   # 实验名称
    n_workers: int = None,         # 工作进程数量
    # ... 其他参数
):
```

#### 1.1 配置验证与设置
- 验证工具配置的有效性
- 设置实验目录和日志系统
- 确定工作进程数量（基于GPU数量和模型类型）

```python
# 根据模型类型自动配置工作进程数
n_devices = torch.cuda.device_count()
if n_workers is None:
    match llm:
        case "llama3_8b":
            n_workers = 8
        case "llama3_70b":
            n_workers = 3  # 8个A100 GPU只能容纳3个副本
        case _:
            n_workers = n_devices * 2  # 每个GPU 2个工作进程
```

#### 1.2 数据集加载与采样
- 加载指定的基准测试数据集
- 支持多种采样策略：
  - 指定样本数量 (`n_samples`)
  - 指定特定样本ID (`sample_ids`)
  - 随机采样 (`random_sampling`)

#### 1.3 断点续评估支持
- 检查已完成的样本
- 只处理尚未完成的任务
- 恢复之前的统计信息

### 2. 并行处理架构

#### 2.1 Pool（工作池）设计

**Pool类** 是整个并行系统的核心管理器：

```python
class Pool:
    def __init__(self, n_workers: int, device_assignments: list[int] = None, **kwargs):
        self.n_workers = n_workers
        self.device_assignments = device_assignments  # GPU设备分配
        
        # 三个关键队列
        self._scheduled_tasks = Queue()    # 待处理任务队列
        self._results = Queue()           # 结果队列
        self._errors = Queue()            # 错误队列
        
        self.tasks: dict[str, Task] = {}  # 任务状态跟踪
        self.workers: list[FactCheckerWorker] = []  # 工作进程列表
```

**关键特性**：
- **设备分配策略**：自动将工作进程均匀分配到可用的CUDA设备上
- **异步监督**：通过独立线程监督所有工作进程
- **容错处理**：自动检测和报告工作进程错误
- **优雅终止**：支持安全的进程终止和资源清理

#### 2.2 Worker（工作进程）设计

**FactCheckerWorker** 继承自multiprocessing.Process：

```python
class FactCheckerWorker(Worker):
    def __init__(self, identifier: int, *args, **kwargs):
        self.runner = Runner(worker_id=identifier)
        super().__init__(identifier, *args, target=self.runner.execute, **kwargs)
```

**Runner类** 在子进程中执行实际工作：

```python
class Runner:
    def execute(self, input_queue, output_queue, connection, device_id, **kwargs):
        # 1. 设置CUDA设备
        device = f"cuda:{device_id}" if device_id is not None else None
        
        # 2. 初始化FactChecker实例
        fc = FactChecker(device=device, **kwargs)
        
        # 3. 持续处理任务
        while self.running:
            task = input_queue.get()  # 从队列获取任务
            
            # 处理不同类型的任务
            if isinstance(task.payload, Content):
                # 内容中的声明提取
                claims = fc.extract_claims(task.payload)
                output_queue.put(claims)
            elif isinstance(task.payload, Claim):
                # 声明验证
                doc, meta = fc.verify_claim(task.payload)
                output_queue.put((doc, meta))
```

#### 2.3 Task（任务）设计

**Task类** 封装了单个处理单元：

```python
class Task:
    def __init__(self, payload: Content | Claim, id: str | int, 
                 status: Status = Status.PENDING):
        self.id = str(id)
        self.payload = payload  # 可以是Content或Claim
        self.status = status    # PENDING, RUNNING, DONE, FAILED
        self.worker_id = None   # 分配的工作进程ID
        self.result = None      # 处理结果
```

**状态管理**：
- `PENDING`: 等待处理
- `RUNNING`: 正在处理
- `DONE`: 处理完成
- `FAILED`: 处理失败

### 3. 通信机制

#### 3.1 队列通信
- **任务队列** (`_scheduled_tasks`): 主进程向工作进程分发任务
- **结果队列** (`_results`): 工作进程返回处理结果
- **错误队列** (`_errors`): 收集和报告错误信息

#### 3.2 管道通信
每个工作进程都有一个双向管道用于元数据通信：
- 状态更新（开始处理、完成等）
- 进度报告
- 错误信息
- 日志消息

```python
# 工作进程向主进程报告状态
def report(status_message: str, **kwargs):
    connection.send(dict(
        worker_id=self.worker_id,
        task_id=task.id,
        status_message=status_message,
        status=Status.RUNNING,
        **kwargs
    ))
```

### 4. 执行流程

#### 4.1 任务分发
```python
# 将每个样本转换为任务并添加到队列
for instance in samples_to_evaluate:
    task = Task(instance["input"], id=instance["id"])
    pool.add_task(task)
```

#### 4.2 并行处理
```python
# 等待所有工作进程就绪
pool.wait_until_ready()

# 持续获取结果直到所有任务完成
progress = tqdm(range(n_samples))
while progress.n + pool.n_failed_tasks < n_samples:
    try:
        output = pool.get_result(timeout=60)
        benchmark.process_output(output)  # 处理结果
        progress.update(1)
    except Empty:
        if not pool.is_running():
            break
```

#### 4.3 结果处理
每个基准测试都有专门的`process_output`方法：

```python
def process_output(self, output):
    doc, meta = output
    claim = doc.claim
    prediction = doc.verdict
    
    # 保存预测结果
    self._save_prediction(doc, meta, claim, prediction, 
                         ground_truth_label, ground_truth_justification)
```

### 5. 监控和容错

#### 5.1 进程监控
Pool通过专门的监控线程持续检查：
- 工作进程存活状态
- 任务执行进度
- 错误和异常情况

```python
def run(self):
    """监控线程主循环"""
    self._run_workers()
    self.wait_until_ready()
    while self.is_running() and not self.terminating:
        self.process_messages()  # 处理工作进程消息
        self.report_errors()     # 报告错误
        time.sleep(0.1)
```

#### 5.2 错误处理
- **进程崩溃检测**：自动检测工作进程异常终止
- **任务超时处理**：设置合理的超时机制
- **资源清理**：确保异常情况下的资源正确释放

#### 5.3 优雅终止
```python
def stop(self):
    self.terminating = True
    for worker in self.workers:
        if worker.is_alive():
            worker.terminate()
```

### 6. 性能优化策略

#### 6.1 设备利用率优化
- **智能设备分配**：将工作进程均匀分配到多个GPU
- **内存管理**：根据模型大小调整每个设备的工作进程数
- **负载均衡**：动态调整任务分配策略

#### 6.2 I/O优化
- **批量处理**：合并小的I/O操作
- **异步写入**：结果异步保存到磁盘
- **缓存机制**：缓存常用的模型和数据

#### 6.3 内存优化
- **模型共享**：在可能的情况下共享模型权重
- **渐进式处理**：避免同时加载所有数据
- **垃圾回收**：及时释放不需要的资源

### 7. 结果收集与分析

#### 7.1 实时统计
- 任务完成数量和速度
- 错误率统计
- 资源使用情况
- 预估剩余时间

#### 7.2 最终评估
```python
def finalize_evaluation(experiment_dir, benchmark, stats):
    # 计算各种指标
    - 准确率、精确率、召回率、F1分数
    - 平均处理时间
    - 资源使用统计
    - 成本分析（对于API模型）
    
    # 生成可视化结果
    - 混淆矩阵
    - 性能分析图表
    - 错误案例分析
```

#### 7.3 输出格式
- **machine-readable**: `results.json` - 用于后续分析
- **human-readable**: `results.yaml` - 便于阅读
- **预测结果**: `predictions.csv` - 详细的预测记录
- **实例统计**: `instance_stats.csv` - 每个样本的处理统计

### 8. 扩展性设计

#### 8.1 新基准测试集成
继承`Benchmark`抽象类：
```python
class NewBenchmark(Benchmark):
    def _load_data(self) -> list[dict]:
        # 实现数据加载逻辑
        pass
    
    def process_output(self, output):
        # 实现结果处理逻辑
        pass
```

#### 8.2 新工具集成
通过配置系统轻松添加新工具：
```python
tools_config = {
    "new_tool": {
        "param1": "value1",
        "param2": "value2"
    }
}
```

#### 8.3 新模型支持
通过模型注册机制支持新的语言模型：
```python
# 在available_models.csv中添加新模型
# 实现相应的模型接口
```

## 使用示例

### 基础评估
```python
from defame.eval.evaluate import evaluate

evaluate(
    llm="gpt_4o",
    benchmark_name="averitec",
    tools_config={
        "searcher": {
            "search_config": {"google": {}},
            "limit_per_search": 5
        }
    },
    n_workers=8,
    n_samples=100
)
```

### 大规模并行评估
```python
evaluate(
    llm="llama3_70b",
    benchmark_name="mocheg", 
    tools_config={
        "searcher": {"search_config": {"google": {}}},
        "geolocator": {}
    },
    n_workers=3,  # 3个A100 GPU的限制
    random_sampling=True,
    print_log_level="info"
)
```

### 断点续评估
```python
evaluate(
    # ... 其他参数
    continue_experiment_dir="/path/to/previous/experiment",
    n_workers=16
)
```

## 性能考虑

### 工作进程数量选择
- **CPU密集型任务**：`n_workers = cpu_count()`
- **GPU密集型任务**：`n_workers = gpu_count * k`（k取决于GPU内存）
- **混合任务**：根据瓶颈资源调整

### 内存管理
- 监控每个工作进程的内存使用
- 设置合理的批量大小
- 及时释放不需要的数据

### 网络和I/O
- 对于需要网络请求的工具，考虑请求频率限制
- 使用异步I/O减少阻塞
- 合理配置超时和重试机制

这个并行化设计使得DEFAME能够高效地处理大规模的事实核查评估任务，同时保持良好的扩展性和容错能力。
