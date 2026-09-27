# DPD_MVDR 单天线函数说明

## 用途与范围

`DPD_MVDR.m` 和 `DPD_MVDR.py` 实现每站一根天线、静止源、同步复基带观测下的二维 DPD-HR 空间谱，对应 Tirer 与 Weiss 的 *High Resolution Direct Position Determination of Radio Frequency Sources* 式(27)。只处理单天线，不引入阵列响应。

`DPD_MVDR_Offical.m` 保留为作者示例对照，不是项目调用入口。作者示例直接生成频域系数，其分段数不能直接作为短时域观测的合理默认值。

## 调用方式

MATLAB（原来的8参数、2输出调用继续有效）：

```matlab
[pos, mtr, info] = DPD_MVDR(rcvPos, sig_rcv, init_pos, edge, lamda, fs, band, fc);
% 经验证后，可显式指定分段及其他配置：
opts = struct('J', J_validated, 'DiagLoadMode', 'relative', 'DiagLoad', 1e-6);
[pos, mtr, info] = DPD_MVDR(rcvPos, sig_rcv, init_pos, edge, lamda, fs, band, fc, opts);
```

Python（返回三个值，运行时只依赖 NumPy）：

```python
from DPD_MVDR.DPD_MVDR import DPD_MVDR

pos, mtr, info = DPD_MVDR(rcvPos, sig_rcv, init_pos, edge, lamda, fs, band, fc)
opts = {"J": J_validated, "DiagLoadMode": "relative", "DiagLoad": 1e-6}
pos, mtr, info = DPD_MVDR(rcvPos, sig_rcv, init_pos, edge, lamda, fs, band, fc, opts)
```

`J_validated` 由调用方根据实际观测条件确定；这里不预设正式实验的段数。

| 参数 | 约定 |
|---|---|
| `rcvPos` | L×2接收站坐标，米，L≥2且最大站间距大于0 |
| `sig_rcv` | L×N同步时域IQ，N≥2；转换为双精度后计算 |
| `init_pos` | 两元素搜索中心，米 |
| `edge` | 搜索半宽，非负；0表示只计算中心点 |
| `lamda` | 网格步长，米，严格为正；保留历史参数拼写 |
| `fs`, `band`, `fc` | Hz；0<band≤fs；fc为有限实标量，允许0 |
| `pos` | 一个全局峰坐标；不表示完整多源位置集合 |
| `mtr` | 第一维是x、第二维是y；与原函数布局一致 |
| `info` | 实际计算配置、数值检查及峰值诊断 |

两种语言使用相同字段名。Python中的一维坐标/频率向量对应MATLAB行向量。**info内的索引统一从1开始**；Python数组自身仍按0起始访问。需要绘图时，MATLAB可用 `imagesc(info.x_vec, info.y_vec, mtr.'); axis xy`；绘图在调用方完成。

## 可选配置

| opts字段 | 默认值 | 含义 |
|---|---|---|
| `J` | MATLAB `[]`；Python `None`/`[]` | 自动分段；显式值必须为正整数且每段至少2点 |
| `SegmentMargin` | 40 | 相对于最大站间传播时差上界的工程裕度；不是论文规定的常数 |
| `FrequencyRange` | 空 | 默认选 `[-band/2, band/2]`；非空的 `[f_low,f_high]` 覆盖默认选频范围，含端点，须处于 `[-fs/2,fs/2]` |
| `DiagLoadMode` | `'relative'` | `'relative'` 或 `'absolute'`，大小写不敏感 |
| `DiagLoad` | 1e-6 | 相对加载系数，或绝对加载值；非负 |
| `SpectrumLoad` | 0 | 最终位置矩阵的标量对角加载；非负 |
| `PropagationSpeed` | 299792458 | m/s；作者对照可显式使用300000000 |
| `PeakTieTolerance` | 1e-10 | 与最大谱值的相对差不超过该值时视为并列；取列优先顺序的首项 |

未知配置字段报错，避免拼写错误被静默忽略。

### 自动分段与短观测

记最大站间距为D，传播速度为c。自动模式计算：

```text
minimum_samples = max(2, ceil(fs * SegmentMargin * D/c))
J = floor(N / minimum_samples)
N_fft = floor(N / J)
```

与旧版相比，在整数样本边界上采用向上取整，保证自动模式的实际段长满足所选工程裕度。J<1时返回 `DPD_MVDR:InsufficientDuration`，不自动降低裕度。

显式J允许研究不同分段，但会如实返回 `margin_met`、`segment_margin`、`rank_limited`。`margin_met=true` 只表示通过当前工程裕度，不等于正式验证了有限窗频域近似。D/c是站间时延差的几何上界，不是原文最大绝对传播时延的直接替代证明。

项目的4096点、100 MHz、1000米最大站间距在默认裕度40下不能自动分段。显式4段的段长与D/c之比约3.07，须另行验证近似误差，不将“能运行”当作模型假设成立。

## 两种语言共同遵循的计算规则

1. 从开头顺序截取J个等长、不重叠片段；尾部N−J×N_fft个样本丢弃并报告。
2. 不去均值、不加窗、不补零；FFT不除以长度。奇偶长度都采用居中的正确FFT频率序列。
3. 逐频点计算 `R = sum_j(x_j*x_j^H)/J`，Hermitian对称化后加载：相对模式为 `DiagLoad*real(trace(R))/L`，绝对模式为 `DiagLoad`。
4. 检查加载后矩阵最小/最大特征值比必须大于 `64*eps(float64)`，再解线性方程得到逆协方差。不使用伪逆，也不偷偷追加加载。
5. 每个位置计算时延，再按频率顺序累加相位校正后的逆协方差，取 `1/(lambda_min(S)+SpectrumLoad)`。
6. 谱值必须有限且为正。网格通过相同的整数步数及8倍机器精度端点容差生成；不强制加入非整步终点。
7. 全局峰按 `PeakTieTolerance` 处理近似并列，以MATLAB列优先顺序取首项；不执行多峰检测或源数判决。

载频fc保留在导向相位中以兼容原接口；理论上该自由复增益谱对统一载频平移不变。跨语言FFT、线性代数库可能产生舍入差异，**不保证逐位相同**。检验采用相对容差1e-8和按参考谱幅度缩放的绝对容差，位置索引、离散配置及错误标识要求一致。正好位于分支或峰值容差边界的输入仍可能受浮点舍入影响。

### 作者实现对照

统一输入频域系数的归一化、传播速度和频率后，设置：

```matlab
opts = struct('J', J_validated, 'DiagLoadMode', 'absolute', ...
    'DiagLoad', 0.001, 'SpectrumLoad', 0.001, 'PropagationSpeed', 300000000);
```

作者脚本的谱布局是(y,x)，对比时应转置。固定绝对加载只适用于相同数据尺度，不能直接套到项目物理功率IQ上。通过IFFT封装作者频域系数只用于公式对照，不能证明真实有限时域分段模型成立。

## 诊断和错误

- `N_fft`是每段采样点数，`K`是实际参与求和的频点数，含义分开。
- `rank_bound=min(L,J)`仅是样本协方差的秩上界；J≥L不保证满秩。
- `cov_eigenvalue_ratio`记录加载后每频点最小/最大特征值比。
- `diag_load_per_frequency`记录实际加载量；`N_used/N_discarded`记录样本使用情况。
- `peak_index`为1起始的[x索引,y索引]；`peak_linear_index`为1起始列优先索引。
- `peak_tie_count`记录近似并列候选数；`peak_value`为返回点的值，`maximum_value`为全图最大值。
- `status`为 `OK` 或 `LOW_SEGMENT_MARGIN`；同时须检查 `rank_limited`。
- 无效输入、零能量、空频率集合、不可逆协方差等均明确报错。Python捕获 `DPD_MVDR_Error` 并读取 `.code`；该字段与MATLAB的 `ME.identifier` 一致。

## 复核入口

```powershell
& 'D:\Software\anaconda3\envs\PyTorch\python.exe' -m py_compile DPD_MVDR.py verify_DPD_MVDR.py
& 'D:\Software\anaconda3\envs\PyTorch\python.exe' -m ruff check DPD_MVDR.py verify_DPD_MVDR.py --select F,E9
& 'D:\Software\anaconda3\envs\PyTorch\python.exe' verify_DPD_MVDR.py
```

MATLAB未在PATH时，加 `--matlab 'D:\Software\MATLAB\R2025b\bin\matlab.exe'`。检查程序依赖NumPy、SciPy及MATLAB，不自动安装依赖。每次创建新的 `../outputs_e2e/verification/dpd_mvdr` 子目录，保存共享输入、MATLAB输出、源文件SHA256、日志及JSON报告；失败目录保留。

检查涵盖双语言有效/无效输入、作者单天线公式、已知位置峰、幅度缩放、站序置换、载频不变性及并列峰规则。它是工程回归，不是正式定位精度、分辨率或短观测适用性实验。本次验证事实统一记录在[修改记录.md](../修改记录.md)。
