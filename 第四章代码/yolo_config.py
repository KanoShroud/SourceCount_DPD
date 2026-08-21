"""
yolo_config.py — YOLOv8 定位系统配置参数
"""

from chapter_runtime import DEFAULT_DEVICE as RUNTIME_DEFAULT_DEVICE

# ═══════════════════════════════════════
#  系统参数
# ═══════════════════════════════════════
EDGE      = 2000       # 监测区域 ±2000m
LAMDA     = 10         # 网格步长 10m
GRID_SIZE = 401        # 网格大小 2*EDGE/LAMDA + 1
MAX_SRC   = 3          # 最大源数
R_RCV     = 500.0      # 接收站半径

# ═══════════════════════════════════════
#  YOLOv8 网络参数
# ═══════════════════════════════════════
CHANNELS   = (32, 64, 128, 256, 512)   # 各 stage 通道数
C2F_N      = 1                          # C2f 中 Bottleneck 数量
REG_MAX    = 16                         # DFL 分布离散化bins数
STRIDE_P3  = 8                          # P3 stride
STRIDE_P4  = 16
STRIDE_P5  = 32
STRIDES    = [STRIDE_P3, STRIDE_P4, STRIDE_P5]

# P3/P4/P5 通道数（PAN 输出）
PAN_CHANNELS = (CHANNELS[2], CHANNELS[3], CHANNELS[4])   # (128, 256, 512)

# ═══════════════════════════════════════
#  标签参数（仅 gen_ch4_loc_data.py 使用，记录在此供参考）
#  实际值由 gen_ch4_loc_data.py 的命令行参数控制
# ═══════════════════════════════════════
# hyp_sigma  = 15.0m (1.5px)    距离场双曲线 σ
# gauss_sigma = 2.0px           高斯热力图 σ
# hyp_mode   = 'sum'            距离场叠加模式

# ═══════════════════════════════════════
#  推理参数（训练和评估时使用）
# ═══════════════════════════════════════
PEAK_SIZE      = 9        # NMS/DPD峰值搜索窗口（像素），所有方法统一
BOX_SIZE       = 9        # bbox 标签框大小（像素），仅 D1

# ═══════════════════════════════════════
#  D4 多通道参数
# ═══════════════════════════════════════
SOFTARGMAX_TEMP    = 10.0     # soft-argmax 温度（越小越尖锐）
COORD_LOSS_WEIGHT  = 1.0     # coord loss 权重

# ═══════════════════════════════════════
#  训练参数
# ═══════════════════════════════════════
DEFAULT_EPOCHS    = 200
DEFAULT_BATCH     = 192
DEFAULT_LR        = 1e-3
DEFAULT_PATIENCE  = 30
DEFAULT_DEVICE    = RUNTIME_DEFAULT_DEVICE
