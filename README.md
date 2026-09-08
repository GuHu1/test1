# ResWorld + OSZ：为遮挡特定处理的时序残差世界模型

本仓库在 **ResWorld**（ICLR 2026，Temporal Residual World Model for End-to-End Autonomous Driving）基础上，把**遮挡感知**作为一等公民注入世界模型的**数据管线内部**，而不是在 BEV 特征后面加一个支路。所有几何先验来自 **Occlusion Shadow Zone（OSZ）**——用 MiDaS 单目深度 + LiDAR 配准后的高度图做 360° 射线投射，离线自动产出，**不引入任何检测/地图人工标注**，保住了 ResWorld "perception-free" 的卖点。

---

## 一、为什么要做：ResWorld 范式在遮挡下的三个结构漏洞

ResWorld 的核心动作很简单：把相邻帧 BEV 特征投影到同一坐标系，用 TokenLearner 各自压成稀疏场景查询，然后**做差**得到"时序残差" `R_i = S_i − S_{i−1}`，世界模型只吃残差来预测动态物体的未来分布。

这套机制在遮挡面前有三个天然缺陷：

1. **残差混淆"运动"与"可见性变化"**。一个物体驶入卡车背后，两帧特征从"看得到"变"看不到"，会制造一个巨大的伪残差；反过来，物体从盲区里探出来也会产生"涌现"伪残差。世界模型分不清这是动力学信号还是观测噪声，只会照单全收地外推。
2. **最危险的物体反而没有残差**。持续躲在盲区里的物体（比如将穿过马路的行人）两帧都不可见 → 残差≈0 → 世界模型对它完全失明。ResWorld 自己也在论文 Limitation 里承认了这一点。
3. **未来 BEV 没有"未知"概念**。FGTR 模块把预测的 `B_future` 当作完整场景做碰撞检查，但遮挡区在特征上和"确认无物"的区域没有区别，模型学不会"那里可能有东西"。

**核心数学切入**：用 OSZ 提供的逐帧可见性图 `V_i` 把原始残差**按物理成因分解**，而不是让网络自己猜。

---

## 二、改动总览：一个"看不到也能脑补"的残差世界模型

整条改动都发生在 ResWorld 的既有数据流上：**BEV 逐帧生成 → 场景查询 → 残差提取 → TR-World → 未来 BEV 合成 → FGTR**。OSZ 资产在数据管线入口注入（见第四节），不改变任何下游结构。

四个 IDEA 及其组合关系：

```
逐帧BEV特征 ──► 场景查询 S_i ──► 残差 R_i = S_i − S_{i−1}
                  │                      │
            [IDEA2] 注意力偏置    [IDEA4] 永续门控（多步残差加权）
                  │                      │
                  ▼                      ▼
    [IDEA1] 可见性解耦：R → R_motion ⊗ R_emerge ⊗ Q_occ(幻影token)
                  │
                  ▼
            TR-World → 未来动态分布
                  │
            [IDEA3] 盲区年龄风险场注入
                  ▼
         B_future → FGTR → 修正轨迹
```

---

## 三、四个 IDEA 的设计

### IDEA 1：可见性解耦残差（Visibility-Disentangled Residual）

**人话**：残差这条河里有两种水——"物体真的在动"和"物体只是从视线里出现/消失"。先给它们各开一条水道，再让世界模型分别处理，而不是混在一起。

**公式**：令 `v_i[s] = Σ_hw A[s, hw] · V_i[hw]` 为查询 `s` 在第 `i` 帧的可见度（`A` 是 TokenLearner 的空间注意力，`V_i = 1 − osz_eye_i` 是逐像素可见性）。定义相邻两帧的联合可见度

$$\psi_i = \sqrt{\,v_i \cdot v_{i-1}\,}$$

则残差被分解为：

$$\underbrace{R^{motion}_i = R_i \odot \psi_i}_{\text{两帧都可见 → 真运动}} ,\qquad
\underbrace{R^{emerge}_i = R_i \odot (1 - \psi_i)}_{\text{可见性跳变 → 出遮挡/被遮挡}}$$

世界模型的输入从单一残差变为 `[R^motion; \; R^emerge]` 双 token 流（仍走同一个 `res_latent_decoder`，不是新支路）。配套两个扩展：

- **幻影 token（Q_occ）**：用风险图 `risk = (1 − V_0) · σ(age)` 作注意力偏置，从**高龄盲区**额外采样 `N_occ` 个查询 token，代表"这里可能隐藏着物体"。模型从此有了显式的"看不见但可能存在"的占位符。
- **可见性转移自监督**：在 `B_future` 上挂一个小头 ∠ `f`，预测下一帧可见性图：`V̂_{t+1} = f(B_future)`，用 OSZ 免费产出的下一帧真实可见性做 BCE 监督。**刻意只监督这一条通道**而不是稠密 BEV 外观——ResWorld 的消融（Table 4）已经证明稠密未来监督会损害规划，而"哪里会变成盲区"恰恰是最该学、又不需要标注的遮挡专属信号。

### IDEA 2：遮挡引导的稀疏注意力（Occlusion-Guided TokenLearner）

**人话**：16 个查询预算有限，原来的 TokenLearner 只学会了"哪里显著"，没学会"哪里危险"。把遮挡先验直接加进注意力分配，让查询的"注意力预算"自动向高龄盲区倾斜。

**公式**（改 `tokenlearner.py` 的 logit，约 10 行）：

$$\text{selected} = \text{softmax}\Big(\text{MLP}(x) + \lambda \cdot \underbrace{(1-V_0) \cdot \sigma\big(\frac{age}{\tau}\big)}_{\text{风险图}}\Big)$$

再加一个**注意力覆盖损失**，推动注意力质量分布向风险图对齐（无遮挡的样本自动屏蔽）：

$$\mathcal{L}_{cover} = - \sum_{hw} \hat{risk}_{hw} \cdot \log \hat{attn}_{hw}$$

### IDEA 3：盲区年龄风险场注入 B_future 合成

**人话**：`B_future` 的合成公式（ResWorld Eq.7：`B_future = B_fuse + TokenFuser(R̂)`）没有"未知"通道。我们把"盲区有多久没被观察"当作一个空间风险场，直接加进未来 BEV 的合成式，让 FGTR 的参考点注意力能"闻到"隐藏物体的气味。

**公式**：

$$B_{future} = B_{fuse} + \text{TokenFuser}(\hat R) + \underbrace{\beta \cdot risk \cdot \text{Conv}(B_{future})}_{\text{风险场注入}} ,\qquad
risk = (1 - V_0) \cdot \frac{age}{age + \tau}$$

`β` 是可学习标量且**零初始化**——训练开始时退化为基线，模型按需"打开"风险注入。年龄 `age` 来自 OSZ 的 `occlusion_age`（盲区持续时长的递归累计），是这个设计里"认知不确定性"的天然代理。

### IDEA 4：残差永续门控（Residual Permanence Gating）

**人话**：原版代码算出了两个残差 `R_t`（当前与前一帧）和 `R_{t-1}`（更早两帧）却**只用了前一个**。而物体在刚被遮挡的那一刻，它消失前的运动状态正好留在旧残差里。把它捡回来，按可见性转移做加权聚合——物体被挡住后，模型用它的"最后速度"做匀速外推式脑补，正是人类司机的行为。

**公式**：

$$\hat R = \sum_{i} g_i \cdot \text{SelfAttn}(R_i), \qquad
g_i \propto 
\begin{cases}
v_t, & i = t \ (\text{可见则信最新}) \\
(1-v_t)\cdot v_i, & i < t \ (\text{刚被遮挡则回退旧残差})
\end{cases}$$

严格地说，这**没有引入任何检测 query**（区别于 BeyondSight 那类 object permanence 工作），"物体永续"完全发生在残差潜空间内部。

---

## 四、数据管线改动（OSZ 资产如何进模型）

```
tools/osz/run_export.py  ──►  data/osz/npz/{token}.npz
                              ├── osz_eye / osz_ground / semi   (500×500 整数掩码)
                              ├── occlusion_age                (500×500 浮点, 盲区年龄/秒)
                              └── bev_height                   (障碍物高度, 中间产物)

数据集 get_data_info ──► 为当前帧+相邻帧注入 osz_path，并注入 next_info（下一帧，用于
                         可见性转移自监督；场景末尾为 None，对应样本自动置 valid=0）

LoadOSZAnnotations（pipeline 变换，新增）
  1. 读当前/相邻/下一帧的 npz
  2. 用 ego2global 位姿把每帧 OSZ 从各自自车系 warp 到当前帧自车系
     （与 ResWorld 的 BEV 对齐同源，保证一致性；几何经平移/旋转/复合单测验证）
  3. 裁剪重采样到 head 级 BEV 网格（100×100, x∈[-15,15], y∈[-30,30]）
     越界填充：可见性=1、年龄=0（视作安全区）
  4. 输出 osz_vis (F,H,W) / osz_age (F,H,W) / osz_vis_next / osz_vis_next_valid
```

所有开关在 `projects/configs/resworld/resworld_osz_config.py` 的 `pts_bbox_head` 下，可独立开/关做消融：

| 开关 | 对应 | 默认 | 说明 |
|---|---|---|---|
| `use_osz` | 总 | True | 总开关 |
| `osz_permanence` | IDEA4 | True | 永续门控（无新增参数，可 warm-start） |
| `osz_disentangle` | IDEA1a | True | 残差解耦（**改变 TokenFuser 形状**，不可 warm-start） |
| `osz_risk_inject` | IDEA3 | True | 风险场注入（β 零初始化，可 warm-start） |
| `osz_attn_bias` | IDEA2a | 2.0 | 注意力 logit 偏置强度 |
| `osz_occ_tokens` | IDEA1b | 4 | 幻影 token 数（**改变 TokenFuser 形状**） |
| `osz_occ_bias` | IDEA1b | 4.0 | 幻影 token 偏置强度 |
| `osz_vis_trans_weight` | IDEA1c | 1.0 | 可见性转移自监督 BCE 权重 |
| `osz_attn_cover_weight` | IDEA2b | 0.1 | 注意力覆盖损失权重 |

> ⚠️ `osz_disentangle` / `osz_occ_tokens` 会改变 TokenFuser 的输入 token 数（16→32 / 36），这类开关开启时**不能**加载基线 checkpoint warm-start；其余三个机制零新增参数或零初始化，可安全 warm-start。

---

## 五、服务器操作命令（ResWorld 仓库根目录执行）

环境：Python 3.8 / torch 1.9.1+cu111 / numpy 1.19.5 / nuscenes-devkit / scipy（与 `reference_code/requirements.txt` 一致）。

### 0. 准备 MiDaS 深度模型（一次性）

```bash
mkdir -p weights third_party
git clone git@github.com:isl-org/MiDaS.git third_party/MiDaS
cd weights
wget https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
```

### 1. 抽查一个场景（建议先做，验证 OSZ 产出与对齐）

```bash
python tools/osz/run_export.py \
  --dataroot data/nuscenes \
  --version v1.0-trainval \
  --outdir data/osz \
  --max-scenes 1 --sample-token ca9a282c9e77460f8360f564131a8af5 --no-viz
```

### 2. 全量导出 OSZ 资产（8 分片后台并行，长任务用 nohup）

盲区年龄沿场景时间轴递推，必须按场景分片保证递推不断裂：

```bash
nohup bash -c '
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python tools/osz/run_export.py \
    --dataroot data/nuscenes \
    --version v1.0-trainval \
    --outdir data/osz \
    --scene-shard $i --num-scene-shards 8 --no-viz \
    > work_dirs/osz/export_shard$i.log 2>&1 &
done
wait
' > work_dirs/export_all.log 2>&1 &

#查看导出进度
ls data/osz/npz | wc -l
```


### 3. 训练完整模型

```bash
mkdir -p work_dirs
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash tools/dist_train.sh \
    projects/configs/resworld/resworld_osz_config.py 4 \
    > work_dirs/train.log 2>&1 &
```

若需先跑最小验证（只开 IDEA4，可 warm-start 基线 checkpoint）：

```bash
nohup python tools/train.py \
  projects/configs/resworld/resworld_osz_config.py \
  --gpus 8 --options pts_bbox_head.osz_disentangle=False \
  pts_bbox_head.osz_occ_tokens=0 pts_bbox_head.osz_vis_trans_weight=0 \
  pts_bbox_head.osz_attn_cover_weight=0 \
  > work_dirs/train_osz_min.log 2>&1 &
```

### 4. 测试 / 评估

```bash
nohup python tools/test.py \
  projects/configs/resworld/resworld_osz_config.py \
  work_dirs/latest.pth \
  --eval bbox \
  > work_dirs/test_osz.log 2>&1 &
```

### 5. 消融矩阵（建议顺序）

| 组 | 关闭项（其余保持默认） | 验证什么 |
|---|---|---|
| A | 关闭 4 个 IDEA 相关损失/开关至最简 | 基线 + OSZ 数据管线本身 |
| B | `osz_permanence=False` | IDEA4 |
| C | `osz_disentangle=False` `osz_occ_tokens=0` | IDEA1（残差解耦+幻影token） |
| D | `osz_risk_inject=False` | IDEA3 |
| E | `osz_attn_bias=0` `osz_attn_cover_weight=0` | IDEA2 |

评估建议：除 nuScenes 全量 L2 / 碰撞率外，用 OSZ 导出时的 `summary.csv` 按 `osz_ground_ratio` 把验证集切成遮挡轻/中/重三桶分别出表——遮挡重桶的提升就是论文的主证据。

---

## 六、文件清单（改动点）

```
tools/osz/                                    # 新增：OSZ 离线生产工具（移植自 OSZMiner，格式兼容）
  run_export.py  osz_config.py  coords.py  lidar.py  loader.py
  depth_estimator.py  image_to_ego.py  bev_height_builder.py
  ray_casting.py  drivable_filter.py  pipeline.py  viz.py
projects/mmdet3d_plugin/datasets/pipelines/
  osz_loading.py                              # 新增：LoadOSZAnnotations 变换
  __init__.py  formating.py                   # 改：注册 + sys 打包 osz 张量
projects/mmdet3d_plugin/datasets/
  nuscenes_resworld_dataset.py                # 改：osz_root 注入 + next_info
projects/mmdet3d_plugin/resworld/
  resworld_head.py                            # 改：IDEA 1-4 全部机制 + 新损失
  resworld.py                                 # 改：osz 张量 train/test 透传
  tokenlearner.py                             # 改：logit_bias 参数
projects/configs/resworld/
  resworld_osz_config.py                      # 新增：OSZ 全套配置（默认全开）
```