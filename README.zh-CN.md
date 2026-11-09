# Open Sprite RL

这是 Sprite0825（0.95 m、31 自由度）人形机器人的可复现强化学习项目，目标是在
RTX 4090 级别的机器上复现当前已经通过 Isaac Lab 与 MuJoCo sim2sim 验证的结果。

[English README](README.md) · 稳定行走工程笔记：
[中文](docs/SPRITE0825_STABLE_WALK_HANDOFF.md) /
[English](docs/SPRITE0825_STABLE_WALK_HANDOFF.en.md)

## 当前结果

当前默认策略为 [G74 model3000](baselines/sprite0825_stage2_g74_model3000/README.md)，
即开始研究后退之前，四肩 J4340P 的 Sprite0825 在真机上稳定行走使用的版本。
通过 Isaac Lab 与 MuJoCo 的站立、前进、转弯和重复启停验收；真机有监督地验证了
`0.10 m/s` 指令前进、启停和低速手柄转向。没有宣称后退或侧移已经通过。
Policy 为 50 Hz，仿真物理及 PD 为 500 Hz，输出 31 个关节位置目标。

Actor observation 为 795 维：8 帧关节位置、关节速度、上一时刻 action、
base angular velocity、projected gravity，再加当前 `vx/vy/yaw_rate` 命令。
其中没有水平本体速度、全局 root 位置和全局 yaw。对应的 Jetson 真机控制源码见
[固定的真机 runtime](#固定的真机-runtime)。

最终 checkpoint：

```text
baselines/sprite0825_stage2_g74_model3000/model_3000.pt
SHA256 1b186bb10ee3cfa011d464e7c86b4f72b4c90a466c1576e3885cea5cd223f4f5
```

## 真正的训练主线

G74 使用 PM01 风格的速度指令 PPO + 无条件 AMP，从零训练，不从 G60 续训，也不是
SuperMimic。使用 G71 清理过接触的行走参考、站立专家、匹配的 v5 资产和修正后的
肩 roll 中心。历史 50 Hz 与 100 Hz 主线都被保留：

1. **G59**：以最终部署频率 50 Hz 从零训练。
2. **G60**：从 G59 model2999 继续，按物理时间修正 action 正则项，并用软约束
   控制腰部 roll；最终选择 model3450。
3. **G57**：历史 100 Hz 主线在 Sprite0825 上从零开始。AMP 数据为 10% 原生站立和 90% 已确认的
   PM01 正常直行，策略从一开始就有可部署的速度命令接口。
4. **G58A**：加入 Sprite 实际 J4340P 与差动 J4310P 的 torque-speed envelope。
5. **G58B**：恢复 0.45 m/s 高速段，同时保持动作数据与整体方法不变。
6. **G58F**：从视觉确认过的 G58B model925 出发，温和加入 yaw 命令覆盖，最终
   选择 model1050。

G59 是全新的 50 Hz scratch run，不是从 G58F 续训；G60 才是从 G59 续训。
`candidates/` 中另行保留了四个肩部 J4340P 的 G74 model5999，但它目前不是默认发布。

## 环境

实际验证环境为 Ubuntu 24.04、RTX 4090 24 GB、14 CPU 核、50 GB RAM、
Isaac Sim 5.1.0。固定版本如下：

- Isaac Lab：`b4c321024792976150ca55fddb26fa34480d974e`
- EngineAI AMP：`83ba64bbb58a02e14483e52adce5f893f3f31cdf`
- `rsl-rl-lib==5.0.1`
- Python 3.11、PyTorch 2.7.0+cu128、训练侧 NumPy 1.26.0

## 安装

```bash
git clone https://github.com/isaac-sim/IsaacLab.git "$HOME/IsaacLab"
git -C "$HOME/IsaacLab" checkout b4c321024792976150ca55fddb26fa34480d974e

git clone https://github.com/engineai-robotics/engineai_amp.git "$HOME/engineai_amp"
git -C "$HOME/engineai_amp" checkout 83ba64bbb58a02e14483e52adce5f893f3f31cdf

cd "$HOME/IsaacLab"
./isaaclab.sh --install rsl_rl
./isaaclab.sh -p -m pip install -e "$HOME/engineai_amp"

cd /path/to/open_sprite_rl
export ISAACLAB_ROOT="$HOME/IsaacLab"
./scripts/install_overlay.sh
./scripts/verify_release.py
sha256sum -c ARTIFACT_SHA256SUMS
```

若从 Windows 主机 clone，由于冻结审计包保留了原始深层路径，请先运行
`git config --global core.longpaths true`。

无界面训练前需要接受 EULA：

```bash
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=yes
```

## 训练

```bash
export ISAACLAB_ROOT="$HOME/IsaacLab"
export SPRITE_RL_ROOT="$PWD"

./scripts/07_train_g74.sh
```

G74 默认 4096 个环境、6000 次更新。早期 50 Hz 主线仍可运行
`05_train_g59.sh` 和 `06_train_g60.sh`；仓库自带的 G59 model2999 是 G60 的准确
handoff。历史 100 Hz 主线仍可按
`01_train_g57.sh` 到 `04_train_g58f.sh` 运行。强化学习本身有随机性，复现目标是
进入相同验收区间并得到相同风格，不是要求 checkpoint 每个字节相同。

严格从零训练时，每一段完成后应先跑资格测试，再把合格 checkpoint 的路径通过
`SOURCE_CHECKPOINT` 传给下一段；不要盲目选择训练时间最长的模型。

在 Isaac Lab 中播放当前默认的 `model3000`：

```bash
./scripts/play_isaac.sh
```

检查自己训练出的 G74 checkpoint 时可设置 `CHECKPOINT=/path/to/model_N.pt`。
两个播放脚本均可用 `BASELINE=g60` 切回早期 G60 仿真版本。

## MuJoCo 播放

MuJoCo 播放不需要 Isaac Sim：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mujoco.txt
./scripts/play_mujoco.sh 120
```

按键：`Z` 前进，`X` 停止，`Q` 左转，`E` 右转。
服务器或 CI 无显示器检查可运行：`VIEWER=0 ./scripts/play_mujoco.sh 10`。

更完整的训练与验收说明见：

- [训练流水线](docs/TRAINING_PIPELINE.md)
- [复现契约](docs/REPRODUCIBILITY.md)
- [数据来源](docs/DATA_PROVENANCE.md)
- [硬件和部署边界](docs/HARDWARE_AND_DEPLOYMENT.md)

## 固定的真机 runtime

**G74 model3000 对应的真机源码固定为
[`open_sprite_runtime@4515206`](https://github.com/BeijingDynamics/open_sprite_runtime/tree/4515206c24404e18a2d5c168eccdb7468aea6ab7)。**
这是开始训练后退之前的 Jetson 版本，使用 ICM45686 800 Hz 原始六轴，
已在监督下验证 Xbox 前进、停步和低速左右转向。请检出固定提交，不要用随时会更新的
`main` 替代：

```bash
git clone https://github.com/BeijingDynamics/open_sprite_runtime.git
git -C open_sprite_runtime checkout --detach 4515206c24404e18a2d5c168eccdb7468aea6ab7
```

本仓库的 `runtime/` 目录是仿真工具。两个仓库都不是克隆后就能上电的部署包；
每台机器人及目标 Jetson 仍需准备：

- **策略和控制参数：** 配套的 G74 model3000 ONNX、观测和动作契约，以及生成的
  native 参数包。参数包需包含真机验证过的 PD 增益、命令与反馈限值和看门狗设置，
  不能把仿真契约的 PD 直接发给电机。
- **31 个电机的现场清单：** CAN FD 接口、命令和反馈 ID、编码器零点及符号、
  直连关节方向和传动比、实测位置边界、扭矩/速度/电流/温度限值，以及各驱动
  回读的 MIT PMAX/VMAX/TMAX。
- **并联机构标定：** 左踝、右踝及头部 pitch/roll 分别测量 2×2 矩阵，
  记录电机行顺序、零偏和有效范围。
- **IMU 和手柄：** 核对 pelvis 安装旋转、SPI 与 INT1 DRDY 配置，
  验证投影重力和陀螺方向；启动时保持静止 15 秒估计陀螺零偏。
  配对 Xbox 手柄后，验证现场轴、按键及松开 LT 后速度指令回零的行为。
- **目标机编译和验收：** 安装匹配的 ONNX Runtime 库，编译并验收 native 策略、
  CAN 和 ICM 程序，重新生成该机器的验收校验文件。确认实时调度权限，完成零增益预检，
  并确认防跌支撑和独立断电后，才另行批准有动力测试。

详细步骤见固定版本的 [Xbox 部署说明](https://github.com/BeijingDynamics/open_sprite_runtime/blob/4515206c24404e18a2d5c168eccdb7468aea6ab7/docs/XBOX_DEPLOYMENT_20261019.md)
和[硬件契约](https://github.com/BeijingDynamics/open_sprite_runtime/blob/4515206c24404e18a2d5c168eccdb7468aea6ab7/docs/HARDWARE_CONTRACT.md)。
公开快照不包含现场标定、真机可执行文件或机器专用验收校验文件。

TWIST2 未被使用或修改。

## 许可证

本项目原创代码和 Sprite 资产采用 GNU Affero General Public License v3.0
only（`AGPL-3.0-only`）。第三方组件及 PM01 派生 expert data 继续遵循其
上游许可证和声明，详见 `NOTICE` 与 `third_party_licenses/`。
