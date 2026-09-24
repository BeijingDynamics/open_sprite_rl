# Open Sprite RL

这是 Sprite0825（0.95 m、31 自由度）人形机器人的可复现强化学习项目，目标是在
RTX 4090 级别的机器上复现当前已经通过 Isaac Lab 与 MuJoCo sim2sim 验证的结果。

## 当前结果

当前默认策略为 G60 model3450，支持站立、起步、停止、再次起步、
0.15/0.30/0.45 m/s 直行以及双向转弯。Policy 频率 50 Hz，物理仿真和外部
PD 为 500 Hz，输出 31 个关节位置目标。

Actor observation 为 795 维：8 帧关节位置、关节速度、上一时刻 action、
base angular velocity、projected gravity，再加当前 `vx/vy/yaw_rate` 命令。
其中明确没有水平本体速度、全局 root 位置和全局 yaw，便于后续 sim2real。

最终 checkpoint：

```text
baselines/sprite0825_stage2_g60_model3450_current/model_3450.pt
SHA256 6a1a80a2a2f7073698c0886133c325a46462ace6bbd3cf7de70246beafb85f15
```

## 真正的训练主线

当前结果采用 EngineAI PM01 风格的 command policy + AMP 路线。当前 50 Hz
主线与历史 100 Hz 主线都被保留：

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

./scripts/05_train_g59.sh
./scripts/06_train_g60.sh
```

仓库自带的 G59 model2999 是 G60 的准确 handoff。历史 100 Hz 主线仍可按
`01_train_g57.sh` 到 `04_train_g58f.sh` 运行。强化学习本身有随机性，复现目标是
进入相同验收区间并得到相同风格，不是要求 checkpoint 每个字节相同。

严格从零训练时，每一段完成后应先跑资格测试，再把合格 checkpoint 的路径通过
`SOURCE_CHECKPOINT` 传给下一段；不要盲目选择训练时间最长的模型。

在 Isaac Lab 中播放当前默认的 `model3450`：

```bash
./scripts/play_isaac.sh
```

检查自己训练出的 G60 checkpoint 时可设置 `CHECKPOINT=/path/to/model_N.pt`。

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

## 安全边界

本仓库保存训练与 sim2sim 结果，不是可以直接上真机的控制器；真机执行属于
`open_sprite_runtime`。真机前必须核对电机方向、零位、CAN ID、软硬限位、
并联机构解算、IMU 坐标、急停、电流和温度限制。

TWIST2 未被使用或修改。

## 许可证

本项目原创代码和 Sprite 资产采用 GNU Affero General Public License v3.0
only（`AGPL-3.0-only`）。第三方组件及 PM01 派生 expert data 继续遵循其
上游许可证和声明，详见 `NOTICE` 与 `third_party_licenses/`。
