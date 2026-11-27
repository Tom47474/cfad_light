import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import warnings

warnings.filterwarnings("ignore")

# -------------------------- 1. 配置参数（严格复用 paper1.pdf 实验逻辑）--------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 5  # 输入时序长度（5分钟，参考 paper1.pdf 预测粒度）
ACTION_DIM = 3  # 每个智能体的动作维度（-Δ、0、+Δ）
STATE_DIM = 8  # 状态维度（单链4维+跨链4维，见下文特征工程）
GAMMA = 0.95  # 折扣因子，与 paper1.pdf DW-DQN 一致（第1-106节）
LR = 1e-3  # 学习率（Adam优化器），与 paper1.pdf 第1-124节一致
BATCH_SIZE = 64  # 批次大小，参考 paper1.pdf 训练参数
EPOCHS = 50  # 训练轮次（强化学习迭代次数）
MEMORY_SIZE = 10000  # 经验回放池大小，与 paper1.pdf DW-DQN 一致（第1-106节）
EPS_START = 0.9  # ε-greedy 初始探索率
EPS_END = 0.1  # ε-greedy 最终探索率
EPS_DECAY = 200  # ε-greedy 衰减步数

# 跨链参数调整步长（基于 paper1.pdf 块大小范围扩展）
BLOCK_SIZE_STEP = 20  # 块大小调整步长（KB），与前文多智能体动作空间一致
BATCH_SIZE_STEP = 10  # 中继批处理量调整步长（笔）

# 场景配置（基于 paper1.pdf 多领域应用，第1-7节扩展）
SCENE_CONFIG = {
    "Finance_HFT": {
        "latency_thresh": 0.5,  # 延迟阈值（秒）
        "throughput_thresh": 120,  # 吞吐量阈值（TPS）
        "base_weights": {"w_l": 0.5, "w_t": 0.3, "w_p": 0.05, "w_s": 0.15},  # 基础权重
        "alpha": 0.8  # 动态权重调整系数
    },
    "Healthcare_Data": {
        "latency_thresh": 2.0,
        "throughput_thresh": 50,
        "base_weights": {"w_l": 0.1, "w_t": 0.1, "w_p": 0.5, "w_s": 0.3},
        "alpha": 0.6
    },
    "IoT_Sensing": {
        "latency_thresh": 1.0,
        "throughput_thresh": 150,
        "base_weights": {"w_l": 0.2, "w_t": 0.4, "w_p": 0.1, "w_s": 0.3},
        "alpha": 0.9
    },
    "SupplyChain": {
        "latency_thresh": 2.0,
        "throughput_thresh": 80,
        "base_weights": {"w_l": 0.3, "w_t": 0.2, "w_p": 0.2, "w_s": 0.3},
        "alpha": 0.7
    }
}


# -------------------------- 2. 环境定义（跨链参数调优环境，贴合真实跨链逻辑）--------------------------
class CrossChainEnv:
    def __init__(self, data, scaler, scene_label="Finance_HFT"):
        """
        跨链强化学习环境：模拟跨链参数调优过程，返回状态、奖励与终止信号
        参数：
            data: 预处理后的跨链数据集（HFBDP-Cross.csv）
            scaler: 特征标准化器（paper1.pdf 公式10：Z-score）
            scene_label: 场景标签（从 SCENE_CONFIG 中选择）
        """
        self.data = data.reset_index(drop=True)
        self.scaler = scaler
        self.scene = SCENE_CONFIG[scene_label]
        self.scene_label = scene_label

        # 状态字段（paper1.pdf 状态字段+跨链扩展字段）
        self.state_cols = [
            "tx_arrival_rate", "block_size_current", "orderer_node_count", "latency_local",
            "latency_relay", "tx_cross_count", "latency_cross", "throughput_cross"
        ]
        # 可调整参数（跨链优化目标）
        self.params = {
            "block_size": 300.0,  # 初始块大小（KB），参考 paper1.pdf 中间值
            "batch_size_relay": 20.0  # 初始中继批处理量（笔）
        }
        # 数据索引（时序遍历）
        self.current_idx = SEQ_LEN  # 从第SEQ_LEN步开始（需输入时序特征）
        self.max_idx = len(self.data) - 1

    def get_state(self):
        """获取当前状态（输入时序特征+当前参数，参考 paper1.pdf 状态定义）"""
        # 1. 时序特征（最近SEQ_LEN步的状态字段）
        seq_state = self.data.iloc[self.current_idx - SEQ_LEN:self.current_idx][self.state_cols].values
        # 2. 加入当前参数（块大小、中继批处理量）
        param_state = np.array([self.params["block_size"], self.params["batch_size_relay"]])
        # 3. 拼接并标准化（paper1.pdf 公式10）
        state = np.concatenate([seq_state.flatten(), param_state], axis=0)
        state = self.scaler.transform(state.reshape(1, -1)).flatten()
        return torch.FloatTensor(state).to(DEVICE)

    def step(self, actions):
        """
        执行动作（多智能体分工调整参数），返回下一个状态、奖励、终止信号
        actions: 智能体动作列表 [源链动作, 中继动作, 目标链动作]
            - 源链动作：调整块大小（-1: -BLOCK_SIZE_STEP, 0: 不变, 1: +BLOCK_SIZE_STEP）
            - 中继动作：调整批处理量（-1: -BATCH_SIZE_STEP, 0: 不变, 1: +BATCH_SIZE_STEP）
            - 目标链动作：调整验证节点数（简化为0，暂不优化，聚焦核心参数）
        """
        # 1. 执行动作（更新跨链参数，限制在合理范围）
        # 源链智能体：调整块大小（paper1.pdf 块大小范围 10-800 KB）
        block_size_delta = actions[0] * BLOCK_SIZE_STEP
        self.params["block_size"] = np.clip(
            self.params["block_size"] + block_size_delta, 10.0, 800.0
        )
        # 中继智能体：调整批处理量（范围 5-50 笔，符合跨链转发实际）
        batch_size_delta = actions[1] * BATCH_SIZE_STEP
        self.params["batch_size_relay"] = np.clip(
            self.params["batch_size_relay"] + batch_size_delta, 5.0, 50.0
        )

        # 2. 计算下一个状态（时序推进1步）
        self.current_idx += 1
        done = self.current_idx >= self.max_idx  # 数据遍历完则终止
        next_state = self.get_state() if not done else torch.zeros_like(self.get_state())

        # 3. 计算奖励（基于动态权重，扩展 paper1.pdf DW-DQN 奖励逻辑）
        # 3.1 获取当前跨链性能（真实值，从数据集中读取）
        current_perf = self.data.iloc[self.current_idx - 1] if not done else self.data.iloc[-1]
        latency_cross = current_perf["latency_cross"]
        throughput_cross = current_perf["throughput_cross"]
        privacy_risk = 0.05 if self.scene_label == "Healthcare_Data" else 0.01  # 隐私风险（场景相关）
        security_cost = 0.1 * current_perf["orderer_node_count"] / 9  # 安全开销（节点数越多开销越大）

        # 3.2 动态权重调整（基于场景基础权重+性能预测偏差，扩展 paper1.pdf 公式24）
        base_w = self.scene["base_weights"]
        # 延迟偏差：超过阈值则提升延迟权重
        latency_bias = max(0, latency_cross - self.scene["latency_thresh"]) / self.scene["latency_thresh"]
        w_l = base_w["w_l"] * (1 + self.scene["alpha"] * latency_bias)
        # 吞吐量偏差：低于阈值则提升吞吐量权重
        throughput_bias = max(0, self.scene["throughput_thresh"] - throughput_cross) / self.scene["throughput_thresh"]
        w_t = base_w["w_t"] * (1 + self.scene["alpha"] * throughput_bias)
        # 隐私/安全权重：场景固定（简化，实际可基于隐私风险动态调整）
        w_p = base_w["w_p"]
        w_s = base_w["w_s"]
        # 权重归一化（确保和为1，与 paper1.pdf 权重约束一致）
        total_w = w_l + w_t + w_p + w_s
        w_l, w_t, w_p, w_s = w_l / total_w, w_t / total_w, w_p / total_w, w_s / total_w

        # 3.3 多目标奖励计算（扩展 paper1.pdf 公式21-23）
        r_l = -max(0, latency_cross - self.scene["latency_thresh"])  # 延迟奖励（低于阈值为正）
        r_t = throughput_cross / self.scene["throughput_thresh"]  # 吞吐量奖励（归一化）
        r_p = -privacy_risk  # 隐私奖励（风险越低越好）
        r_s = -security_cost  # 安全奖励（开销越低越好）
        reward = w_l * r_l + w_t * r_t + w_p * r_p + w_s * r_s

        return next_state, reward, done

    def reset(self):
        """重置环境（用于每轮训练开始）"""
        self.current_idx = SEQ_LEN
        self.params = {"block_size": 300.0, "batch_size_relay": 20.0}
        return self.get_state()


# -------------------------- 3. 经验回放池（复用 paper1.pdf DW-DQN 机制）--------------------------
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """添加经验（state: 状态, action: 动作, reward: 奖励, next_state: 下一个状态, done: 终止信号）"""
        self.buffer.append((state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done))

    def sample(self, batch_size=BATCH_SIZE):
        """采样批次经验（转换为Tensor）"""
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(DEVICE)
        actions = torch.LongTensor(np.array([b[1] for b in batch])).to(DEVICE)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(DEVICE)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# -------------------------- 4. 模型定义（单智能体DQN与多智能体MA-RL）--------------------------
### 4.1 单智能体DQN（用于DW-DQN-Paper1基线，复用 paper1.pdf DW-DQN 结构）
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # 隐藏层1，参考 paper1.pdf 网络规模
        self.fc2 = nn.Linear(128, 128)  # 隐藏层2
        self.fc3 = nn.Linear(128, action_dim)  # 输出层（动作价值）

    def forward(self, x):
        """x: (batch_size, state_dim) → (batch_size, action_dim)"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


### 4.2 多智能体MA-RL（创新点：场景识别+动态权重，每个智能体一个DQN）
class MARL_Agent:
    def __init__(self, agent_name, state_dim, action_dim):
        """
        多智能体中的单个智能体
        agent_name: 智能体名称（source: 源链, relay: 中继, target: 目标链）
        state_dim: 状态维度
        action_dim: 动作维度
        """
        self.agent_name = agent_name
        # DQN网络（主网络+目标网络，复用 paper1.pdf DW-DQN 双层架构）
        self.q_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_q_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 初始同步
        # 优化器与损失函数（与 paper1.pdf 一致）
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()  # paper1.pdf 第1-106节：MSE时序差分损失

    def select_action(self, state, eps):
        """ε-greedy 选择动作（平衡探索与利用）"""
        if random.random() < eps:
            return random.choice(range(self.q_net.fc3.out_features))  # 探索：随机动作
        else:
            with torch.no_grad():
                q_values = self.q_net(state)
                return q_values.argmax().item()  # 利用：选择Q值最大的动作

    def update(self, replay_buffer, gamma=GAMMA, batch_size=BATCH_SIZE):
        """更新DQN网络（复用 paper1.pdf 公式25：时序差分损失）"""
        if len(replay_buffer) < batch_size:
            return 0.0  # 经验不足，不更新

        # 采样经验
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 计算当前Q值（主网络）
        q_current = self.q_net(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值（目标网络，冻结参数）
        with torch.no_grad():
            q_next = self.target_q_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + gamma * q_next * (1 - dones)

        # 计算损失并优化
        loss = self.criterion(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        """同步目标网络（与 paper1.pdf 目标网络更新机制一致）"""
        self.target_q_net.load_state_dict(self.q_net.state_dict())


class MA_RL:
    def __init__(self, state_dim, action_dim_per_agent=ACTION_DIM):
        """多智能体强化学习控制器：管理源链、中继、目标链智能体"""
        self.agents = {
            "source": MARL_Agent("source", state_dim, action_dim_per_agent),  # 源链智能体（调块大小）
            "relay": MARL_Agent("relay", state_dim, action_dim_per_agent),  # 中继智能体（调批处理量）
            "target": MARL_Agent("target", state_dim, action_dim_per_agent)  # 目标链智能体（暂不优化）
        }
        self.replay_buffer = ReplayBuffer()  # 共享经验回放池（多智能体协同）

    def select_actions(self, state, eps):
        """所有智能体选择动作（返回动作列表）"""
        actions = [
            self.agents["source"].select_action(state, eps),
            self.agents["relay"].select_action(state, eps),
            self.agents["target"].select_action(state, eps)  # 目标链动作暂为0（不调整）
        ]
        return actions

    def update(self):
        """更新所有智能体的DQN网络"""
        total_loss = 0.0
        for agent in self.agents.values():
            loss = agent.update(self.replay_buffer)
            total_loss += loss
        return total_loss / len(self.agents)

    def sync_targets(self):
        """同步所有智能体的目标网络"""
        for agent in self.agents.values():
            agent.sync_target()

    def save_model(self, path_prefix="MA-RL"):
        """保存模型（用于后续评估）"""
        for agent_name, agent in self.agents.items():
            torch.save(agent.q_net.state_dict(), f"{path_prefix}_{agent_name}_q_net.pth")

    def load_model(self, path_prefix="MA-RL"):
        """加载模型（用于评估最优性能）"""
        for agent_name, agent in self.agents.items():
            agent.q_net.load_state_dict(torch.load(f"{path_prefix}_{agent_name}_q_net.pth"))
            agent.target_q_net.load_state_dict(agent.q_net.state_dict())


# -------------------------- 5. 对比方法实现（静态配置、DW-DQN-Paper1）--------------------------
### 5.1 静态配置（Static Config）
def static_config_evaluate(env):
    """评估静态配置方案（固定参数，不调整）"""
    env.reset()
    total_reward = 0.0
    latency_list = []
    throughput_list = []
    done = False

    while not done:
        # 执行空动作（不调整参数）
        action = [0, 0, 0]  # 源链、中继、目标链均不调整
        next_state, reward, done = env.step(action)
        # 记录性能指标
        current_perf = env.data.iloc[env.current_idx - 1]
        latency_list.append(current_perf["latency_cross"])
        throughput_list.append(current_perf["throughput_cross"])
        total_reward += reward

    # 计算平均指标
    avg_latency = np.mean(latency_list)
    avg_throughput = np.mean(throughput_list)
    return {
        "avg_reward": total_reward / len(latency_list),
        "avg_latency": avg_latency,
        "avg_throughput": avg_throughput
    }


### 5.2 DW-DQN-Paper1（单智能体扩展版）
class DW_DQN_Paper1:
    def __init__(self, state_dim, action_dim):
        """复用 paper1.pdf DW-DQN 结构，扩展为跨链单智能体"""
        self.q_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_q_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, eps):
        """单智能体动作选择（同时调整块大小和批处理量，动作维度=ACTION_DIM^2）"""
        if random.random() < eps:
            return random.choice(range(self.q_net.fc3.out_features))
        else:
            with torch.no_grad():
                return self.q_net(state).argmax().item()

    def action_decode(self, action):
        """解码单智能体动作（映射为源链+中继动作）"""
        source_action = action // ACTION_DIM
        relay_action = action % ACTION_DIM
        source_action -= 1  # 转换为 [-1, 0, 1]
        relay_action -= 1
        return [source_action, relay_action, 0]  # 目标链动作=0

    def update(self):
        """更新DW-DQN网络（复用 paper1.pdf 公式25）"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        q_current = self.q_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            q_next = self.target_q_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + GAMMA * q_next * (1 - dones)
        loss = self.criterion(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, path="DW-DQN-Paper1.pth"):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path="DW-DQN-Paper1.pth"):
        self.q_net.load_state_dict(torch.load(path))
        self.target_q_net.load_state_dict(self.q_net.state_dict())


# -------------------------- 6. 数据预处理（遵循 paper1.pdf 标准化逻辑）--------------------------
def load_and_preprocess_env_data(data_path="HFBDP-Cross.csv", scene_label="Finance_HFT", train_ratio=0.7):
    """加载数据并预处理，划分训练/测试集（用于环境初始化）"""
    # 1. 加载数据并按时间排序
    data = pd.read_csv(data_path)
    data = data.sort_values("timestamp").reset_index(drop=True)

    # 2. 划分训练集/测试集（paper1.pdf 7:3 划分逻辑）
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    # 3. 标准化（paper1.pdf 公式10：Z-score，训练集拟合）
    state_cols = [
        "tx_arrival_rate", "block_size_current", "orderer_node_count", "latency_local",
        "latency_relay", "tx_cross_count", "latency_cross", "throughput_cross"
    ]
    # 状态+参数的总特征维度（时序特征维度=SEQ_LEN*len(state_cols) + 参数维度=2）
    total_state_dim = SEQ_LEN * len(state_cols) + 2
    # 拟合标准化器
    scaler = StandardScaler()
    # 生成训练集所有可能的状态样本（用于拟合标准化器）
    train_states = []
    for i in range(SEQ_LEN, len(train_data)):
        seq_state = train_data.iloc[i - SEQ_LEN:i][state_cols].values.flatten()
        param_state = np.array([300.0, 20.0])  # 初始参数
        train_states.append(np.concatenate([seq_state, param_state], axis=0))
    scaler.fit(np.array(train_states))

    # 4. 初始化训练/测试环境
    train_env = CrossChainEnv(train_data, scaler, scene_label=scene_label)
    test_env = CrossChainEnv(test_data, scaler, scene_label=scene_label)

    return train_env, test_env, scaler, total_state_dim


# -------------------------- 7. 训练与评估函数（遵循 paper1.pdf 实验逻辑）--------------------------
def train_marl(train_env, marl_agent, total_epochs=EPOCHS):
    """训练多智能体MA-RL"""
    eps = EPS_START
    best_reward = -float("inf")

    for epoch in range(total_epochs):
        state = train_env.reset()
        total_reward = 0.0
        total_loss = 0.0
        done = False
        step = 0

        while not done:
            # 1. 选择动作（ε-greedy衰减）
            actions = marl_agent.select_actions(state, eps)
            eps = max(EPS_END, eps - (EPS_START - EPS_END) / EPS_DECAY)

            # 2. 执行动作并获取经验
            next_state, reward, done = train_env.step(actions)
            marl_agent.replay_buffer.add(state, actions, reward, next_state, done)

            # 3. 更新智能体网络
            loss = marl_agent.update()
            total_loss += loss
            total_reward += reward
            state = next_state
            step += 1

        # 4. 每10轮同步目标网络（与 paper1.pdf 目标网络更新频率一致）
        if (epoch + 1) % 10 == 0:
            marl_agent.sync_targets()

        # 5. 保存最优模型
        avg_reward = total_reward / step
        if avg_reward > best_reward:
            best_reward = avg_reward
            marl_agent.save_model()

        print(
            f"MA-RL Epoch {epoch + 1}/{total_epochs} | Avg Reward: {avg_reward:.4f} | Avg Loss: {total_loss / step:.4f} | Eps: {eps:.3f}")

    return marl_agent


def train_dqn_paper1(train_env, dqn_agent, total_epochs=EPOCHS, action_dim=ACTION_DIM ** 2):
    """训练DW-DQN-Paper1（单智能体）"""
    eps = EPS_START
    best_reward = -float("inf")

    for epoch in range(total_epochs):
        state = train_env.reset()
        total_reward = 0.0
        total_loss = 0.0
        done = False
        step = 0

        while not done:
            # 1. 选择动作（解码为跨链参数调整）
            action = dqn_agent.select_action(state, eps)
            actions = dqn_agent.action_decode(action)
            eps = max(EPS_END, eps - (EPS_START - EPS_END) / EPS_DECAY)

            # 2. 执行动作并获取经验
            next_state, reward, done = train_env.step(actions)
            dqn_agent.replay_buffer.add(state, action, reward, next_state, done)

            # 3. 更新DQN网络
            loss = dqn_agent.update()
            total_loss += loss
            total_reward += reward
            state = next_state
            step += 1

        # 4. 同步目标网络
        if (epoch + 1) % 10 == 0:
            dqn_agent.sync_target()

        # 5. 保存最优模型
        avg_reward = total_reward / step
        if avg_reward > best_reward:
            best_reward = avg_reward
            dqn_agent.save_model()

        print(
            f"DW-DQN-Paper1 Epoch {epoch + 1}/{total_epochs} | Avg Reward: {avg_reward:.4f} | Avg Loss: {total_loss / step:.4f} | Eps: {eps:.3f}")

    return dqn_agent


def evaluate_agent(env, agent, is_marl=True):
    """评估智能体性能（MA-RL 或 DW-DQN-Paper1）"""
    env.reset()
    total_reward = 0.0
    latency_list = []
    throughput_list = []
    block_size_list = []
    batch_size_list = []
    done = False
    step = 0

    while not done:
        state = env.get_state()
        # 选择动作（无探索，ε=0）
        if is_marl:
            actions = agent.select_actions(state, eps=0.0)
        else:
            action = agent.select_action(state, eps=0.0)
            actions = agent.action_decode(action)

        # 执行动作
        next_state, reward, done = env.step(actions)
        # 记录指标
        current_perf = env.data.iloc[env.current_idx - 1]
        latency_list.append(current_perf["latency_cross"])
        throughput_list.append(current_perf["throughput_cross"])
        block_size_list.append(env.params["block_size"])
        batch_size_list.append(env.params["batch_size_relay"])
        total_reward += reward
        step += 1

    # 计算平均指标
    avg_latency = np.mean(latency_list)
    avg_throughput = np.mean(throughput_list)
    avg_block_size = np.mean(block_size_list)
    avg_batch_size = np.mean(batch_size_list)
    avg_reward = total_reward / step

    return {
        "avg_reward": avg_reward,
        "avg_latency": avg_latency,
        "avg_throughput": avg_throughput,
        "avg_block_size": avg_block_size,
        "avg_batch_size": avg_batch_size,
        "latency_list": latency_list,
        "throughput_list": throughput_list
    }


# -------------------------- 8. 主实验流程（对比所有方法）--------------------------
def run_marl_experiment(data_path="HFBDP-Cross.csv", scene_label="Finance_HFT"):
    # 1. 加载数据与初始化环境
    train_env, test_env, scaler, total_state_dim = load_and_preprocess_env_data(
        data_path=data_path, scene_label=scene_label
    )
    print(f"=== 实验场景：{scene_label} ===")
    print(f"训练数据长度：{len(train_env.data)} | 测试数据长度：{len(test_env.data)}")
    print(f"状态维度：{total_state_dim} | 动作维度（每智能体）：{ACTION_DIM}")

    # 2. 初始化对比方法
    # 2.1 静态配置（无需训练）
    print("\n=== 评估静态配置方案 ===")
    static_res = static_config_evaluate(test_env)
    print(
        f"静态配置 - 平均延迟：{static_res['avg_latency']:.4f}s | 平均吞吐量：{static_res['avg_throughput']:.4f}TPS | 平均奖励：{static_res['avg_reward']:.4f}")

    # 2.2 DW-DQN-Paper1（单智能体，动作维度=3*3=9）
    print("\n=== 训练 DW-DQN-Paper1（paper1.pdf 扩展版） ===")
    dqn_paper1 = DW_DQN_Paper1(state_dim=total_state_dim, action_dim=ACTION_DIM ** 2)
    dqn_paper1 = train_dqn_paper1(train_env, dqn_paper1)
    # 评估DW-DQN-Paper1
    print("\n=== 评估 DW-DQN-Paper1 ===")
    dqn_res = evaluate_agent(test_env, dqn_paper1, is_marl=False)
    print(
        f"DW-DQN-Paper1 - 平均延迟：{dqn_res['avg_latency']:.4f}s | 平均吞吐量：{dqn_res['avg_throughput']:.4f}TPS | 平均奖励：{dqn_res['avg_reward']:.4f}")

    # 2.3 MA-RL（多智能体，创新点）
    print("\n=== 训练 MA-RL（Ours） ===")
    marl_agent = MA_RL(state_dim=total_state_dim, action_dim_per_agent=ACTION_DIM)
    marl_agent = train_marl(train_env, marl_agent)
    # 评估MA-RL
    print("\n=== 评估 MA-RL（Ours） ===")
    marl_res = evaluate_agent(test_env, marl_agent, is_marl=True)
    print(
        f"MA-RL (Ours) - 平均延迟：{marl_res['avg_latency']:.4f}s | 平均吞吐量：{marl_res['avg_throughput']:.4f}TPS | 平均奖励：{marl_res['avg_reward']:.4f}")

    # 3. 整理对比结果（类似 paper1.pdf 表III/表IV）
    print("\n=== 所有方法对比结果汇总 ===")
    result_df = pd.DataFrame([
        {
            "Method": "Static Config",
            "Avg Latency (s)": static_res["avg_latency"],
            "Avg Throughput (TPS)": static_res["avg_throughput"],
            "Avg Reward": static_res["avg_reward"],
            "Latency Reduction (%)": 0.0,  # 基准
            "Throughput Improvement (%)": 0.0  # 基准
        },
        {
            "Method": "DW-DQN-Paper1",
            "Avg Latency (s)": dqn_res["avg_latency"],
            "Avg Throughput (TPS)": dqn_res["avg_throughput"],
            "Avg Reward": dqn_res["avg_reward"],
            "Latency Reduction (%)": (static_res["avg_latency"] - dqn_res["avg_latency"]) / static_res[
                "avg_latency"] * 100,
            "Throughput Improvement (%)": (dqn_res["avg_throughput"] - static_res["avg_throughput"]) / static_res[
                "avg_throughput"] * 100
        },
        {
            "Method": "MA-RL (Ours)",
            "Avg Latency (s)": marl_res["avg_latency"],
            "Avg Throughput (TPS)": marl_res["avg_throughput"],
            "Avg Reward": marl_res["avg_reward"],
            "Latency Reduction (%)": (static_res["avg_latency"] - marl_res["avg_latency"]) / static_res[
                "avg_latency"] * 100,
            "Throughput Improvement (%)": (marl_res["avg_throughput"] - static_res["avg_throughput"]) / static_res[
                "avg_throughput"] * 100
        }
    ])

    # 保留4位小数，按延迟降低率排序
    result_df = result_df.round(4)
    result_df = result_df.sort_values("Latency Reduction (%)", ascending=False).reset_index(drop=True)
    print(result_df)

    # 4. 保存结果（用于后续分析）
    result_df.to_csv(f"MA-RL_Experiment_Results_{scene_label}.csv", index=False)
    # 保存性能时序曲线数据
    curve_data = pd.DataFrame({
        "Timestamp": test_env.data["timestamp"].iloc[SEQ_LEN:SEQ_LEN + len(marl_res["latency_list"])],
        "Static_Latency": static_res["latency_list"],
        "DW-DQN_Latency": dqn_res["latency_list"],
        "MA-RL_Latency": marl_res["latency_list"],
        "Static_Throughput": static_res["throughput_list"],
        "DW-DQN_Throughput": dqn_res["throughput_list"],
        "MA-RL_Throughput": marl_res["throughput_list"]
    })
    curve_data.to_csv(f"MA-RL_Performance_Curve_{scene_label}.csv", index=False)

    print(f"\n实验结果已保存至：")
    print(f"- 对比表格：MA-RL_Experiment_Results_{scene_label}.csv")
    print(f"- 时序曲线：MA-RL_Performance_Curve_{scene_label}.csv")

    return result_df


# -------------------------- 9. 执行实验 --------------------------
if __name__ == "__main__":
    # 需先确保 HFBDP-Cross.csv 已生成（运行前文数据集代码）
    # 可选择不同场景（Finance_HFT/Healthcare_Data/IoT_Sensing/SupplyChain）
    run_marl_experiment(
        data_path="HFBDP-Cross.csv",
        scene_label="Finance_HFT"  # 高频交易场景（低延迟需求）
        # scene_label="Healthcare_Data"  # 医疗数据场景（高隐私需求）
        # scene_label="IoT_Sensing"  # 物联网场景（高吞吐量需求）
        # scene_label="SupplyChain"  # 供应链场景（中延迟需求）
    )