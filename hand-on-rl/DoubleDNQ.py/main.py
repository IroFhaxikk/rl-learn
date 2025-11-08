from config import Config
from model import Model
from train import Trainer
import torch
import gymnasium as gym
import ale_py
def main():
    print("🎮 初始化强化学习环境和模型...")
    
    # 1️⃣ 创建配置对象
    config = Config()

    # 2️⃣ 初始化模型
    # 注意 Model 输入通道数固定为4（因为堆叠了4帧图像）
    in_channels = 4
    n_actions = config.env.action_space.n
    model = Model(in_channels, n_actions)

    # 3️⃣ 创建训练器
    trainer = Trainer(config, Model)

    # 4️⃣ 填充经验回放（预采样）
    print("🧠 正在预填充经验回放缓冲区...")
    trainer.prefill_relaybuffer()
    print(f"✅ 经验回放已填充完成，共 {len(trainer.replay_buffer)} 条样本")

    # 5️⃣ 开始训练
    print("🚀 开始训练 Double DQN 模型！")
    trainer.train()

    # 6️⃣ 保存训练结果
    save_path = "doubledqn_model.pth"
    torch.save(trainer.q_net.state_dict(), save_path)
    print(f"💾 模型已保存到 {save_path}")


if __name__ == "__main__":
    main()
