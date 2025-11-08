from config import Config
from model import Model
from train import Trainer
import torch
import gymnasium as gym
import ale_py
import os

def main():
    print("🎮 初始化强化学习环境和模型...")
    
    # 1️⃣ 创建配置对象
    config = Config()
    # 添加设备检查
    print(f"🖥️  当前使用的设备: {config.device}")
    print(f"🔍 CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📊 GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # 2️⃣ 初始化模型
    in_channels = 4
    n_actions = config.env.action_space.n
    model = Model(in_channels, n_actions)

    # 3️⃣ 创建训练器
    trainer = Trainer(config, Model)  # 注意：这里传入的是类

    # 4️⃣ 智能处理经验回放缓冲区
    replay_buffer_file = "replay_buffer.pkl"
    
    # 尝试加载已有的缓冲区
    if os.path.exists(replay_buffer_file):
        print("📥 检测到已有的经验回放缓冲区文件，尝试加载...")
        if trainer.load_replay_buffer(replay_buffer_file):
            print("✅ 经验回放缓冲区加载成功！")
        else:
            print("❌ 加载失败，开始预填充...")
            trainer.prefill_relaybuffer()
    else:
        print("🧠 未找到经验回放缓冲区文件，开始预填充...")
        trainer.prefill_relaybuffer()
    
    # 5️⃣ 开始训练
    print("🚀 开始训练 Double DQN 模型！")
    
    try:
        trainer.train()
        
        # 6️⃣ 训练完成后保存缓冲区
        print("💾 训练完成，保存经验回放缓冲区...")
        trainer.save_replay_buffer(replay_buffer_file)
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断，保存当前进度...")
        # 保存模型和缓冲区
        trainer.save_replay_buffer(replay_buffer_file)
        save_path = "C:\Users\Administrator\Desktop\rl-learn\savafiles\doubledqn_model_interrupted.pth"
        torch.save(trainer.q_net.state_dict(), save_path)
        print(f"💾 中断进度已保存到 {save_path} 和 {replay_buffer_file}")

    # 7️⃣ 保存最终模型
    save_path = "doubledqn_model_final.pth"
    torch.save(trainer.q_net.state_dict(), save_path)
    print(f"💾 最终模型已保存到 {save_path}")

if __name__ == "__main__":
    main()