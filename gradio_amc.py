import gradio as gr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
from Model.model import ConformerClassifier
from utils import iq2ap, normalize
import warnings
warnings.filterwarnings('ignore')

# 定义信号类别映射
signal_classes = {
    0: 'QPSK',
    1: 'PAM4', 
    2: 'AM-DSB',
    3: 'GFSK',
    4: 'QAM64',
    5: 'AM-SSB',
    6: 'QAM16',
    7: '8PSK',
    8: 'WBFM',
    9: 'BPSK',
    10: 'CPFSK'
}

def parse_args():
    """模拟命令行参数，用于模型初始化"""
    parser = argparse.ArgumentParser(description="Training parameters for IQ signal processing with Transformer.")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer (default: 1e-4)')
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension of the IQ data (default: 2)')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='Maximum sequence length for the input data (default: 128)')
    parser.add_argument('--num_workers', type=int, default=12, help='Num workers for data loader')
    parser.add_argument('--model_name', type=str, default="Our", help='Num workers for data loader')
    parser.add_argument('--task_name', type=str, default="wtc", help='The task name')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
    args = parser.parse_args()
    return args

def load_model():
    """加载预训练模型"""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型实例 - 参数需要与训练时的参数一致
    model = ConformerClassifier(
        input_dim=2,        # IQ数据的输入维度
        model_dim=256,      # 模型维度
        num_heads=4,        # 注意力头数
        num_layers=16,      # 层数
        ff_hidden_dim=512,  # 前馈网络隐藏层维度
        num_classes=11,     # 信号类别数
        max_len=1024        # 最大序列长度
    ).to(device)
    
    # 加载预训练权重
    checkpoint_path = f'Checkpoint/{args.task_name}/{args.model_name}.pt'
    
    # 检查checkpoint目录是否存在，不存在则创建
    import os
    os.makedirs(f'Checkpoint/{args.task_name}', exist_ok=True)
    
    try:
        # 尝试加载模型权重
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Warning: Model checkpoint file '{checkpoint_path}' not found. Using untrained model.")
        print("Note: You need to train the model first or provide a valid checkpoint file.")
    except Exception as e:
        print(f"Warning: Error loading model: {e}. Using untrained model.")
    
    model.eval()
    return model, device

def process_signal_data(df):
    """处理信号数据，将DataFrame转换为模型可接受的格式"""
    # 按sample_id分组
    grouped = df.groupby('sample_id')
    
    processed_samples = []
    sample_ids = []
    
    for sample_id, group in grouped:
        # 提取I和Q列
        i_values = group['i'].values
        q_values = group['q'].values
        
        # 组合成IQ数据 (seq_len, 2)
        iq_data = np.column_stack([i_values, q_values])
        
        processed_samples.append(iq_data)
        sample_ids.append(sample_id)
    
    # 转换为numpy数组
    processed_samples = np.array(processed_samples)
    
    return processed_samples, sample_ids

def predict_signal_type(file_path):
    """预测信号类型的主要函数"""
    try:
        # 读取数据文件
        df = pd.read_csv(file_path)
        
        # 检查必要的列是否存在
        required_columns = ['sample_id', 'i', 'q']
        for col in required_columns:
            if col not in df.columns:
                return f"Error: Missing required column '{col}' in the data file."
        
        # 处理数据
        processed_samples, sample_ids = process_signal_data(df)
        
        # 加载模型
        model, device = load_model()
        
        # 数据预处理
        # 转置数据 (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        processed_samples = np.transpose(processed_samples, (0, 2, 1))
        
        # 应用AP转换
        processed_samples = iq2ap(processed_samples)
        
        # 归一化
        processed_samples = normalize(processed_samples)
        
        # 转换为torch tensor并移动到设备
        input_tensor = torch.FloatTensor(processed_samples).to(device)
        
        # 执行推理
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_probs = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)
        
        # 准备结果
        results = []
        for i, sample_id in enumerate(sample_ids):
            pred_class_idx = predicted_classes[i].item()
            pred_class_name = signal_classes.get(pred_class_idx, f"Unknown({pred_class_idx})")
            confidence = predicted_probs[i][pred_class_idx].item()
            
            results.append({
                'sample_id': sample_id,
                'predicted_signal_type': pred_class_name,
                'confidence': f"{confidence:.4f}"
            })
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
        
    except Exception as e:
        return f"Error during prediction: {str(e)}"

def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="Electromagnetic Signal Classification") as demo:
        gr.Markdown("# 电磁信号分类系统")
        gr.Markdown("上传包含电磁信号数据的CSV文件，系统将预测每个样本的信号类型。")
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="上传电磁信号数据文件 (CSV格式)", file_types=[".csv"])
                submit_btn = gr.Button("开始预测", variant="primary")
            
            with gr.Column():
                result_output = gr.Dataframe(
                    label="预测结果",
                    headers=["Sample ID", "预测信号类型", "置信度"],
                    datatype=["str", "str", "str"]
                )
        
        # 示例说明
        gr.Markdown("### 数据文件格式说明")
        gr.Markdown("""
        CSV文件应包含以下列：
        - `sample_id`: 样本ID
        - `i`: I路信号值
        - `q`: Q路信号值
        - `label`: 标签 (可选)
        - `snr`: 信噪比 (可选) 
        - `type`: 信号类型 (可选)
        
        同一个sample_id的多行数据构成一个样本。
        """)
        
        # 绑定事件
        submit_btn.click(
            fn=predict_signal_type,
            inputs=file_input,
            outputs=result_output
        )
    
    return demo

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)