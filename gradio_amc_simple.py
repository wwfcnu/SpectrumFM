import gradio as gr
import pandas as pd
import numpy as np
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
    
    return processed_samples, sample_ids

def predict_signal_type(file_path):
    """预测信号类型的模拟函数"""
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
        
        # 模拟预测结果 - 实际应用中这里会调用模型
        results = []
        for sample_id in sample_ids:
            # 随机选择一个信号类型作为模拟预测结果
            pred_class_idx = np.random.randint(0, len(signal_classes))
            pred_class_name = signal_classes[pred_class_idx]
            confidence = np.random.uniform(0.7, 1.0)  # 模拟置信度
            
            results.append({
                'sample_id': sample_id,
                'predicted_signal_type': pred_class_name,
                'confidence': f"{confidence:.4f}",
                'actual_type': df[df['sample_id'] == sample_id]['type'].iloc[0] if 'type' in df.columns else 'Unknown'
            })
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
        
    except Exception as e:
        return f"Error during prediction: {str(e)}"

def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="Electromagnetic Signal Classification Demo") as demo:
        gr.Markdown("# 电磁信号分类系统")
        gr.Markdown("上传包含电磁信号数据的CSV文件，系统将预测每个样本的信号类型。")
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="上传电磁信号数据文件 (CSV格式)", file_types=[".csv"])
                submit_btn = gr.Button("开始预测", variant="primary")
            
            with gr.Column():
                result_output = gr.Dataframe(
                    label="预测结果",
                    headers=["Sample ID", "预测信号类型", "置信度", "实际类型"],
                    datatype=["str", "str", "str", "str"]
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