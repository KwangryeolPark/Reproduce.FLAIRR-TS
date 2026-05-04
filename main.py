import argparse
import os
import sys
import numpy as np
import pandas as pd
from core.data_factory import data_provider
from core.agents import LLMClient, RetrievalAgent, ForecasterAgent, RefinerAgent
from core.logger import setup_exp_dir, Logger
from tqdm import tqdm

def calculate_mae(predicted, actual):
    if not predicted or not actual or len(predicted) != len(actual):
        return float('inf')
    return np.mean(np.abs(np.array(predicted) - np.array(actual)))

def main(args):
    # 0. Setup Experiment Directory and Logger
    exp_dir = setup_exp_dir(args)
    logger = Logger(exp_dir)
    
    # Redirect stdout and stderr to the logger
    sys.stdout = logger
    sys.stderr = logger
    
    print(f"Starting Experiment: {exp_dir}")

    # 1. Initialize Agents
    client = LLMClient(model_name=args.model)
    retriever = RetrievalAgent(M=args.raft_m)
    forecaster = ForecasterAgent(
        client, 
        os.path.join(args.prompt_root, "forecaster/base_prompt.txt"),
        os.path.join(args.prompt_root, "forecaster/raft_template.txt")
    )
    refiner = RefinerAgent(
        client,
        os.path.join(args.prompt_root, "refiner/refiner_system.txt"),
        os.path.join(args.prompt_root, "refiner/synthesizer_system.txt")
    )

    # 2. Load Data
    train_data = data_provider(args, flag='train')
    test_data = data_provider(args, flag='test')
    X_train_all = train_data.data_x
    
    # 3. FLAIRR-TS Iterative Loop
    sample_indices = np.random.choice(len(test_data), args.sample_size, replace=False)
    
    current_instructions = ""
    history = []
    
    best_instructions = ""
    min_overall_mae = float('inf')

    for it in range(args.max_iter):
        print(f"\n--- Iteration {it + 1} ---")
        
        batch_results = []
        total_mae = 0
        
        for idx in tqdm(sample_indices, desc="Processing Samples"):
            seq_x, seq_y = test_data[idx]
            actual_y = seq_y[args.label_len : args.label_len + args.pred_len].flatten()
            
            # Retrieval
            retrieved = retriever.retrieve(X_train_all, seq_x)
            
            # Forecast
            response = forecaster.forecast(args, seq_x, retrieved, current_instructions, logger=logger)
            preds = forecaster.parse_predictions(response)
            
            # Calculate MAE
            mae = calculate_mae(preds, actual_y)
            total_mae += mae
            
            batch_results.append({
                "predictions": preds,
                "ground_truth": actual_y.tolist(),
                "mae": mae
            })
            
        avg_mae = total_mae / len(sample_indices)
        print(f"Average MAE: {avg_mae:.4f}")
        
        history.append({
            "instructions": current_instructions,
            "mae": avg_mae
        })
        
        if avg_mae < min_overall_mae:
            min_overall_mae = avg_mae
            best_instructions = current_instructions
            
        # Refinement
        learnings, next_instructions, done = refiner.refine(
            it + 1, current_instructions, history, batch_results, logger=logger
        )
        
        print(f"Learnings: {learnings[:100]}...")
        print(f"Done: {done}")
        
        if done:
            break
            
        current_instructions = next_instructions

    print("\n" + "="*30)
    print("Optimization Finished")
    print(f"Best MAE: {min_overall_mae:.4f}")
    print(f"Best Instructions: {best_instructions}")
    print("="*30)
    
    # 3.3 Save metric.csv
    metric_df = pd.DataFrame([{"mae": min_overall_mae}])
    metric_df.to_csv(os.path.join(exp_dir, "metric.csv"), index=False)
    print(f"Metrics saved to {os.path.join(exp_dir, 'metric.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FLAIRR-TS: Forecasting LLM-Agents with Iterative Refinement and Retrieval')

    # Data
    parser.add_argument('--data', type=str, default='ETTh1', help='data name')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    
    # Forecasting lengths
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # FLAIRR-TS Hyperparameters
    parser.add_argument('--model', type=str, default='deepseek-r1:latest', help='Ollama model name')
    parser.add_argument('--max_iter', type=int, default=5, help='ts_max_iter')
    parser.add_argument('--tau_stop', type=float, default=0.05, help='ts_stopping_criteria')
    parser.add_argument('--sample_size', type=int, default=3, help='ts_sample_size')
    parser.add_argument('--raft_m', type=int, default=2, help='raft_m_retrieval')
    parser.add_argument('--prompt_root', type=str, default='./agent-promps/', help='root of prompt templates')
    
    # iTransformer compatibility
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

    args = parser.parse_args()
    main(args)
