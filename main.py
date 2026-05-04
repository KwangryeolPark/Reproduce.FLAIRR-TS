import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from core.data_factory import data_provider
from core.agents import LLMClient, RetrievalAgent, ForecasterAgent, RefinerAgent
from core.logger import Logger

def setup_exp_dir(args):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.split('/')[-1].replace(':', '_')
    exp_name = f"{args.data}_{model_name}_{now}"
    exp_dir = os.path.join("results", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def calculate_mae(predicted, actual):
    if not predicted or len(predicted) == 0:
        return float('inf')
    actual_truncated = actual[:len(predicted)]
    return np.mean(np.abs(np.array(predicted) - np.array(actual_truncated)))

def main(args, initial_instructions=""):
    # 0. Setup Experiment Directory and Logger
    exp_dir = setup_exp_dir(args)
    logger = Logger(exp_dir)
    logger.log(f"Starting Experiment: {exp_dir}")
    
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

    # 2. Load Data (Correct Split: Train, Val, Test)
    train_data = data_provider(args, flag='train')
    val_data = data_provider(args, flag='val')
    test_data = data_provider(args, flag='test')
    X_train_all = train_data.data_x
    
    # 3. ACL Loop (Training Phase on Validation Set)
    current_instructions = initial_instructions
    history = []
    best_instructions = initial_instructions
    min_overall_mae = float('inf')

    # Calculate total windows in Validation set
    total_val_windows = len(val_data)
    
    skip_acl = False
    if args.test is not None and args.test != "auto":
        # Load prompt from specified path
        if os.path.exists(args.test):
            with open(args.test, 'r') as f:
                best_instructions = f.read().strip()
            print(f"Skipping ACL loop. Loaded best prompt from: {args.test}")
            skip_acl = True
        else:
            print(f"Warning: Test prompt path '{args.test}' not found. Starting ACL loop anyway.")

    if not skip_acl:
        # Determine sample size for training phase
        if str(args.sample_size).lower() == 'all':
            train_sample_size = total_val_windows
        else:
            train_sample_size = min(int(args.sample_size), total_val_windows)

        print(f"Starting ACL loop (Training on VAL) for {args.it} iterations with sample_size={train_sample_size}")
        
        for iteration in range(1, args.it + 1):
            print(f"\n--- Iteration {iteration} (ACL on Validation Set) ---")
            batch_results = []
            total_mae = 0
            
            # Pick random samples from VALIDATION set for refinement
            indices = np.random.choice(total_val_windows, train_sample_size, replace=False)
            
            for idx in tqdm(indices, desc=f"ACL Processing (Val)"):
                seq_x, seq_y = val_data[idx]
                actual_y = seq_y[args.label_len : args.label_len + args.pred_len].flatten()
                
                # RAFT Retrieval remains from Training set
                retrieved = retriever.retrieve(X_train_all, seq_x)
                
                # Forecast
                response = forecaster.forecast(args, seq_x, retrieved, current_instructions, logger=logger)
                preds = forecaster.parse_predictions(response, args.pred_len)
                
                mae = calculate_mae(preds, actual_y)
                total_mae += mae
                
                batch_results.append({
                    "predictions": preds,
                    "ground_truth": actual_y.tolist(),
                    "mae": mae
                })
                
            avg_mae = total_mae / train_sample_size
            print(f"Validation Average MAE: {avg_mae:.4f}")
            
            history.append({
                "instructions": current_instructions,
                "mae": avg_mae
            })
            
            if avg_mae < min_overall_mae:
                min_overall_mae = avg_mae
                best_instructions = current_instructions
                with open(os.path.join(exp_dir, "best_instructions.txt"), "w") as f:
                    f.write(best_instructions)
                print(f"New best prompt saved to {exp_dir}")
                
            # Refinement (ACL)
            learnings, next_instructions, done = refiner.refine(
                iteration, current_instructions, history, batch_results, logger=logger
            )
            
            if done:
                print("Stopping early due to convergence.")
                break
            current_instructions = next_instructions

    # 4. Final Evaluation (Test Phase on Full Test Set)
    if args.test is not None:
        total_test_windows = len(test_data)
        print("\n" + "="*30)
        print("Starting Final Evaluation on Full TEST Set")
        print(f"Using Best Instructions found on Validation Set.")
        print("="*30)
        
        final_total_mae = 0
        test_indices = range(total_test_windows)
        
        for idx in tqdm(test_indices, desc="Final Testing (Full Test Set)"):
            seq_x, seq_y = test_data[idx]
            actual_y = seq_y[args.label_len : args.label_len + args.pred_len].flatten()
            retrieved = retriever.retrieve(X_train_all, seq_x)
            
            # Use BEST prompt from Val set to predict Test set
            response = forecaster.forecast(args, seq_x, retrieved, best_instructions, logger=None)
            preds = forecaster.parse_predictions(response, args.pred_len)
            
            mae = calculate_mae(preds, actual_y)
            final_total_mae += mae

        final_avg_mae = final_total_mae / total_test_windows
        print(f"\nFinal Test Set MAE: {final_avg_mae:.4f}")
        
        pd.DataFrame([{"final_test_mae": final_avg_mae}]).to_csv(os.path.join(exp_dir, "final_metric.csv"), index=False)
    
    print("\n" + "="*30)
    print(f"Experiment Finished.")
    print(f"Best Instructions Location: {os.path.join(exp_dir, 'best_instructions.txt')}")
    print("="*30)

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
    parser.add_argument('--sample_size', type=str, default='3', help='ts_sample_size (Training Batch)')
    parser.add_argument('--raft_m', type=int, default=2, help='raft_m_retrieval')
    parser.add_argument('--prompt_root', type=str, default='./agent-promps/', help='root of prompt templates')
    
    # Evaluation Phase
    parser.add_argument('--test', type=str, nargs='?', const='auto', default=None, 
                        help='Enable testing. If path provided, loads that prompt. Otherwise uses best from Val set.')
    
    # iTransformer compatibility
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--it', type=int, default=3, help='number of iterations')
    parser.add_argument('--prompt_type', type=str, default='base', help='Initial prompt strategy (e.g., dungeon-master)')

    args = parser.parse_args()

    initial_instructions = ""
    if args.prompt_type != 'base':
        library_path = os.path.join('agent-promps', 'library', f"{args.prompt_type}.txt")
        if os.path.exists(library_path):
            with open(library_path, 'r') as f:
                initial_instructions = f.read().strip()
                initial_instructions = initial_instructions.replace("{prediction_length}", str(args.pred_len))
            print(f"Loaded initial ACL strategy: {args.prompt_type}")

    main(args, initial_instructions)
