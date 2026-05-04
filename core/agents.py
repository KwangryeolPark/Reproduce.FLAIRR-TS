import os
import re
import sys
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, model_name="deepseek-r1:latest"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=os.getenv("OLLAMA_API_BASE"),
            api_key="ollama" # dummy key for ollama
        )

    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        # Create a summarized version of the prompt for terminal output
        display_prompt = user_prompt
        if len(user_prompt) > 1000:
            display_prompt = user_prompt[:500] + "\n... [Data Truncated for Display] ...\n" + user_prompt[-500:]
            
        sys.stdout.write(f"\n[User Input (Summarized)]:\n{display_prompt}\n")
        sys.stdout.flush()
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        full_response = ""
        is_thinking = False
        has_started_answer = False
        
        for chunk in response:
            delta = chunk.choices[0].delta
            
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                if not is_thinking:
                    sys.stdout.write("\n[Thinking Process]\n")
                    is_thinking = True
                reasoning = delta.reasoning_content
                full_response += reasoning
                sys.stdout.write(reasoning)
                sys.stdout.flush()
                
            elif delta.content:
                content = delta.content
                if "<think>" in content and not is_thinking:
                    sys.stdout.write("\n[Thinking Process]\n")
                    is_thinking = True
                    clean_content = content.replace("<think>", "")
                    if clean_content:
                        sys.stdout.write(clean_content)
                        sys.stdout.flush()
                elif "</think>" in content:
                    clean_content = content.replace("</think>", "")
                    sys.stdout.write(clean_content)
                    sys.stdout.write("\n\n[Final Answer]\n")
                    sys.stdout.flush()
                    is_thinking = False
                    has_started_answer = True
                else:
                    if not is_thinking and not has_started_answer:
                        sys.stdout.write("\n[Answer]\n")
                        has_started_answer = True
                    sys.stdout.write(content)
                    sys.stdout.flush()
                full_response += content
        
        sys.stdout.write("\n")
        return full_response

class RetrievalAgent:
    def __init__(self, M=2):
        self.M = M

    def retrieve(self, X_train, X_recent):
        L = len(X_recent)
        N = len(X_train)
        correlations = []
        for i in range(0, N - L - L, 1):
            segment = X_train[i:i+L]
            corr = np.corrcoef(X_recent.flatten(), segment.flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append((i, corr))
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, corr in correlations[:self.M]]
        retrieved_segments = []
        for idx in top_indices:
            retrieved_segments.append({
                "context": X_train[idx:idx+L],
                "outcome": X_train[idx+L : idx+L+L]
            })
        return retrieved_segments

class ForecasterAgent:
    def __init__(self, client: LLMClient, base_prompt_path, raft_template_path):
        self.client = client
        with open(base_prompt_path, 'r') as f:
            self.base_prompt_template = f.read()
        with open(raft_template_path, 'r') as f:
            self.raft_template = f.read()

    def forecast(self, args, current_window, retrieved_segments, instructions="", logger=None):
        raft_segments_str = ""
        for i, seg in enumerate(retrieved_segments):
            ctx_str = ",".join([f"{v:.4f}" for v in seg["context"].flatten()])
            out_str = ",".join([f"{v:.4f}" for v in seg["outcome"].flatten()[:args.pred_len]])
            raft_segments_str += f"Segment {i+1}:\nContext: {ctx_str}\nOutcome: {out_str}\n\n"
        
        raft_context_block = self.raft_template.format(
            raft_context="Retrieved Analogs",
            retrieved_segments_data=raft_segments_str
        )
        
        instructions_block = f"Forecasting Instructions: {instructions}" if instructions else ""
        
        user_prompt = self.base_prompt_template.format(
            target_variable=args.target,
            data_name=args.data,
            data_description=args.data,
            prediction_length=args.pred_len,
            instructions_block=instructions_block,
            raft_context_block=raft_context_block,
            previous_sequence_length_data=f"{args.seq_len} steps",
            previous_data=",".join([f"{v:.4f}" for v in current_window.flatten()])
        )
        
        # Use 1536 as a strict but sufficient limit for 96 predictions + reasoning
        response = self.client.generate("You are a helpful forecasting assistant.", user_prompt, temperature=0.0, max_tokens=1536)
        if logger:
            logger.log_agent("Forecaster", user_prompt, response)
        return response

    def parse_predictions(self, response_text):
        clean_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        match = re.search(r"Predicted Values:\s*\[(.*?)\]", clean_text, re.DOTALL)
        if match:
            values_str = match.group(1)
            cleaned_str = re.sub(r'[^0-9,.\-]', '', values_str)
            try:
                return [float(v.strip()) for v in cleaned_str.split(',') if v.strip()]
            except ValueError:
                return []
        return []

class RefinerAgent:
    def __init__(self, client: LLMClient, refiner_system_path, synthesizer_system_path):
        self.client = client
        with open(refiner_system_path, 'r') as f:
            self.refiner_system_prompt = f.read()
        with open(synthesizer_system_path, 'r') as f:
            self.synthesizer_system_prompt = f.read()

    def refine(self, iteration, current_instructions, history, samples, logger=None):
        history_str = "\n".join([f"Iter {i}: MAE={h['mae']:.4f}, Instructions: {h['instructions']}" for i, h in enumerate(history)])
        samples_str = ""
        for i, s in enumerate(samples):
            samples_str += f"Sample {i+1}:\nPredictions: {s['predictions']}\nGround Truth: {s['ground_truth']}\n\n"
        
        user_prompt = f"Iteration {iteration}\nHistory:\n{history_str}\n\nDetailed Samples:\n{samples_str}"
        refiner_system = self.refiner_system_prompt.format(it=iteration-1, current_instructions_under_review=current_instructions, mae_to_report_to_teacher=history[-1]['mae'])
        refiner_output = self.client.generate(refiner_system, user_prompt, temperature=0.3, max_tokens=2048)
        
        if logger:
            logger.log_agent("Refiner", user_prompt, refiner_output)
        
        clean_refiner_output = re.sub(r"<think>.*?</think>", "", refiner_output, flags=re.DOTALL)
        learnings_match = re.search(r"Learnings:\s*(.*?)\s*Done:", clean_refiner_output, re.DOTALL)
        done_match = re.search(r"Done:\s*(True|False)", clean_refiner_output, re.IGNORECASE)
        
        learnings = learnings_match.group(1).strip() if learnings_match else "No specific learnings."
        done = done_match.group(1).lower() == "true" if done_match else False
        
        if not done:
            synthesizer_user_prompt = f"Learnings you received:\n{learnings}"
            synthesizer_system = self.synthesizer_system_prompt.format(current_learnings=learnings)
            refined_instructions = self.client.generate(synthesizer_system, synthesizer_user_prompt, temperature=0.3, max_tokens=2048)
            if logger:
                logger.log_agent("Synthesizer", synthesizer_user_prompt, refined_instructions)
            refined_instructions = re.sub(r"<think>.*?</think>", "", refined_instructions, flags=re.DOTALL)
            refined_instructions = refined_instructions.replace("Refined Prompt Forecasting Instructions:", "").strip()
        else:
            refined_instructions = current_instructions
            
        return learnings, refined_instructions, done
