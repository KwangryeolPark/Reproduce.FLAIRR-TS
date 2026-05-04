import os
import sys
import datetime
import yaml

class Logger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.log_file_path = os.path.join(exp_dir, "log.log")
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        
        # Save stdout and stderr for redirection
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, message):
        if not message:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # We want to prepend timestamp only for new lines
        # and ensure we don't double-timestamp if message already contains newlines
        lines = message.splitlines(keepends=True)
        for line in lines:
            if not line: continue
            
            # If we are at the start of a logical output or previous write ended with newline
            # This is a bit tricky with redirection, let's simplify: 
            # If the message is a single chunk of a stream, don't timestamp it.
            # If it's a full line (ends with \n), we can timestamp it.
            
            # New approach: Use a flag to track if we need a timestamp
            if getattr(self, '_need_timestamp', True):
                formatted = f"{timestamp}-{line}"
                self._need_timestamp = False
            else:
                formatted = line
            
            if line.endswith('\n'):
                self._need_timestamp = True
                
            self.log_file.write(formatted)
            self.stdout.write(formatted)
            
        self.log_file.flush()
        self.stdout.flush()
            
    def flush(self):
        self.log_file.flush()
        self.stdout.flush()

    def log_agent(self, agent_name, user_prompt, response_text):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract think block if exists (for DeepSeek-R1)
        import re
        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        think_content = think_match.group(1).strip() if think_match else ""
        
        # Remove think block from output for cleaner logging if needed
        output_content = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        
        self.log_file.write(f"{timestamp}-[{agent_name}]:\n")
        self.log_file.write(f"\t[Input]:\t{user_prompt}\n")
        
        if think_content:
            # Split think content into lines for [Think 1], [Think 2] etc if requested, 
            # but usually one block is enough. Let's do line by line as requested.
            think_lines = think_content.split("\n")
            for i, line in enumerate(think_lines):
                if line.strip():
                    self.log_file.write(f"\t[Think {i+1}]:\t{line.strip()}\n")
        
        self.log_file.write(f"\t[Output]:\t{output_content}\n")
        self.log_file.write("\n")
        self.log_file.flush()

def setup_exp_dir(args):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.replace(":", "_").replace("/", "_")
    exp_name = f"{args.data}_{model_name}_{now}"
    exp_dir = os.path.join("results", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save args.yaml
    with open(os.path.join(exp_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)
        
    return exp_dir
