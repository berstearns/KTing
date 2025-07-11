import os
import json
import shutil
import argparse
import datetime
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Optional
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class SimpleKTDataset(Dataset):
    """Dataset class for knowledge tracing"""
    def __init__(self, data_path, max_seq_len=100):
        self.max_seq_len = max_seq_len
        self.data = self.load_and_preprocess(data_path)
        
    def load_and_preprocess(self, data_path):
        try:
            df = pd.read_csv(data_path)
            if not {'user_id', 'question_id', 'correct'}.issubset(df.columns):
                raise ValueError("CSV must contain 'user_id', 'question_id', and 'correct' columns")
                
            grouped = df.groupby('user_id')
            sequences = []
            
            for _, group in grouped:
                sequences.append({
                    'user_id': group['user_id'].values[0],
                    'question_ids': group['question_id'].values,
                    'responses': group['correct'].values
                })
            return sequences
        except Exception as e:
            raise ValueError(f"Failed to load data from {data_path}: {str(e)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        seq_len = min(len(seq['question_ids']), self.max_seq_len)
        
        return {
            'user_id': seq['user_id'],
            'question_ids': torch.LongTensor(np.pad(
                seq['question_ids'][:seq_len], 
                (0, self.max_seq_len - seq_len),
                'constant'
            )),
            'responses': torch.LongTensor(np.pad(
                seq['responses'][:seq_len],
                (0, self.max_seq_len - seq_len),
                'constant'
            )),
            'seq_len': seq_len
        }

class SimpleKTModel(nn.Module):
    """Knowledge tracing model with user and question embeddings"""
    def __init__(self, num_users, num_questions, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.question_embedding = nn.Embedding(num_questions, embedding_dim)
        self.predictor = nn.Linear(embedding_dim * 2, 2)
        
    def forward(self, user_ids, question_ids):
        user_emb = self.user_embedding(user_ids)
        question_emb = self.question_embedding(question_ids)
        combined = torch.cat([user_emb, question_emb], dim=-1)
        return F.softmax(self.predictor(combined), dim=-1)

@dataclass
class KTConfig:
    # Model parameters
    embedding_dim: int = 64
    max_seq_len: int = 100
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data parameters
    data_path: Optional[str] = None
    outputs_dir: str = "./outputs"
    create_archive: bool = False
    checkpoint_freq: int = 5  # Save every 5 epochs
    
    # Dummy data parameters
    use_dummy_data: bool = False
    num_users: int = 1000
    num_questions: int = 10000
    dummy_samples_per_user: int = 10
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.use_dummy_data and not self.data_path:
            raise ValueError("Must provide either data_path or set use_dummy_data=True")
        
        os.makedirs(self.outputs_dir, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary with validation"""
        try:
            return cls(**config_dict)
        except TypeError as e:
            raise ValueError(f"Invalid configuration: {str(e)}")
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load config from JSON file"""
        try:
            with open(json_path, 'r') as f:
                return cls.from_dict(json.load(f))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_path}: {str(e)}")
        except IOError as e:
            raise ValueError(f"Could not read {json_path}: {str(e)}")
    
    @classmethod
    def from_json_str(cls, json_str: str):
        """Load config from JSON string"""
        try:
            return cls.from_dict(json.loads(json_str))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {str(e)}")
    
    @classmethod
    def from_cli(cls):
        """Create config from command line arguments"""
        parser = argparse.ArgumentParser(description="Knowledge Tracing Model")
        
        # Model config
        parser.add_argument("--embedding_dim", type=int, default=64)
        parser.add_argument("--max_seq_len", type=int, default=100)
        
        # Training config
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--num_epochs", type=int, default=10)
        parser.add_argument("--device", type=str, 
                          default="cuda" if torch.cuda.is_available() else "cpu")
        
        # Data config
        parser.add_argument("--data_path", type=str)
        parser.add_argument("--outputs_dir", type=str, default="./outputs")
        
        # Dummy data
        parser.add_argument("--use_dummy_data", action="store_true")
        parser.add_argument("--num_users", type=int, default=1000)
        parser.add_argument("--num_questions", type=int, default=10000)
        parser.add_argument("--dummy_samples_per_user", type=int, default=10)
        
        # JSON config
        parser.add_argument("--json", type=str, 
                          help="JSON config string or path to JSON file")
        
        args = parser.parse_args()
        
        # Handle JSON input
        if args.json:
            try:
                # First try to parse as JSON string
                return cls.from_json_str(args.json)
            except ValueError:
                # If that fails, try as file path
                try:
                    return cls.from_json(args.json)
                except ValueError as e:
                    raise ValueError(f"Could not parse JSON input: {str(e)}")
        
        # Build from CLI args
        return cls.from_dict(vars(args))

def generate_dummy_data(config):
    """Generate synthetic data for testing"""
    data = {
        'user_id': np.repeat(range(config.num_users), config.dummy_samples_per_user),
        'question_id': np.random.randint(0, config.num_questions, 
                                       config.num_users * config.dummy_samples_per_user),
        'correct': np.random.randint(0, 2, 
                                   config.num_users * config.dummy_samples_per_user)
    }
    path = os.path.join(config.outputs_dir, "dummy_data.csv")
    pd.DataFrame(data).to_csv(path, index=False)
    return path

class OutputManager:
    """Manages all output resources in an organized directory structure"""
    
    def __init__(self, base_output_dir="outputs"):
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(base_output_dir)
        self.run_dir = self.base_dir / self.run_id
        
        # Create directory structure
        self.dirs = {
            'configs': self.run_dir / 'configs',
            'models': self.run_dir / 'models',
            'training': self.run_dir / 'training',
            'data': self.run_dir / 'data',
            'logs': self.run_dir / 'logs',
            'checkpoints': self.run_dir / 'training' / 'checkpoints'
        }
        
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        # Create symlink to latest run
        self.update_latest_symlink()
    
    def update_latest_symlink(self):
        """Create/update symlink to latest run"""
        latest = self.base_dir / 'latest'
        if latest.exists():
            latest.unlink()
        latest.symlink_to(self.run_dir, target_is_directory=True)
    
    def save_config(self, config, source='runtime'):
        """Save configuration files"""
        config_path = self.dirs['configs'] / f"{source}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        # Optionally save CLI args if available
        if hasattr(config, 'cli_args'):
            with open(self.dirs['configs'] / 'cli_args.txt', 'w') as f:
                f.write(" ".join(config.cli_args))
    
    def save_model(self, model, metadata=None):
        """Save model and related artifacts"""
        # Save model weights
        model_path = self.dirs['models'] / 'model.pt'
        torch.save(model.state_dict(), model_path)
        
        # Save model metadata
        if metadata is None:
            metadata = {
                'model_type': model.__class__.__name__,
                'save_time': datetime.datetime.now().isoformat()
            }
        
        with open(self.dirs['models'] / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_training_metrics(self, metrics):
        """Save training metrics and curves"""
        metrics_path = self.dirs['training'] / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Example of saving a plot (would need matplotlib)
        # self.save_learning_curve(metrics)
    
    def save_data_artifacts(self, data, name):
        """Save processed data artifacts"""
        if isinstance(data, pd.DataFrame):
            data.to_csv(self.dirs['data'] / f"{name}.csv", index=False)
        elif isinstance(data, (str, Path)) and Path(data).exists():
            shutil.copy(data, self.dirs['data'] / Path(data).name)
    
    def get_path(self, resource_type, filename):
        """Get full path for a resource"""
        return self.dirs[resource_type] / filename
    
    def archive(self, format='zip'):
        """Create archive of the run"""
        archive_path = self.base_dir / self.run_id
        return shutil.make_archive(archive_path, format, self.run_dir)

# Example Usage in Training Code
def train_model(config):
    output_mgr = OutputManager(config.outputs_dir)
    
    try:
        # 1. Save configuration
        output_mgr.save_config(config)
        
        # 2. Setup and train model
        model = SimpleKTModel(...)
        metrics = {'accuracy': [], 'loss': []}
        
        for epoch in range(config.num_epochs):
            # Training loop...
            metrics['accuracy'].append(accuracy)
            metrics['loss'].append(loss)
            
            # Optional: Save checkpoint
            if epoch % config.checkpoint_freq == 0:
                checkpoint_path = output_mgr.get_path(
                    'checkpoints', 
                    f'model_epoch{epoch}.pt'
                )
                torch.save(model.state_dict(), checkpoint_path)
        
        # 3. Save final artifacts
        output_mgr.save_model(model, {
            'num_users': num_users,
            'num_questions': num_questions,
            'embedding_dim': config.embedding_dim
        })
        
        output_mgr.save_training_metrics(metrics)
        
        if config.use_dummy_data:
            output_mgr.save_data_artifacts(dummy_data_path, 'dummy_data')
        
        # 4. Create archive (optional)
        if config.create_archive:
            output_mgr.archive()
            
        return output_mgr.run_dir
    
    except Exception as e:
        # Save error log
        with open(output_mgr.get_path('logs', 'errors.log'), 'w') as f:
            f.write(f"Training failed: {str(e)}\n")
        raise


if __name__ == "__main__":
    try:
        config = KTConfig.from_cli()
        print("Configuration loaded successfully:")
        print(json.dumps(config.__dict__, indent=2))
        
        train_model(config)
    except ValueError as e:
        print(f"Configuration error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Runtime error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)
