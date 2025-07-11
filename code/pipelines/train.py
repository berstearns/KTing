import os
import json
import shutil
import argparse
import datetime
from pathlib import Path
import sys
from dataclasses import dataclass, asdict, fields
from typing import Optional, Dict, Any
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
    checkpoint_freq: int = 5
    
    # Data parameters
    data_path: Optional[str] = None
    outputs_dir: str = "./KT_outputs"
    create_archive: bool = False
    
    # Dummy data parameters
    use_dummy_data: bool = False
    num_users: int = 1000
    num_questions: int = 10000
    dummy_samples_per_user: int = 10

    def __post_init__(self):
        """Simple validation"""
        if not self.use_dummy_data and not self.data_path:
            raise ValueError("Must provide either data_path or set use_dummy_data=True")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create from dictionary - simplest possible version"""
        return cls(**config_dict)

    @classmethod
    def from_cli(cls):
        """Handle all input methods simply and reliably"""
        parser = argparse.ArgumentParser()
        
        # Just two options: JSON input or normal CLI args
        parser.add_argument("--json", type=str, help="JSON config string or file path")
        
        # Regular parameters as optional overrides
        parser.add_argument("--use_dummy_data", action="store_true")
        parser.add_argument("--num_users", type=int)
        parser.add_argument("--num_questions", type=int)
        # [Add other parameters similarly...]
        
        args = parser.parse_args()
        
        # Start with empty config
        config = {}
        
        # Load from JSON if provided (file or string)
        if args.json:
            try:
                if Path(args.json).exists():
                    with open(args.json) as f:
                        config.update(json.load(f))
                else:
                    config.update(json.loads(args.json))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")
        
        # Override with any explicit CLI args
        for field in fields(cls):
            if (val := getattr(args, field.name, None)) is not None:
                config[field.name] = val
        
        return cls.from_dict(config)


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
        """Save configuration files with proper type conversion"""
        config_dict = config.__dict__.copy()
        
        # Convert numpy types to native Python types
        for key, value in config_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                config_dict[key] = int(value) if isinstance(value, np.integer) else float(value)
        
        config_path = self.dirs['configs'] / f"{source}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)  # Added default=str for other non-serializable types
    
    
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
        """Save training metrics with proper type conversion"""
        metrics_dict = metrics.copy()
        
        # Convert numpy types in metrics
        for key in metrics_dict:
            if isinstance(metrics_dict[key], list):
                metrics_dict[key] = [
                    float(x) if isinstance(x, (np.floating, np.integer)) else x 
                    for x in metrics_dict[key]
                ]
    
        metrics_path = self.dirs['training'] / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
    
    
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


def train_model(config):
    output_mgr = OutputManager(config.outputs_dir)
    
    try:
        # 1. Save configuration
        output_mgr.save_config(config)
        
        # 2. Setup data and determine model dimensions
        if config.use_dummy_data:
            data_path = generate_dummy_data(config)
        else:
            data_path = config.data_path
            
        dataset = SimpleKTDataset(data_path, config.max_seq_len)
        
        # Calculate number of unique users and questions
        num_users = max(seq['user_id'] for seq in dataset.data) + 1
        num_questions = max(max(seq['question_ids']) for seq in dataset.data) + 1

        # 3. Initialize model with proper dimensions
        model = SimpleKTModel(
            num_users=num_users,
            num_questions=num_questions,
            embedding_dim=config.embedding_dim
        ).to(config.device)
        
        # 4. Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        metrics = {'accuracy': [], 'loss': []}
        
        # 5. Training loop
        for epoch in range(config.num_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                # Prepare batch
                user_ids = batch['user_id'].to(config.device)
                question_ids = batch['question_ids'].to(config.device)
                responses = batch['responses'].to(config.device)
                seq_lens = batch['seq_len']
                
                # Create mask for valid positions
                mask = torch.arange(config.max_seq_len, device=config.device)[None, :] < seq_lens[:, None]
                mask = mask.view(-1)
                
                # Forward pass
                outputs = model(
                    user_ids.repeat_interleave(config.max_seq_len)[mask],
                    question_ids.view(-1)[mask]
                )
                loss = criterion(outputs, responses.view(-1)[mask])
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += mask.sum().item()
                correct += (predicted == responses.view(-1)[mask]).sum().item()
            
            epoch_acc = correct / total
            epoch_loss = total_loss / len(dataloader)
            metrics['accuracy'].append(epoch_acc)
            metrics['loss'].append(epoch_loss)
            
            print(f"Epoch {epoch+1}/{config.num_epochs}: "
                  f"Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % config.checkpoint_freq == 0:
                checkpoint_path = output_mgr.get_path(
                    'checkpoints', 
                    f'model_epoch{epoch+1}.pt'
                )
                torch.save(model.state_dict(), checkpoint_path)
        
        # 6. Save final artifacts
        output_mgr.save_model(model, {
            'num_users': num_users,
            'num_questions': num_questions,
            'embedding_dim': config.embedding_dim
        })
        
        output_mgr.save_training_metrics(metrics)
        
        if config.use_dummy_data:
            output_mgr.save_data_artifacts(data_path, 'dummy_data')
        
        if config.create_archive:
            output_mgr.archive()
            
        return output_mgr.run_dir
    
    except Exception as e:
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
