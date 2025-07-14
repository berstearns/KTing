import pandas as pd
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
