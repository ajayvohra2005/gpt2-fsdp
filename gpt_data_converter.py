
import argparse
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 


# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text']) 
    ids.append(enc.eot_token) 
    out = {'ids': ids, 'len': len(ids)}
    return out


class GptDataConverter:

    def __init__(self, hf_dataset:str = "openwebtext", 
                 test_size:float=0.0005, 
                 num_proc:int=8,
                 output_dir:str="data",
                 seed:int=2357) -> None:
        self.__hf_dataset = hf_dataset
        self.__num_procs = num_proc
        self.__seed = seed
        self.__test_size = test_size
        self.__output_dir = os.path.join(output_dir, hf_dataset)
        os.makedirs(self.__output_dir, exist_ok=False)
       

    def __call__(self) -> None:
        
        dataset = load_dataset(self.__hf_dataset, num_proc=self.__num_procs, trust_remote_code=True)

        # split train dataset into train and validation
        splits = dataset["train"].train_test_split(test_size=self.__test_size, seed=self.__seed, shuffle=True)
        splits['val'] = splits.pop('test')

        # tokenizer map
        tokenized_dataset = splits.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=self.__num_procs,
        )

        # concatenate token ids in each dataset split into a file
        for split, dataset_split in tokenized_dataset.items():
            token_array_len = np.sum(dataset_split['len'], dtype=np.uint64)
            filename = os.path.join(self.__output_dir, f'{split}.bin')
            dtype = np.uint16 # enc.max_token_value == 50256
            token_array_numpy_memmap = np.memmap(filename, dtype=dtype, mode='w+', shape=(token_array_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                batch = dataset_split.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                token_array_batch = np.concatenate(batch['ids'])
                token_array_numpy_memmap[idx : idx + len(token_array_batch)] = token_array_batch
                idx += len(token_array_batch)
            token_array_numpy_memmap.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='DataConverter',
                        description='Convert Hugging Face Dataset to GPT raw data',
                        epilog='Tested with openwebtext dataset')

    parser.add_argument('--dataset', help="Hugging face dataset", 
                        type=str, default="openwebtext")  
    parser.add_argument('--output-dir', help="Output directory for converted dataset", 
                        type=str, default="data")
    parser.add_argument('--test-size', help="Test split proportion > 0.0 and < 1.0", 
                        type=float, default=0.0005)
    parser.add_argument('--num-proc', help="Number of processes for Hugging Face dataset download", 
                        type=int, default=1)

    args, _ = parser.parse_known_args()

    data_converter = GptDataConverter(
        hf_dataset=args.dataset, 
        test_size=args.test_size, 
        num_proc = args.num_proc,
        output_dir=args.output_dir, )
    data_converter()