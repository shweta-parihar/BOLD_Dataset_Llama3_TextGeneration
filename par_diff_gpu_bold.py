import multiprocessing
import numpy as np
import transformers
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from time import time
from tqdm import tqdm
import pandas as pd
import os


# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds

def generate_response(prompt, pipeline):
    generated_text = ""
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    while True:
        outputs = pipeline(
            prompt + generated_text,
            pad_token_id=pipeline.model.config.eos_token_id,
            max_new_tokens=32,  # Generate tokens in small chunks
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        generated_text += outputs[0]["generated_text"][len(prompt + generated_text):]

        # Check if a full stop is encountered
        if '.' in generated_text:
            break

    # Combine the prompt with the generated text
    full_sentence = prompt + generated_text.split('.')[0] + '.'
    
    return full_sentence

def llm_process(df, gpu_device, file_name):
    
    time_1 = time()
    
    device = torch.device(f"cuda:{gpu_device}")
    
    config_data = json.load(open("config.json"))
    
    HF_TOKEN = config_data["HF_TOKEN"]
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              token=HF_TOKEN)
    # tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # device_map="auto",
        # device_map={cuda:gpu_device},
        device_map=device,
        quantization_config=quantization_config,
        token=HF_TOKEN
    )
    
    text_gen_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # device_map="auto",
        # device=gpu_device,
    )
    
    # if torch.cuda.is_available():
    #     gpu_count = torch.cuda.device_count()
    #     for i in range(gpu_count):
    #         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    #         print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
    #         print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
    #         print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")


    
    # Apply the function to each row in the DataFrame and save the results in a new column
    
    tqdm.pandas()  # Enable the progress bar for pandas apply
    
    
    
    # df = pd.read_parquet("df_short.parquet", engine='fastparquet')  
        
    df['answer'] = df['prompt'].progress_apply(lambda x: generate_response(x, text_gen_pipeline))

    # Save the dataframe with results
    file_name_parts = file_name.split('.')
    df.to_parquet(file_name_parts[0] + '_results_' + str(gpu_device) + '.parquet', engine='fastparquet', compression='gzip')

    time_2 = time()
    
    hours, minutes, seconds = format_time(int(time_2 - time_1))
    # output.put('Process ' + str(gpu_device) + ' finished in {hours} hours, {minutes} minutes, {seconds} seconds')
    print(f'\n------------------------------------------------------------------------\nProcess {gpu_device} finished in  ==>  {hours} hours, {minutes} mins, {seconds} secs.\nWaiting for other processes to finish ...\n------------------------------------------------------------------------\n')


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # Set start method to 'spawn'

    if torch.cuda.is_available():
        # Clear GPU cache 
        torch.cuda.empty_cache()
        num_gpus = torch.cuda.device_count()
    else:
        raise RuntimeError("No GPUs available!")

    
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")


    file_name = "df_bold_filtered.parquet"
    df = pd.read_parquet(file_name, engine='fastparquet') 
    df = df[:10]
    df_list = np.array_split(df, num_gpus)

    # Create a queue to collect results
    # output = multiprocessing.Queue()

    process_list = []
    for i in range(len(df_list)):    
        # Create processes for each list
        process_list.append(multiprocessing.Process(target=llm_process, args=(df_list[i], i, file_name)))

    time_3 = time()
    for process in process_list:
        # Start the processes
        process.start()

    for process in process_list:
        # Wait for both processes to finish
        process.join()

    time_4 = time()
    
    # for i in range(len(process_list)):
    #     # Retrieve the results from the queue
    #     print(output.get())
    
    # Print the results
    print("\n------------------------------------------------------------------------\nAll processes finished!\n------------------------------------------------------------------------\n")
    print("\nTotal time: ", (time_4-time_3))

