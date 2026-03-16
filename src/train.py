from huggingface_hub import hf_hub_download, login
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline, AutoConfig
#from src.helper import PROMPT_TEMPLATE
#from llama_cpp import Llama
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, PeftConfig
from trl import SFTTrainer
#from google.colab import userdata
import torch
import argparse
import pandas as pd



class Train_llm:
    def __init__(self) :
        self.base_model_id = "microsoft/Phi-3-mini-4k-instruct"
        self.model_id = "RaviKanur/Phi-3.5-mini-4k-instruct-text2sql"
        #self.base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        #self.model_id = "RaviKanur/TinyLlama1"
        print("Initializing Text2sql class")

    def create_chat_template(question, context, answer):
        template = f"""\
            <|system|>
            Given the context, generate an SQL query for the following question and don't provide any other extra information.<|end|>
            <|user|>
            context:{context}.
            question:{question}<|end|>
            <|assistant|>
            {answer}.<|end|>
            """
            # Remove any leading whitespace characters from each line in the template.
        template = "\n".join([line.lstrip() for line in template.splitlines()])
        return template
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, token=HUGGINGFACE_API_KEY)
        print("Downloaded the tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = "right"

        return tokenizer
    
    def load_and_prepare_dataset(self, path, category=None, train_count=0):
        print()
        data = pd.read_csv(path)
        data['text'] = data.apply(lambda x: Train_llm.create_chat_template(x["question"], x["context"], x["answer"]), axis=1)
        if category != None:
            data.fillna("train", inplace=True)
            train_data = data[data['category'] == "train"]
            test_data = data[data['category'] == "test"]
            train_data = train_data.drop('category', axis=1, inplace=True)
            test_data = test_data.drop('category', axis=1, inplace=True)
        elif train_count != 0:
            train_data = data.iloc[:train_count, :]
            test_data = data.iloc[train_count:, :]
        else:
            raise ValueError("Either category or train_count should be provided")
        return train_data, test_data
    
    def get_model(self) :
        print("Entered get_llm method")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            load_in_8bit=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=HUGGINGFACE_API_KEY,
            #attn_implementation="flash_attention_2"
        )
        print("Downloaded the model")
                        
        return model
    
    def train_model(self, model, tokenizer, train_data, test_data):
        print("Entered train_model method")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = TrainingArguments(output_dir="Hrida-sqlllm-v1", per_device_train_batch_size=4,eval_strategy="steps",
            gradient_accumulation_steps=4, optim="paged_adamw_32bit", do_eval=True,
            learning_rate=2e-4, lr_scheduler_type="cosine", save_strategy="epoch",
            gradient_checkpointing=True, overwrite_output_dir=True,
            num_train_epochs=20, bf16=True, logging_steps=10, hub_token=HUGGINGFACE_API_KEY, report_to='none',
            #max_steps=500,
            )
        
        trainer = SFTTrainer(model=model, train_dataset=train_data, eval_dataset=test_data, args=training_args, peft_config=peft_config, processing_class=tokenizer)
        
        trainer.train()

    def parse_arguements():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_id", type=str, default=None)
        parser.add_argument("--per_device_train_batch_size", type=int, default=4)
        parser.add_argument("--gradient_accumulation_steps", type=4, default=4)
        parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
        parser.add_argument("--learning_rate", type=float, default=5e-4)
        parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
        parser.add_argument("--gradient_checkpointing", type=bool, default=True)
        parser.add_argument("--num_train_epochs", type=int, default=5)
        parser.add_argument("--bf16", type=bool, 
                            default=True if torch.cuda.get_device_capability()[0] == 8 else False)
        parser.add_argument("--logging_steps", type=int, default=2)
        parser.add_argument("--report_to", type=str, default="none")



def main():
    train_llm = Train_llm()
    train_data, test_data = train_llm.load_and_prepare_dataset("./temp_1.csv", train_count=700)
    train_data_1, test_data_1 = train_llm.load_and_prepare_dataset("./temp1.csv", category="train")
    train_data = pd.concat([train_data, train_data_1])
    test_data = pd.concat([test_data, test_data_1])
    train_data_f = Dataset.from_pandas(train_data)
    test_data_f = Dataset.from_pandas(test_data)
    tokenizer = train_llm.get_tokenizer()
    model = train_llm.get_model()
    train_llm.train_model(model=model, tokenizer=tokenizer, train_data=train_data_f, test_data=test_data_f)


if __name__ == '__main__':
    main()

    
