from transformers import AutoTokenizer,AutoModelForQuestionAnswering


tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-squadv2")

model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/bert-tiny-finetuned-squadv2")