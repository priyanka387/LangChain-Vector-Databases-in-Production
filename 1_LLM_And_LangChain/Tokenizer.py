from transformers import AutoTokenizer

#Download and load the auto tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

#rint(tokenizer.vocab)

token_ids = tokenizer.encode("This is sample text to test the tokenizer")
print("Tokens : ",  tokenizer.convert_ids_to_tokens(token_ids))
print("Token IDs : ", token_ids)
