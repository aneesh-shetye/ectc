
class MyCollate(): 
    
    def __init__(self, tokenizer): 
        self.tokenizer = tokenizer 
        self.pad_idx = self.tokenizer.pad_token_id
        
    def __call__(self, batch): 
        
        eng = [item['en'].lower() for item in batch] 
        de = [item['de'].lower() for item in batch] 

        eng_batch = self.tokenizer(eng, max_length=256, 
                                   padding='max_length', 
                                   truncation=True, 
                                   return_tensors='pt')['input_ids'].T
        de_batch = self.tokenizer(de, max_length=256, 
                                  padding='max_length', 
                                  truncation=True, 
                                  return_tensors='pt')['input_ids'].T

        return eng_batch, de_batch

        
