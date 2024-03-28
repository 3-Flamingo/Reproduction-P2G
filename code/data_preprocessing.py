import pickle
from transformers import BertTokenizer
import torch

# Assuming you have installed the 'transformers' library using 'pip install transformers'
# You may need to install other required packages as well.

# Define the tokenizer and load pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('/home/chenzhen/work/BERT')

# Function to convert a single sample to BERT input format
# def convert_to_bert_input(sample):
#     events, candidates, correct_index = sample
#
#     # Combine all events into a single string
#     events_text = ' '.join([f'{event[0]} {event[1]}' for event in events])
#
#     # Tokenize the text and add special tokens
#     tokens = tokenizer.encode(events_text, add_special_tokens=True)
#
#     # Create attention mask
#     attention_mask = [1] * len(tokens)
#
#     # Convert candidates to a single string
#     candidates_text = ' '.join([f'{event[0]} {event[1]}' for event in candidates])
#
#     # Tokenize the candidates and add special tokens
#     candidate_tokens = tokenizer.encode(candidates_text, add_special_tokens=True)
#
#     # Create attention mask for candidates
#     candidate_attention_mask = [1] * len(candidate_tokens)
#
#     # Pad tokens and attention masks to the same length
#     padding_length = max(len(tokens), len(candidate_tokens))
#
#     tokens += [tokenizer.pad_token_id] * (padding_length - len(tokens))
#     attention_mask += [0] * (padding_length - len(attention_mask))
#
#     candidate_tokens += [tokenizer.pad_token_id] * (padding_length - len(candidate_tokens))
#     candidate_attention_mask += [0] * (padding_length - len(candidate_attention_mask))
#
#     # Convert lists to PyTorch tensors
#     tokens = torch.tensor(tokens)
#     attention_mask = torch.tensor(attention_mask)
#     candidate_tokens = torch.tensor(candidate_tokens)
#     candidate_attention_mask = torch.tensor(candidate_attention_mask)
#     correct_index = torch.tensor(correct_index)
#
#     return tokens, attention_mask, candidate_tokens, candidate_attention_mask, correct_index




# Assuming you have installed the 'transformers' library using 'pip install transformers'
# You may need to install other required packages as well.

# Define the tokenizer and load pre-trained BERT model

# Function to convert a single sample to the required format
def convert_to_bert_input(sample):
    transformed_data = []

    concatenated_events = []
    for event in sample[0]:
        concatenated_event = [event[0], event[3], event[4], event[5]]  # 提取动词、主语、宾语和间接宾语
        concatenated_event = ["NULL" if item is None else item for item in concatenated_event]  # 将None替换为"NULL"
        concatenated_events.extend(concatenated_event)
    # 添加候选事件
    for event in sample[1]:
        concatenated_event = [event[0], event[3], event[4], event[5]]  # 提取动词、主语、宾语和间接宾语
        concatenated_event = ["NULL" if item is None else item for item in concatenated_event]  # 将None替换为"NULL"
        concatenated_events.extend(concatenated_event)
    # 构建token_mask列表
    x = tokenizer(concatenated_events, padding=True, add_special_tokens=False, max_length=3, truncation=True)
    token_mask = x['attention_mask']
    concatenated_events = x['input_ids']

    # 遍历inputs列表中的每个元素
    for sublist in concatenated_events:
        # 遍历子列表中的每个元素
        for i in range(len(sublist)):
            # 如果子列表的长度小于3，则用0补充到长度为3
            while len(sublist) < 3:
                sublist.append(0)

    # 遍历inputs列表中的每个元素
    for sublist in token_mask:
        # 遍历子列表中的每个元素
        for i in range(len(sublist)):
            # 如果子列表的长度小于3，则用0补充到长度为3
            while len(sublist) < 3:
                sublist.append(0)

    transformed_data.append([concatenated_events, sample[2], token_mask])

    return transformed_data




train_data = pickle.load(open("/home/chenzhen/work/NEEG/data/corpus_index_train0.txt",'rb'))
dev_data = pickle.load(open("/home/chenzhen/work/NEEG/data/corpus_index_dev.txt",'rb'))
test_data = pickle.load(open("/home/chenzhen/work/NEEG/data/corpus_index_test.txt",'rb'))
# Convert all samples to BERT input format

bert_inputs = [convert_to_bert_input(sample) for sample in dev_data]
bert_inputs = [item[0] for item in bert_inputs]
# Save the processed data as a pickle file
with open('/home/chenzhen/work/P2G/data/original_basic_dev_bertdata.pickle', 'wb') as file:
    pickle.dump(bert_inputs, file)
print("write done dev_data")

bert_inputs = [convert_to_bert_input(sample) for sample in test_data]
bert_inputs = [item[0] for item in bert_inputs]
# Save the processed data as a pickle file
with open('/home/chenzhen/work/P2G/data/original_basic_test_bertdata.pickle', 'wb') as file:
    pickle.dump(bert_inputs, file)
print("write done test_data")

bert_inputs = [convert_to_bert_input(sample) for sample in train_data]
bert_inputs = [item[0] for item in bert_inputs]
# Save the processed data as a pickle file
with open('/home/chenzhen/work/P2G/data/original_basic_train_bertdata.pickle', 'wb') as file:
    pickle.dump(bert_inputs, file)
print("write done train_data")

print("write done!")