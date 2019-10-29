from run_bert import convert_examples_to_features, read_examples
from pytorch_transformers.tokenization_bert import BertTokenizer

train_examples = read_examples('./data/train.csv', is_training=True)
train_examples = train_examples[:10]
tokenizer = BertTokenizer.from_pretrained('chinese_RoBERTa_zh_Large_pytorch', do_lower_case=True)
train_features = convert_examples_to_features(train_examples, tokenizer, 512, 3, True)
print(train_examples[0])
