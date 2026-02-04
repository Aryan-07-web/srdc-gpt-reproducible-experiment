import tensorflow as tf
from transformers import GPT2Tokenizer
import numpy as np

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

# Binary label mapping 
# Goodware -> 0
# Malware  -> 1

FEATURE_COLUMNS = [
    'apiFeatures',
    'dropFeatures',
    'regFeatures',
    'filesFeatures',
    'filesEXTFeatures',
    'dirFeatures',
    'strFeatures'
]

MAX_LEN = 1024


def encode_sample(row):
    """
    Tokenize the 7 SRDC feature types for ONE sample
    """
    texts = [row[col] if row[col] is not None else "" for col in FEATURE_COLUMNS]

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np"
    )

    return {
        "input_ids": encoded["input_ids"],           # (7, 1024)
        "attention_mask": encoded["attention_mask"]  # (7, 1024)
    }


def build_tf_dataset(dataframe, batch_size=8, shuffle=True):
    dataframe = dataframe.fillna("")

    input_ids_list = []
    attention_mask_list = []
    label_list = []

    for _, row in dataframe.iterrows():
        encoded = encode_sample(row)

        input_ids_list.append(encoded["input_ids"])
        attention_mask_list.append(encoded["attention_mask"])

        family = row["family"]
        label = 1 if (family >= 1 and family <= 11) else family // checking via the familty detection of ransomware

        label_list.append(label)

    # Convert to arrays
    input_ids = np.array(input_ids_list)          # (N, 7, 1024)
    attention_mask = np.array(attention_mask_list)
    labels = np.array(label_list, dtype=np.int32) # (N,)

    # Build tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            },
            labels
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
