import time

import numpy as np
import pytorch_lightning as pl
import torch
from box import Box
from transformers import BertModel, BertTokenizer

MODEL_PATH = "./epoch=3.ckpt"

config = dict(
    pretrained_model_name="cl-tohoku/bert-base-japanese-whole-word-masking",
    data_module=dict(
        batch_size=16,
        max_length=32,
    ),
    model=dict(
        hidden_lf_layer=256,
        hidden_comma_period_layer=2048,
    ),
)

config = Box(config)

tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name)
tokenizer.add_tokens(["[ANS]"])


class MyModel(pl.LightningModule):
    THRESHOLD = 0.5

    def __init__(
        self,
        tokenizer,
        pretrained_model_name,
        config,
    ):
        super().__init__()
        self.config = config

        self.bert = BertModel.from_pretrained(pretrained_model_name, return_dict=True)
        self.bert.resize_token_embeddings(len(tokenizer))

        # ラインフィードの判定 二値分類
        self.hidden_lf_layer = torch.nn.Linear(
            self.bert.config.hidden_size, config.model.hidden_lf_layer
        )
        self.lf_layer = torch.nn.Linear(config.model.hidden_lf_layer, 1)

        # 挿入なし, comma, periodの判定 三値分類
        self.hidden_comma_period_layer = torch.nn.Linear(
            self.bert.config.hidden_size, config.model.hidden_comma_period_layer
        )
        self.comma_period_layer = torch.nn.Linear(
            config.model.hidden_comma_period_layer, 3
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        lf_outputs = torch.relu(self.hidden_lf_layer(outputs.pooler_output))
        lf_predictions = torch.sigmoid(self.lf_layer(lf_outputs)).flatten()

        comma_period_outputs = torch.relu(
            self.hidden_comma_period_layer(outputs.pooler_output)
        )
        comma_period_predictions = torch.softmax(
            self.comma_period_layer(comma_period_outputs), dim=1  # row
        )
        return 0, [lf_predictions, comma_period_predictions]


model = MyModel(
    tokenizer,
    pretrained_model_name=config.pretrained_model_name,
    config=config,
)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))["state_dict"]
)
model.eval()
model.freeze()

threshold = 0.5

while True:
    text = input("Text (exit): ")
    if text == "exit":
        break
    elif "[ANS]" not in text:
        print("Please input [ANS] in your text.")
        continue

    t0 = time.time()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config.data_module.max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    predictions = model(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
    )[1]
    print(f"[Time: {time.time() - t0:.2f} sec]")
    print(
        f"LF: {predictions[0].item() * 100:.3f}%, Comma: {predictions[1][0][1].item() * 100:.3f}%, Period: {predictions[1][0][2].item() * 100:.3f}%"
    )

    print(text.split("[ANS]")[0], end="")
    if np.argmax(predictions[1]) == 1:
        print("、", end="")
    elif np.argmax(predictions[1]) == 2:
        print("。", end="")
    if predictions[0] > threshold:
        print("")
    print(text.split("[ANS]")[1], end="\n\n")
