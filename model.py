from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn as nn


class ForMultipleAns(nn.Module):
    def __init__(self, checkpoint: str, num_labels: int):
        super(ForMultipleAns, self).__init__()
        self.model = AutoModel.from_pretrained(checkpoint,
                                               config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,
                                                                                 output_hidden_states=True))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, num_labels)  # hidden_state, num_labels
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        output = self.model(input_ids, attention_mask)
        output = output.pooler_output
        output = self.dropout(output)
        output = self.classifier(output)  # extract last hidden state from [cls] token and put into classifier
        loss = None
        if labels is not None:
            loss = self.loss(output, labels.float())

        return TokenClassifierOutput(loss=loss, logits=output)


class ForBinaryClassifier(nn.Module):
    def __init__(self, checkpoint: str):
        super(ForBinaryClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(checkpoint,
                                               config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,
                                                                                 output_hidden_states=True))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 1)  # hidden_state, num_labels
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        output = self.model(input_ids, attention_mask)
        output = output.pooler_output
        output = self.dropout(output)
        output = self.classifier(output)  # extract last hidden state from [cls] token and put into classifier
        loss = None
        if labels is not None:
            loss = self.loss(output, labels.float())

        return TokenClassifierOutput(loss=loss, logits=output)
