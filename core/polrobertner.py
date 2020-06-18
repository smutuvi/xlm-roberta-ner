import codecs
import os
import torch
from torch.utils.data.dataloader import DataLoader

from core.model.xlmr_for_token_classification import XLMRForTokenClassification
from core.utils.data_utils import InputExample, convert_examples_to_features, create_dataset


class PolRobertaNer:

    def __init__(self, model_path, roberta_embeddings_path):
        if not os.path.exists(model_path):
            raise ValueError("Model not found on path '%s'" % model_path)

        if not os.path.exists(roberta_embeddings_path):
            raise ValueError("RoBERTa language model not found on path '%s'" % roberta_embeddings_path)

        self.label_list = PolRobertaNer.load_labels(os.path.join(model_path, 'labels.txt'))
        model = XLMRForTokenClassification(pretrained_path=roberta_embeddings_path,
                                           n_labels=len(self.label_list) + 1,
                                           hidden_size=768 if 'base' in roberta_embeddings_path else 1024)
        state_dict = torch.load(open(os.path.join(model_path, 'model.pt'), 'rb'))
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    @staticmethod
    def load_labels(path):
        return [line.strip() for line in codecs.open(path, "r", "utf8").readlines() if len(line.strip()) > 0]

    def process(self, sentences):
        """
        @param sentences -- array of array of words, [['Jan', 'z', 'Warszawy'], ['IBM', 'i', 'Apple']]
        """
        examples = []
        for idx, tokens in enumerate(sentences):
            guid = str(idx)
            text_a = ' '.join(tokens)
            text_b = None
            label = ["O"] * len(tokens)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        eval_features = convert_examples_to_features(examples, self.label_list, 256, self.model.encode_word)
        eval_dataset = create_dataset(eval_features)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1)

        y_true = []
        y_pred = []

        label_map = {i: label for i, label in enumerate(self.label_list, 1)}

        for input_ids, label_ids, l_mask, valid_ids in eval_dataloader:
            with torch.no_grad():
                logits = self.model(input_ids, labels=None, labels_mask=None, valid_mask=valid_ids)

            logits = torch.argmax(logits, dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()

            for i, cur_label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []

                for j, m in enumerate(cur_label):
                    if valid_ids[i][j]:
                        temp_1.append(label_map[m])
                        temp_2.append(label_map[logits[i][j]])

                assert len(temp_1) == len(temp_2)
                y_true.append(temp_1)
                y_pred.append(temp_2)

        return y_pred
