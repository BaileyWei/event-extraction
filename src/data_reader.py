from abc import ABC

from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import Field, TextField, ListField, LabelField, ArrayField, MetadataField
from allennlp.data.tokenizers import Token
from data_convert import process_event_dict

import json
import codecs
import re
from overrides import overrides
import torch
from tqdm import tqdm

role_map = {
            'cause_mention': 0,
            'cause_actor': 1,
            'cause_action': 2,
            'cause_object': 3,
            'cause_region': 4,
            'cause_industry': 5,
            'cause_organization': 6,
            'cause_product': 7,
            'effect_mention': 8,
            'effect_actor': 9,
            'effect_action': 10,
            'effect_object': 11,
            'effect_region': 12,
            'effect_industry': 13,
            'effect_organization': 14,
            'effect_product': 15
}


class CustomSpanField(Field[torch.Tensor]):
    def __init__(self,
                 span_start: int,
                 span_end: int,
                 span_label: int,
                 extra_id: int) -> None:
        self.span_start = span_start
        self.span_end = span_end
        self.span_label = span_label
        self.extra_id = extra_id

        if not isinstance(span_start, int) or not isinstance(span_end, int):
            raise TypeError(f"SpanFields must be passed integer indices. Found span indices: "
                            f"({span_start}, {span_end}) with types "
                            f"({type(span_start)} {type(span_end)})")
        if span_start > span_end:
            raise ValueError(f"span_start must be less than span_end, "
                             f"but found ({span_start}, {span_end}).")

    @overrides
    def get_padding_lengths(self):
        # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self, padding_lengths) -> torch.Tensor:
        # pylint: disable=unused-argument
        tensor = torch.LongTensor([self.span_start, self.span_end, self.span_label, self.extra_id])
        return tensor

    @overrides
    def empty_field(self):
        return CustomSpanField(-1, -1, -1, -1)

    def __str__(self) -> str:
        return f"SpanField with spans: ({self.span_start}, {self.span_end}, {self.span_label}, {self.extra_id})."

    def __eq__(self, other) -> bool:
        if isinstance(other, tuple) and len(other) == 4:
            return other == (self.span_start, self.span_end, self.span_label, self.extra_id)
        else:
            return id(self) == id(other)


class MentionReader(DatasetReader, ABC):

    def __init__(self, token_indexer, train_path='../data/train.json', valid_path='../data/valid.json'):
        super().__init__()
        self.token_indexer = token_indexer
        self.wordpiece_tokenizer = token_indexer['tokens'].wordpiece_tokenizer
        self.train_path = train_path
        self.valid_path = valid_path

    def str_2_instance(self, line):
        line = json.loads(line)
        text_id = line.get('text_id')
        ori_text = line.get('text')
        mention_text = line.get('cause_effect_mention')
        tokens = [Token(word) for word in ori_text]
        start_span_m = re.search(re.escape(mention_text), ori_text).span(0)[0]
        end_span_m = start_span_m + len(mention_text) - 1

        mention_filed = CustomSpanField(start_span_m, end_span_m, 1, 0)  # 0 represent this is mention text
        words_field = MetadataField(ori_text)
        text_id_field = MetadataField(text_id)
        sentence_field = TextField(tokens, self.token_indexer)

        fields = {'sentence': sentence_field,
                  'text_id': text_id_field,
                  'mention': mention_filed,
                  'ori_text': words_field
                  }

        return Instance(fields)

    def _read(self, file_path=None, train=True):
        if file_path:
            mention_file = file_path
        else:
            if train:
                mention_file = self.train_path
            else:
                mention_file = self.valid_path

        with codecs.open(mention_file, 'r', 'UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    instance = self.str_2_instance(line)
                    yield instance
                # 有几个mention_text没办法在ori_text找到准确的span
                except:
                    pass


class RoleReader(DatasetReader):

    def __init__(self, token_indexer, train_path='../data/train.json', valid_path='../data/valid.json'):
        super().__init__()
        self.token_indexer = token_indexer
        self.wordpiece_tokenizer = token_indexer['tokens'].wordpiece_tokenizer
        self.train_path = train_path
        self.valid_path = valid_path

    def str_2_instance(self, line):
        line = json.loads(line)
        text_id = line.get('text_id')
        ori_text = line.get('text')
        dict_id = line['cause_effect_list'][0].get('id')
        mention_text = line.get('cause_effect_mention')
        start_span_s = re.search(re.escape(mention_text), ori_text).span(0)[0]
        tokens = [Token(word) for word in ori_text]

        role_field_list = []
        for k, v in line['cause_effect_list'][0].items():
            role_per_event = process_event_dict(k, v, mention_text, ori_text, start_span_s, role_map)
            for role_tuple in role_per_event:
                role_field = CustomSpanField(role_tuple[0], role_tuple[1], role_tuple[2], 0)
                role_field_list.append(role_field)

        roles_field = ListField(role_field_list)
        text_id_field = MetadataField(text_id)
        dict_id_field = MetadataField(dict_id)
        words_field = MetadataField(ori_text)
        sentence_field = TextField(tokens, self.token_indexer)
        event_type_field = LabelField(label=0, skip_indexing=True, label_namespace='event_labels')

        fields = {'sentence': sentence_field,
                  'text_id': text_id_field,
                  'roles': roles_field,
                  'ori_text': words_field,
                  'event_type': event_type_field,
                  'dict_id': dict_id_field
                  }
        return Instance(fields)

    def _read(self, file_path, split=False):

        with codecs.open(file_path, 'r', 'UTF-8') as f:
            lines = f.readlines()
            for i,line in tqdm(enumerate(lines)):
                try:
                    instance = self.str_2_instance(line)
                    yield instance
                # 有几个role没办法在ori_text找到准确的span
                except:
                    pass

# class MentionRoleReader(DatasetReader, ABC):

if __name__ == "__main__":
    from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
    bert_indexer = {'tokens': PretrainedBertIndexer(
        pretrained_model='bert-base-chinese',
        use_starting_offsets=True,
        do_lowercase=False)}
    role_reader = RoleReader(bert_indexer)
    role_reader.read(r'../data/train.json')