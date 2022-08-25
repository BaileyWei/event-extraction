from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import ListField, LabelField, ArrayField
from allennlp.nn import util
from allennlp.data.instance import Instance
from allennlp.data.iterators import BucketIterator

import argparse
import numpy as np
import os
import re
import torch
import pickle as pkl
from tqdm import tqdm
import codecs
import json

from model import ArgumentExtractor
from metrics import ExtractorMetric
from data_reader import RoleReader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
device_num = -1
if torch.cuda.is_available():
    device_num = torch.cuda.current_device()

print(device)

parser = argparse.ArgumentParser(description='Predict DuEE or DuEE-Fin')
# parser.add_argument('--pretrained_bert', type=str, default='BertPara/pytorch/chinese_roberta_wwm_ext')
# parser.add_argument('--bert_vocab', type=str, default='BertPara/pytorch/chinese_roberta_wwm_ext/vocab.txt')
parser.add_argument('--pretrained_bert', type=str, default='bert-base-chinese')
parser.add_argument('--bert_vocab', type=str, default='/mnt/yubai/ptmodel/bert-base-chinese/vocab.txt')
parser.add_argument('--role_num', type=int, default=16)

# parser.add_argument('--save_trigger_dir', type=str, default='./save/DuEE/bert_large/trigger/model_state_epoch_27.th')
# parser.add_argument('--save_role_dir', type=str, default='./save/DuEE/bert_large/role/model_state_epoch_29.th')

parser.add_argument('--save_role_dir', type=str,
                    default='./save/origin/role1/model_state_epoch_39.th')
parser.add_argument('--extractor_train_file', type=str, default='../data/train.json')
parser.add_argument('--extractor_val_file', type=str, default='../data/valid.json')
parser.add_argument('--extractor_test_file', type=str, default='../data/test.json')

parser.add_argument('--extractor_batch_size', type=int, default=6)
parser.add_argument('--extractor_argument_prob_threshold', type=float, default=0.5)

parser.add_argument('--task_name', type=str, default='DuEE', help="DuEE or DuEE-Fin")  # DuEE or DuEE-Fin


args = parser.parse_args()


# role 提取步骤
def argument_extractor_deal(instances, iterator, argument_model_path):
    pretrained_bert = PretrainedBertEmbedder(
        pretrained_model=args.pretrained_bert,
        requires_grad=True,
        top_layer_only=True)

    argument_extractor = ArgumentExtractor(
        vocab=Vocabulary(),
        embedder=pretrained_bert,
        role_num=args.role_num,
        event_roles=[[_ for _ in range(args.role_num)]],
        prob_threshold=args.extractor_argument_prob_threshold)

    # 加载argument_model_path下的模型
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
    # model_state = torch.load(argument_model_path, map_location=util.device_mapping(-1))
    model_state = torch.load(argument_model_path, map_location=device)
    argument_extractor.load_state_dict(model_state)
    argument_extractor.to(device)
    argument_extractor.eval()

    batch_idx = 0
    pred_spans = {}
    for data in iterator(instances, num_epochs=1):
        print(batch_idx)
        batch_idx += 1
        data = util.move_to_device(data, cuda_device=device_num)
        sentence = data['sentence']
        sentence_id = data['text_id']
        event_type = data['event_type']
        output = argument_extractor(sentence, sentence_id, event_type)
        batch_spans = argument_extractor.metric.get_span(output['start_logits'], output['end_logits'], event_type)
        argument_extractor.metric.metric(batch_spans, data['roles'])

        for idb, batch_span in enumerate(batch_spans):
            s_id = sentence_id[idb]
            if s_id not in pred_spans:
                pred_spans[s_id] = []
            pred_spans[s_id].extend(batch_span)
    metric = argument_extractor.metric.get_metric(False)
    # print(pred_spans)
    return pred_spans, metric

if __name__ == "__main__":
    import pandas as pd
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

    def get_result_dict(text_id, dict_id):
        dict = {
            "text_id": text_id,
            "cause_effect_list":[
                {"id": dict_id,
                 "cause_mention": "",
                 "cause_actor": [""],
                 "cause_action": "",
                 "cause_object": [""],
                 "cause_region": [""],
                 "cause_industry": [""],
                 "cause_organization": [""],
                 "cause_product": [""],
                 "effect_mention": "",
                 "effect_actor": [""],
                 "effect_action": "",
                 "effect_object": [""],
                 "effect_region": [""],
                 "effect_industry": [""],
                 "effect_organization": [""],
                 "effect_product": [""]}
            ]
        }
        return dict

    reverse_role_map = dict(zip(role_map.values(), role_map.keys()))

    # ==== indexer and reader =====
    bert_indexer = {'tokens': PretrainedBertIndexer(
        pretrained_model=args.bert_vocab,
        use_starting_offsets=True,
        do_lowercase=False)}

    # ==== iterator =====
    vocab = Vocabulary()
    iterator = BucketIterator(
        sorting_keys=[('sentence', 'num_tokens')],
        batch_size=args.extractor_batch_size)
    iterator.index_with(vocab)

    # trigger_model_path = args.save_trigger_dir
    argument_model_path = args.save_role_dir

    # ==== output path ====
    result_dir = "./output/" + args.task_name
    instance_pkl_path = result_dir + "/tirgger_deal_instance.pkl"
    result_pkl_path = result_dir + "/role_result.pkl"
    output_file = result_dir + "/" + args.task_name + "_result.json"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # ==== predict ====

    role_reader = RoleReader(token_indexer=bert_indexer)
    dataset = role_reader.read(args.extractor_train_file, train=False)  # 用来预测的文件


    print('=====> Extracting arguments...')
    pred_spans,metric = argument_extractor_deal(instances=dataset, iterator=iterator,
                                             argument_model_path=argument_model_path)
    print(metric)


    # ==== output to json ====
    print('=====> output to json files: ', output_file)

    id_sentence = {}  # sid 与sentence 对应关系
    for data in dataset:
        id_sentence[data['text_id'].metadata] = (data['ori_text'].metadata, data['dict_id'].metadata)
    result = []
    with codecs.open(output_file, 'w', 'UTF-8') as f:
        for sid, pred_span in pred_spans.items():
            text = id_sentence[sid][0]
            dict_id = id_sentence[sid][1]
            dict = get_result_dict(sid, dict_id)
            # print(text)
            tmp_elist = []
            for ids, span in enumerate(pred_span):
                e_dict = {}
                argument = text[span[0]: span[1] + 1]
                role = reverse_role_map.get(span[2])
                r_dict = dict['cause_effect_list'][0]
                if isinstance(r_dict[role], list):
                    if not r_dict[role][0]:
                        r_dict[role][0] = argument
                    else:
                        r_dict[role].append(argument)
                else:
                    r_dict[role] = argument

            result.append(str(dict).split(', '))
    re = pd.DataFrame(result)
    re.to_excel('valid_result.xlsx')
            # tmp = json.dumps(tmp, ensure_ascii=False)
            # f.write(tmp + "\n")


    '''
    # DuEE-Fin 篇章级文本处理方案
    if args.task_name == "DuEE-Fin":
        id_sentence = {} # sent_id 与 origin_text 对应关系
        sentid_textid = {} # sent_id 与 text_id 对应关系, sent_id由内容而定, 不唯一
        for data in pre_dataset:
            if data['sentence_id'].metadata in sentid_textid:
                sentid_textid[data['sentence_id'].metadata].append(data['text_id'].metadata)
            else:
                sentid_textid[data['sentence_id'].metadata] = [data['text_id'].metadata]

            id_sentence[data['sentence_id'].metadata] = data['origin_text'].metadata

        with codecs.open(output_file, 'w', 'UTF-8') as f:
            textid_result = {} # {id: event_list}
            for sid, pred_span in pred_spans.items():
                text = id_sentence[sid]
                # text_id = sentid_textid[sid]
                tmp_elist = []
                e_set = set()
                role_dict = {}
                for ids, span in enumerate(pred_span):
                    et = data_meta.get_event_type_name(span[3])
                    role_info = {
                        "role": data_meta.get_role_name(span[2]),
                        "argument": text[span[0]: span[1] + 1]
                    }

                    tmp_elist.append({
                        "event_type": et,
                        "arguments": [role_info]
                    })

                    # 将出现的所有et与role进行存储, 百度官方baseline的postprocess方法
                    e_set.add(et)
                    if role_info['role'] in role_dict:
                        role_dict[role_info['role']].append(role_info)
                    else:
                        role_dict[role_info['role']] = [role_info]

                for text_id in sentid_textid[sid]:
                    # 相同text_id 结果进行合并
                    if text_id in textid_result:
                        textid_result[text_id].extend(tmp_elist)
                    else:
                        textid_result[text_id] = tmp_elist

            # 将结果组合成答案
            for text_id, event_list in textid_result.items():
                tmp = {}
                tmp['id'] = text_id

                # 将event_type相同的结果进行合并
                et_set_dict = {} # 用于et 中的role 去重, 百度官方会进行去重
                et_roles_dict = {} # 用于相同et的role合并
                for e in event_list:
                    tmp_et = e['event_type']
                    # 遇到了新的事件类型
                    if tmp_et not in et_roles_dict:
                        et_roles_dict[tmp_et] = []
                        et_set_dict[tmp_et] = set()
                    for rl in e['arguments']:
                        if rl["role"] + "-" + rl["argument"] not in et_set_dict[tmp_et]:
                            et_roles_dict[tmp_et].append(rl)
                            et_set_dict[tmp_et].add(rl["role"] + "-" + rl["argument"])

                event_list = [{"event_type": et, "arguments": roles} for et, roles in et_roles_dict.items()]

                tmp['event_list'] = event_list

                tmp = json.dumps(tmp, ensure_ascii=False)
                f.write(tmp + "\n")
    '''