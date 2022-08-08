import json
import codecs
from difflib import SequenceMatcher
import re


def process_event_dict(k, v, mention_text, ori_text, start_span_s, role_map):
    role_per_event = []
    if not v or not v[0]:
        return []
    if k in ('cause_action', 'effect_action'):
        role_per_event = process_action_column(k, v, mention_text, start_span_s, role_map)

    elif isinstance(v, str):
        v = re.escape(v)
        if re.search(v, mention_text):
            span_start = re.search(v, mention_text).span(0)[0] + start_span_s
            span_end = span_start+len(v)-1
            role_id = role_map[k]
            role_per_event.append((span_start, span_end, role_id))
        elif re.search(v, ori_text):
            span_start = re.search(v, ori_text).span(0)[0]
            span_end = span_start+len(v)-1
            role_id = role_map[k]
            role_per_event.append((span_start, span_end, role_id))

        elif k != 'id':
            raise ValueError(f'{v} is not in {ori_text}, \n{k}')

    elif isinstance(v, list):
        for _v in v:
            _v = re.escape(_v)
            if re.search(_v, mention_text):
                span_start = re.search(_v, mention_text).span(0)[0] + start_span_s
                span_end = span_start+len(_v)-1
                role_id = role_map[k]
                role_per_event.append((span_start, span_end, role_id))
            elif re.search(_v, ori_text):
                span_start = re.search(_v, ori_text).span(0)[0]
                span_end = span_start+len(_v)-1
                role_id = role_map[k]
                role_per_event.append((span_start, span_end, role_id))
            elif k != 'id':
                raise ValueError(f'{_v} is not in {ori_text}, \n{k}')

    return role_per_event


def process_action_column(k, v, short_text, start_span_s, role_map):
    role_per_event = []
    action = v

    opcodes = SequenceMatcher(a=[_ for _ in short_text], b=[_ for _ in v], ).get_opcodes()
    action_text = ''

    for opcode in opcodes:
        if opcode[0] == 'equal':
            action_text += short_text[opcode[1]:opcode[2]]
            role_per_event.append((start_span_s + opcode[1], start_span_s + opcode[2] - 1, role_map[k]))

    # 如果出现cause_action和cause_effect_mention语序不对的情况，difflib可能会识别错误，所以需要再判断
    # cause_action: 格外重视“身后事”
    # cause_effect_mention: “死者为大”一直根植于传统观念中，因此，不少民众对于身边亲属的“身后事”处理也格外重视
    if action_text != action:
        if action_text in action:
            end = re.search(re.escape(action_text), action).span(0)[0] - 1
            if re.search(re.escape(action[0:end + 1]), short_text):
                span_start = re.search(re.escape(action[0:end + 1]), short_text).span(0)[0] + start_span_s
                role_per_event.append((span_start, span_start + end, role_map[k]))
            else:
                raise ValueError(f'{action_text} not equal {action}')
        else:
            raise ValueError(f'{action_text} not equal {action}')

    return role_per_event