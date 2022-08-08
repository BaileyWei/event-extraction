import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR # 等间隔调整学习率
from pytorch_transformers import RobertaTokenizer
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import _PyTorchLearningRateSchedulerWrapper

from model import ArgumentExtractor
from data_reader import RoleReader
from cfg import args



def train_argument_extractor(vocab, iterator, train_dataset, val_dataset):
    pretrained_bert = PretrainedBertEmbedder(
        pretrained_model=args.pretrained_bert,
        requires_grad=True,
        top_layer_only=True)

    argument_extractor = ArgumentExtractor(
        vocab=vocab,
        embedder=pretrained_bert,
        role_num=args.role_num,
        event_roles=[[_ for _ in range(args.role_num)]],
        prob_threshold=args.extractor_argument_prob_threshold,
        use_loss_weight=args.use_loss_weight).cuda(args.extractor_cuda_device)

    optimizer = optim.Adam([
        {'params': argument_extractor.embedder.parameters(), 'lr': args.extractor_embedder_lr},
        {'params': argument_extractor.start_tagger.parameters(), 'lr': args.extractor_tagger_lr},
        {'params': argument_extractor.end_tagger.parameters(), 'lr': args.extractor_tagger_lr}])
    scheduler = StepLR(
        optimizer,
        step_size=args.extractor_lr_schduler_step,
        gamma=args.extractor_lr_schduler_gamma)
    learning_rate_scheduler = _PyTorchLearningRateSchedulerWrapper(scheduler)
    # learning_rate_scheduler = scheduler
    if args.train_argument_with_generation:
        serialization_dir = os.path.join(args.extractor_generated_role_dir, str(args.extractor_generated_mask_rate))
        serialization_dir = os.path.join(serialization_dir, "x" + str(args.extractor_generated_timex))
        if args.extractor_sorted:
            serialization_dir = serialization_dir + '-sorted'
    elif args.train_argument_only_with_generation:
        serialization_dir = os.path.join(args.extractor_generated_role_dir, str(args.extractor_generated_mask_rate))
        serialization_dir = os.path.join(serialization_dir, "x" + str(args.extractor_generated_timex))
        if args.extractor_sorted:
            serialization_dir = serialization_dir + '-sorted'
        serialization_dir = serialization_dir + '-onlyed'
    else:
        serialization_dir = args.extractor_origin_role_dir

    trainer = Trainer(
        model=argument_extractor,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        num_epochs=args.extractor_epoc,
        serialization_dir=serialization_dir,
        num_serialized_models_to_keep=1,
        validation_metric='+r_c_f',  # 需要与metric中的字段一致, +/- 参照trianer中的定义
        learning_rate_scheduler=learning_rate_scheduler,
        cuda_device=args.extractor_cuda_device)
    trainer.train()
    return argument_extractor


if __name__ == '__main__':

    bert_indexer = {'tokens': PretrainedBertIndexer(
        pretrained_model='bert-base-chinese',
        use_starting_offsets=True,
        do_lowercase=False)}

    # ==== iterator =====
    vocab = Vocabulary()
    iterator = BucketIterator(
        sorting_keys=[('sentence', 'num_tokens')],
        batch_size=args.extractor_batch_size)
    iterator.index_with(vocab)

    # loading dataset
    role_reader = RoleReader(token_indexer=bert_indexer)
    role_train_dataset = role_reader.read(args.extractor_train_file)
    role_val_dataset = role_reader.read(args.extractor_val_file)

    role_extractor = train_argument_extractor(vocab, iterator, role_train_dataset, role_val_dataset)
