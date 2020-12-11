from model_wrappers import LibActNN

from bert_sequence_tagger.bert_utils import make_bert_tag_dict_from_flair_corpus, prepare_flair_corpus
from bert_sequence_tagger import BertForTokenClassificationCustom, SequenceTaggerBert, ModelTrainerBert
from bert_sequence_tagger.bert_utils import get_parameters_without_decay, get_model_parameters
from bert_sequence_tagger.metrics import f1_entity_level

from pytorch_transformers import BertTokenizer, AdamW

from torch.optim.lr_scheduler import ReduceLROnPlateau



def prepare_corpus(corpus):
    X, y = [], []
    for X_i, y_i in prepare_flair_corpus(corpus):
        X.append(X_i)
        y.append(y_i)
    
    return X, y


class BertTrainerWrapper:
    def __init__(self, trainer, n_epochs):
        self._trainer = trainer
        self._n_epochs = n_epochs
        
    def train(self):
        return self._trainer.train(self._n_epochs)


class LibActBertCreator:
    def __init__(self, tokenizer_name, bert_model_type, tag2idx, idx2tag, 
                 cache_dir, n_epochs, lr, bs, ebs, patience, additional_X, additional_y):
        self._bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_name, 
                                                             cache_dir=cache_dir, 
                                                             do_lower_case=('uncased' in tokenizer_name))
        self._tag2idx = tag2idx
        self._idx2tag = idx2tag
        self._cache_dir = cache_dir
        self._lr = lr
        self._n_epochs = n_epochs
        self._bs = bs
        self._ebs = ebs
        self._patience = patience
        self._bert_model_type = bert_model_type
        self._additional_X = additional_X
        self._additional_y = additional_y
    
    def __call__(self, **libact_nn_args):
        def model_ctor():
            model = BertForTokenClassificationCustom.from_pretrained(self._bert_model_type,
                                                                     cache_dir=self._cache_dir, 
                                                                     num_labels=len(self._tag2idx)).cuda()

            seq_tagger = SequenceTaggerBert(model, self._bert_tokenizer, idx2tag=self._idx2tag, 
                                            tag2idx=self._tag2idx, pred_batch_size=self._ebs)
            return seq_tagger
        
        def trainer_ctor(seq_tagger, corpus_len, train_data, val_data):
            optimizer = AdamW(get_model_parameters(seq_tagger._bert_model),
                              lr=self._lr, betas=(0.9, 0.999), 
                              eps=1e-6, weight_decay=0.01, correct_bias=True)

            lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=self._patience)

            trainer = ModelTrainerBert(model=seq_tagger, 
                                       optimizer=optimizer, 
                                       lr_scheduler=lr_scheduler,
                                       train_dataset=train_data, 
                                       val_dataset=val_data,
                                       validation_metrics=[f1_entity_level],
                                       batch_size=self._bs,
                                       update_scheduler='ee',
                                       keep_best_model=True,
                                       restore_bm_on_lr_change=True,
                                       max_grad_norm=1.,
                                       smallest_lr=self._lr/4)
            #validation_metrics=[f1_entity_level],
            #decision_metric=lambda metrics: metrics[0]
            #restore_bm_on_lr_change=False

            return BertTrainerWrapper(trainer, self._n_epochs)
        
        
        return LibActNN(model_ctor=model_ctor, 
                       trainer_ctor=trainer_ctor,
                        additional_X=self._additional_X,
                        additional_y=self._additional_y,
                       **libact_nn_args)
    