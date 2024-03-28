# coding:utf8
# This is the SGNN model described in our ijcai paper.
import os
# os.environ[ "CUDA_VISIBLE_DEVICES" ] = " 4 "
import torch.nn as nn
from torch.nn import Parameter, Module
from new_utils import *
from submodels import *
from log import Log
from transformers import AdamW, get_linear_schedule_with_warmup, BertForMaskedLM, RobertaForMaskedLM, BertTokenizer, \
    RobertaTokenizer
import time
import pdb


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    """
    query: [batch, head, seq_q, dim] Prompt
    key: [batch, head, seq_k, dim]  Chain
    value: [batch, head, seq_k, dim] Chain
    """
    d_k = query.size(-1)
    # pdb.set_trace()
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [batch, head, seq_q, seq_k]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # print(p_attn.shape, value.shape)
    out = torch.matmul(p_attn, value)  # [batch, head, seq, dim] [batch, head, seq_q, d_k]
    # print(out.shape)
    return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x = attention(query, key, value, mask=mask,
                      dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return x





class Model(Module):
    def __init__(self, args, tokenizer):
        super(Model, self).__init__()
        self.svop = args.svop
        self.max_token_len = args.max_token_len
        self.embedding_dim = args.embedding_dim
        self.batch_size = args.batch_size
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.tokenizer = tokenizer
        self.SEP_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])
        self.CLS_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])
        self.MASK_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])

        # self.yes_emb = nn.Parameter(torch.randn(self.embedding_dim))
        self.prompt_len = args.prompt_len
        # self.prompt_embeddings = nn.Parameter(torch.randn(4, self.embedding_dim))
        self.label_tokens = args.nargs

        self.attn1 = MultiHeadedAttention(h=args.heads, d_model=self.embedding_dim)
        self.attn2 = MultiHeadedAttention(h=args.heads, d_model=self.embedding_dim)

        self.model_type = args.model_type
        if self.model_type == 'BERT':
            self.model = BertForMaskedLM.from_pretrained(args.bert_path)
        elif self.model_type == 'Roberta':
            self.model = RobertaForMaskedLM.from_pretrained(args.bert_path)

        # self.label_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
        #                                 nn.ReLU(),
        #                                 nn.Dropout(args.dropout))

        self.metric = args.metric
        self.log_prob = nn.LogSoftmax(dim=-1)
        self.kl_div = nn.KLDivLoss(reduction='none', log_target=True)

        self.mean_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), 
                                          nn.ReLU()  ) 
        self.logvar_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), 
                                          nn.ReLU() )
        self.alpha = args.alpha 

    def adjust_to_svpo(self, inputs):
        """

        :param inputs: [batch_size * (4*max_token_len*13)]
        :return:
        """
        batch_size, lens = inputs.shape
        v_start, s_start, o_start, p_start = 0 * self.max_token_len, 1 * self.max_token_len, 2 * self.max_token_len, 3 * self.max_token_len
        event_len = 4 * self.max_token_len
        # pdb.set_trace()
        v_inputs_ = [inputs[:, i * event_len + v_start: i * event_len + v_start + self.max_token_len] for i in
                     range(0, 13)]  # [batch_size, 13, max_token_len]
        s_inputs_ = [inputs[:, i * event_len + s_start: i * event_len + s_start + self.max_token_len] for i in
                     range(0, 13)]
        o_inputs_ = [inputs[:, i * event_len + o_start: i * event_len + o_start + self.max_token_len] for i in
                     range(0, 13)]
        p_inputs_ = [inputs[:, i * event_len + p_start: i * event_len + p_start + self.max_token_len] for i in
                     range(0, 13)]
        v_inputs = torch.stack(v_inputs_, dim=0)  # [13, batch, max_token_len]
        s_inputs = torch.stack(s_inputs_, dim=0)  # [13, batch, max_token_len]
        o_inputs = torch.stack(o_inputs_, dim=0)  # [13, batch, max_token_len]
        p_inputs = torch.stack(p_inputs_, dim=0)  # [13, batch, max_token_len]

        new_inputs_ = torch.cat([s_inputs, v_inputs, o_inputs, p_inputs], dim=2)  # [13, batch, max_token_len*4]
        new_inputs = new_inputs_.permute(1, 0, 2).contiguous()

        return new_inputs

    def sampling(self, mean, logvar):
        # pdb.set_trace()
        eps = torch.randn(mean.shape).to(mean.device)
        samples = mean + logvar * eps
        return samples

    def forward(self, inputs, token_masks):
        """

        :param inputs: [batch_size, 52*max_token_len]
        :param token_masks: [batch_size, 52*max_token_len]
        :param targets:
        :return:
        """

        # pdb.set_trace()
        batch_size = inputs.shape[0]
        if self.svop:
            inputs = self.adjust_to_svpo(inputs)
            token_masks = self.adjust_to_svpo(token_masks)
        else:
            inputs = inputs.reshape(batch_size, 13, 4 * self.max_token_len)  # [batch_size, 13, 12]
            token_masks = token_masks.reshape(batch_size, 13, 4 * self.max_token_len)
        # pdb.set_trace()
        and_inputs = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(['and'])).to(inputs.device)
        and_mask = torch.FloatTensor([1]).to(inputs.device)
        contexts_inputs_ = torch.cat([inputs[:, :8, :].reshape(batch_size * 8, 4 * self.max_token_len),
                                      and_inputs.unsqueeze(0).expand(batch_size * 8, 1)], dim=1)
        contexts_masks_ = torch.cat([token_masks[:, :8, :].reshape(batch_size * 8, 4 * self.max_token_len),
                                     and_mask.unsqueeze(0).expand(batch_size * 8, 1)], dim=1)
        contexts_inputs_ = contexts_inputs_.reshape(batch_size, 8 * (
                    4 * self.max_token_len + 1))  # [batch_size, 32], # [batch_size * 5, 4]
        contexts_masks_ = contexts_masks_.reshape(batch_size, 8 * (4 * self.max_token_len + 1))

        candidate_inputs = inputs[:, 8:, :].reshape(batch_size * 5, 4 * self.max_token_len)
        candidate_masks = token_masks[:, 8:, :].reshape(batch_size * 5, 4 * self.max_token_len)

        cls, sep = torch.LongTensor(self.CLS_id).to(contexts_inputs_.device), torch.LongTensor(self.SEP_id).to(
            contexts_inputs_.device)
        mMASK = torch.LongTensor(self.MASK_id).to(contexts_inputs_.device)
        contexts_inputs = contexts_inputs_.unsqueeze(1).expand(batch_size, 5, 8 * (4 * self.max_token_len + 1)).reshape(
            batch_size * 5, 8 * (4 * self.max_token_len + 1))
        contexts_masks = contexts_masks_.unsqueeze(1).expand(batch_size, 5, 8 * (4 * self.max_token_len + 1)).reshape(
            batch_size * 5, 8 * (4 * self.max_token_len + 1))
        cls_inputs, cls_masks = cls.unsqueeze(0).expand(batch_size * 5, 1), torch.FloatTensor([1]).to(
            inputs.device).unsqueeze(0).expand(batch_size * 5, 1)
        sep_inputs, sep_masks = sep.unsqueeze(0).expand(batch_size * 5, 1), torch.FloatTensor([1]).to(
            inputs.device).unsqueeze(0).expand(batch_size * 5, 1)
        mMASK_inputs, mMASK_masks = mMASK.unsqueeze(0).expand(batch_size * 5, 1), torch.FloatTensor([1]).to(
            inputs.device).unsqueeze(0).expand(batch_size * 5, 1)

 
        prompt_masks = torch.FloatTensor([1] * 4).to(inputs.device).unsqueeze(0).expand(batch_size * 5, 4)

        context_embs = self.model.bert.embeddings.word_embeddings(contexts_inputs)
        candidate_embs = self.model.bert.embeddings.word_embeddings(candidate_inputs)
        cls_embs = self.model.bert.embeddings.word_embeddings(cls_inputs)
        sep_embs = self.model.bert.embeddings.word_embeddings(sep_inputs)
        mMASK_embs = self.model.bert.embeddings.word_embeddings(mMASK_inputs)

        # pdb.set_trace() 
        info_1_ = self.attn1(candidate_embs, context_embs, context_embs, contexts_masks.unsqueeze(-2)) # [batch*5, 4*3, dim]
        # pdb.set_trace()
        info_2_ = self.attn2(candidate_embs, context_embs, context_embs, contexts_masks.unsqueeze(-2)) # [batch*5, 4*3, dim]
        # pdb.set_trace()
        mean = torch.mean(info_1_.reshape(batch_size * 5, 4, self.max_token_len, self.embedding_dim), dim=2) # [batch*5, 4, dim]
        logvar_ = torch.mean(info_2_.reshape(batch_size * 5, 4, self.max_token_len, self.embedding_dim), dim=2) # [batch*5, 4, dim]
        logvar = torch.exp( logvar_ * 0.5  )
        # pdb.set_trace()
        prompt_embs = self.sampling(mean, logvar) # [batch*5, 4, dim]
        # pdb.set_trace()
        # assert (prompt_embs.shape[1] == len(prompt))

        input_embs = torch.cat([cls_embs, context_embs, prompt_embs, mMASK_embs, candidate_embs, sep_embs], dim=1)
        final_masks = torch.cat([cls_masks, contexts_masks, prompt_masks, mMASK_masks, candidate_masks, sep_masks],
                                dim=1)
        logits = self.model(inputs_embeds=input_embs, attention_mask=final_masks)[0]  # [batch_size*5, seq, vocab]
        # logits = out[0]
        # pdb.set_trace()
        mask_logits = logits[:, -(1 + 4 * self.max_token_len + 1), :]  # [batch*5, vocab]   mMASK
        mask_logits = self.dropout(mask_logits)

        label_tokens = self.label_tokens
        label_token_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(label_tokens)).to(logits.device)
        label_token_embs = self.model.bert.embeddings.word_embeddings(label_token_ids) # [n, dim]
        label_means = self.model.cls( self.mean_layer(label_token_embs) )
        label_logvars_ = self.model.cls( self.logvar_layer(label_token_embs) )
        label_logvars = torch.exp(  label_logvars_ * 0.5 )
        weight = torch.exp( - self.alpha *  label_logvars  )
        new_label_means = torch.sum(  label_means * weight      , dim=0 )
        new_label_logvars = torch.sum(    label_logvars * weight  , dim=0        )
    
        target_token_emb = self.sampling(new_label_means, new_label_logvars )


        project_yes_emb = target_token_emb.unsqueeze(0).expand(batch_size * 5, len(self.tokenizer.vocab))
        mask_log_prob = self.log_prob(mask_logits)
        yes_log_prob = self.log_prob(project_yes_emb)
        final_logits_ = -self.kl_div(mask_log_prob, yes_log_prob)
        final_logits = torch.sum(final_logits_, dim=-1).reshape(batch_size, 5)

        return final_logits


    def predict(self, input, token_masks, targets):

        logits = self.forward(input, token_masks)  # logits_: torch.Size([batch, 5])
        logit_scores = torch.softmax(logits, dim=-1)

        sorted, L = torch.sort(logit_scores, descending=True)
        num_correct0 = torch.sum((L[:, 0] == targets).type(torch.FloatTensor))
        return num_correct0


def get_optimizer(model, args, warmup_steps, num_training_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    return optimizer, scheduler


def our_eval_epoch(batch_size, model, eval_data):
    model.eval()
    acc, acc1, acc2, acc3, acc4 = [], [], [], [], []
    counter = 0
    result = []

    correct_num = 0
    while eval_data.epoch_flag:
        data = eval_data.next_batch(batch_size)
        lens = len(data[1])

        inputs, targets, masks = data[0], data[1], data[2]

        num_correct0 = model.predict(inputs, masks, targets)
        correct_num += num_correct0
        counter += lens

    real_acc = (correct_num / eval_data.corpus_length) * 100

    return real_acc

def train(tokenizer, train_data, dev_data, test_data, args):
    model = Model(args, tokenizer)
    BATCH_SIZE = args.batch_size
    ACCUM_ITER = args.accum_iter
    # ngpu = torch.cuda.device_count()
    # if ngpu > 1:
    #     model = trans_to_cuda(nn.DataParallel(model, device_ids=list(range(ngpu))))
    # elif ngpu == 1:
    model = trans_to_cuda(model)

    # num_training_steps = int(args.global_steps / args.batch_size) * args.epoches
    num_training_steps = args.global_steps
    warmup_steps = int(args.warmup_rate * num_training_steps)
    # loss_function = nn.MultiMarginLoss(margin=args.margin)
    if args.loss_func == 'ce':
        loss_function = nn.CrossEntropyLoss()
    elif args.loss_func == 'margin':
        loss_function = nn.MultiMarginLoss(margin=args.margin)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)

    embedding_parameters = [{'params': [p for p in model.attn1.parameters()]},
                            {'params': [p for p in model.attn2.parameters()]},
                            {'params': [p for p in model.mean_layer.parameters()]},
                            {'params': [p for p in model.logvar_layer.parameters()]}]
    embedding_optimizer = AdamW(embedding_parameters, lr=args.lr, eps=args.adam_epsilon)
    embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=warmup_steps,
                                                          num_training_steps=num_training_steps)

    best_acc = 0.0
    log_handler.info('start training')
    optimizer.zero_grad()
    step_counter = 0
    # for epoch in range(args.epoches):
    while step_counter < args.global_steps:
        loss_list = []
        step = 0
        while train_data.epoch_flag:
            data = train_data.next_batch(BATCH_SIZE)
            model.train()
            inputs, targets, token_masks = data[0], data[1], data[2]


            logits = model(inputs, token_masks)
            loss = loss_function(logits, targets)
            if args.margin > 0:
                # pdb.set_trace()
                softmax_logits = torch.softmax(logits, dim=-1)
                margin_loss = F.multi_margin_loss(softmax_logits, targets, margin=args.margin, reduction='sum')
                loss += margin_loss
            # ce_loss = loss_function(logits, targets)
            # loss = ce_loss + margin_loss

            loss_list.append(loss)
            loss = loss / ACCUM_ITER
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if ((step + 1) % ACCUM_ITER == 0) :

                optimizer.step()
                scheduler.step()
                embedding_optimizer.step()
                embedding_scheduler.step()
                model.zero_grad()

            if (step_counter % 50 == 0):
                log_handler.info('Global Step %d Step %d :Loss: %4f' % (step_counter, step, loss.item()))

            if (step_counter % 1000 == 0) or ((step_counter > args.eval_start) and (step_counter % args.eval_steps == 0)):
                test_data.epoch_flag = True
                real_acc = our_eval_epoch(args.batch_size, model, test_data)
                log_handler.info('Global Step %d Step %d : RealAcc: %4f' % (step_counter, step, real_acc))
                if best_acc < real_acc:
                    torch.save(model.state_dict(), "/home/chenzhen/work/P2G/models/{}.pkl".format(args.task_name))
                    # write_results(args.task_name, result, tokenizer)
                    best_acc = real_acc
                    global_best_step = step_counter
                    best_step = step
            step += 1
            step_counter += 1
            if step_counter >= args.global_steps:
                break

        # t_accuracy, t_accuracy1, t_accuracy2, t_accuracy3, t_accuracy4, _ = eval_epoch(args.batch_size, model, data, data_index=None)
        train_data.epoch_flag = True
        test_data.epoch_flag = True
        real_acc = our_eval_epoch(args.batch_size, model, test_data)
        if best_acc < real_acc:
            # write_results(args.task_name, result, tokenizer)
            torch.save(model.state_dict(), "/home/chenzhen/work/P2G/models/{}.pkl".format(args.task_name))
            best_acc = real_acc
            global_best_step = step_counter
            best_step = step

        loss = sum(loss_list) / len(loss_list)
        log_handler.info('Step %d : Loss: %f' % (step_counter, loss.item()))
        # log_handler.info('Epoch %d : Train  Acc: %f, %f, %f, %f, %f' % (epoch, t_accuracy, t_accuracy1, t_accuracy2, t_accuracy3, t_accuracy4))
        log_handler.info('Global Step  %d : Eval  RealAcc: %f' % (step_counter, real_acc))
        log_handler.info('Global Step  %d Step %d : Best  Acc: %4f' % (step_counter, global_best_step, best_acc))

        # if step_counter >= args.global_steps:
        #         break

    log_handler.info('Step  %d : Best Acc: %f' % (global_best_step, best_acc))
    return best_acc, best_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='event chain')
    parser.add_argument('--task_name', default='seqarg', type=str)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=1, type=float)
    parser.add_argument('--bert_lr', default=1e-5, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--prompt_len", default=5, type=int)
    parser.add_argument('--margin', default=0.015, type=float)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--warmup_rate', default=0.1, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--percent', default=1.0, type=float)
    parser.add_argument('--epoches', default=10, type=int)
    parser.add_argument('--max_token_len', default=3, type=int)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--embedding_dim', default=768, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--data_type', default='ours', type=str)
    # parser.add_argument('--bert_path', default='/home/data/bert-base-uncased', type=str)
    parser.add_argument('--bert_path', default='/home/chenzhen/work/BERT', type=str)
    parser.add_argument('--model_type', default='BERT', type=str)
    parser.add_argument('--eval_steps', default='2500', type=int)
    parser.add_argument('--eval_start', default=7000, type=int)
    parser.add_argument('--metric', default='dot', type=str)
    parser.add_argument('--loss_func', default='ce', type=str)
    parser.add_argument('--training', default=1, type=int)
    parser.add_argument('--model_name', default='test', type=str)
    parser.add_argument('--data_name', default='original', type=str)
    parser.add_argument('--global_steps', default='112000', type=int)
    parser.add_argument('--svop', default=0, type=int)
    parser.add_argument('--heads', default=1, type=int)
    parser.add_argument('--nargs', nargs='+')
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--accum_iter', default=1, type=float)
    args = parser.parse_args()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    log = Log(args.task_name + ".log")
    log_handler = log.getLog()
    for arg in vars(args):
        log_handler.info("{}: {}".format(arg, getattr(args, arg)))
    log_handler.info("\n")

    train_data, test_data, dev_data = load(args.batch_size, args.model_type, args.data_name, args.debug, args.percent)
    if args.model_type == 'BERT':
        tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
    elif args.model_type == 'Roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_path)
    if args.training:
        best_acc, best_epoch = train(tokenizer, train_data, dev_data, test_data,args)
    else:
        model = Model(args, tokenizer)
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        print("Num Params: ", num_params / 1e6)
        trans_to_cuda(model)
        model.load_state_dict(torch.load("../models/" + args.model_name + ".pkl"))
        model.eval()
        real_acc = our_eval_epoch(args.batch_size, model, test_data)