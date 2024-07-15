import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM,BertForSequenceClassification,BertTokenizer, BertModel
from models.config.prompt_config import PromptEncoderReparameterizationType, PromptEncoderConfig
from models.prompt_encoder import PromptEncoder
# s
class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'BERT_SoftPrompt'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.safetensors'  # 模型训练结果
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')  # 设备
        self.state_dict = None
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5   # 学习率
        self.model_path = "models/bert-base-chinese"
        # tokenizer = BertTokenizer.from_pretrained("uer/gpt2-large-chinese-cluecorpussmall")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_size = 768
        self.prompt_length = 16 # 软提示词长度为16，共有16种意图
        self.hidden_dropout_prob = 0.1
        self.embedding_dim = 768
        self.encoder_hidden_size = 768
        self.num_virtual_tokens = 21128
        self.num_transformer_submodules = 1
        self.encoder_reparameterization_type = PromptEncoderReparameterizationType.LSTM
        self.inference_mode = None
        self.encoder_num_layers = 12
        self.encoder_dropout = 0.2
        self.token_dim = 768

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        # self.gpt2 = GPT2LMHeadModel.from_pretrained("uer/gpt2-large-chinese-cluecorpussmall")
        # self.model = BertModel.from_pretrained(config.model_path)
        self.model = torch.load("THUCNews/saved_dict/BERT_02_81_model.pth")
        self.config = config
        self.soft_prompt_embeds = nn.Embedding(config.num_virtual_tokens, config.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )
        self.freeze_weights()

    def freeze_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False  # 冻结权重

    def forward(self, trains):
        # 获取输入嵌入, input_ids 类型 int64
        input_ids = trains[0]
        mask = trains[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        batch_size = input_ids.size(0)
        pseudo_prompt = torch.arange(1, 64).to(self.config.device)
        soft_prompt = self.mlp(self.soft_prompt_embeds(pseudo_prompt))

        '''
        trains结构([batch_size,词嵌入向量],[batch_size],[batch_size,词嵌入向量])
        第一个元素是训练数据，第二个元素是对应训练数据的标签，第三个元素是对应训练数据的mask
        '''
        # 跟据输入的词嵌入的batch_size构建soft-prompt
        # prompt_embeddings维度[batch_size(32),soft-prompt_size(16),hidden_size(1280)]  类型 float32
        '''
        input_ids 的形状为 (batch_size, sequence_length)，其中 sequence_length 是输入序列的长度
        config.gpt2.transformer.wte() 是 GPT-2 模型的词嵌入层，将 input_ids 转换为形状为 (batch_size, sequence_length, hidden_size) 的词嵌入。
        '''
        n_prompt_embeds = soft_prompt.expand(batch_size, -1, -1)
        # [batch_size(32),sequence_length(35),hidden_size(1280)]
        input_embeddings = self.model.model.embeddings.word_embeddings(input_ids)
        inputs_ids_embeds = torch.cat([n_prompt_embeds, input_embeddings], dim=1)
        # 重新调整 attention mask
        prompt_mask = torch.ones((batch_size, n_prompt_embeds.size(1)), device=self.config.device, dtype=torch.int64)
        attention_mask = torch.cat([prompt_mask, mask], dim=1)
        outputs = self.model(trains=trains, inputs_embeds=inputs_ids_embeds, attention_mask=attention_mask,freeze=True)
        return outputs
