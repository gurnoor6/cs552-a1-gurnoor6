from torch.utils.data import Dataset
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNDataset(Dataset):
    def __init__(self,
                 dataset: datasets.arrow_dataset.Dataset,
                 max_seq_length: int,
                 percentage_data: float = 0.1
                ):
        # YOUR CODE HERE
        self.max_seq_length = max_seq_length

        # defining a dictionary that simply maps tokens to their respective index in the embedding matrix
        vocab = set()
        for sent in dataset:
            for word in sent['text'].strip().split():
                vocab.add(word)
        vocab.add("<start>")
        vocab.add("<stop>")
        vocab.add("<pad>")
        vocab = sorted(vocab)
        
        # for using only 10% of data
        new_len_dataset = int(percentage_data * len(dataset))
        self.prepared_dataset = dataset[:new_len_dataset]['text']

        self.word_to_index = {word:idx for idx,word in enumerate(vocab)}
        self.index_to_word = {v:k for k,v in self.word_to_index.items()}
        
        self.pad_idx = self.word_to_index["<pad>"]

    def __len__(self):
        # YOUR CODE HERE
        return len(self.prepared_dataset)
    
    def __getitem__(self, idx):
        # YOUR CODE HERE
        tokens = self.prepared_dataset[idx].split()

        # -2 because we need to take care of start and stop as well
        tokens = tokens[:self.max_seq_length-2]
        tokens = ["<start>"] + tokens + ["<stop>"]
        token_ids = [self.word_to_index[token] for token in tokens]

        assert len(token_ids) <= self.max_seq_length

        X = token_ids + [self.pad_idx] * (self.max_seq_length - len(token_ids))
        Y = X[1:]
        return torch.tensor(X), torch.tensor(Y)
    

class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size, 
                 embedding_dim,
                 hidden_dim,
                 num_layers, 
                 dropout_rate,
                 embedding_weights=None,
                 freeze_embeddings=False):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # pass embeeding weights if exist
        if embedding_weights is not None:
            # YOUR CODE HERE
            self.embedding = embedding_weights
        else:  # train from scratch embeddings
            # YOUR CODE HERE
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        if freeze_embeddings:
          self.embedding.weight.requires_grad = False
        else:
          self.embedding.weight.requires_grad = True

        # YOUR CODE HERE
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout_rate)

        # #TODO
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_id):
        #YOUR CODE HERE
        embedded = self.dropout(self.embedding(input_id))
        outputs, hidden = self.lstm(embedded)
        return self.fc(outputs)

# Note: the following code has been taken from Exercise 3 solution
# as the assignment problem statement mentioned to refer to the 
# exercise 3
class EncoderRNN(nn.Module):
    def __init__(self, in_vocab_size, embedding_dim, hidden_dim):
        super(EncoderRNN, self).__init__()
        self.vocab_size = in_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # TODO: define embedding layer corresponding to given `vocab` and `embedding_dim` #
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # TODO: define a 1-layers, uni-directional RNN with GRU architecture #
        # feel free to use previous implemented function (get_rnn_layer)
        self.rnn = torch.nn.RNN(input_size=100,            # The number of expected features in the input x
                      hidden_size=hidden_dim,            # The size of the hidden state vector h
                      num_layers=1,                 # Number of recurrent layers. E.g., setting num_layers=2 would stack two RNNs together
                      batch_first=True,                 # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
                      bidirectional=False,           # If True, becomes a bidirectional RNN (you can play around to see what would happen :)
                      )

    def forward(self, input, hidden):
        # TODO: calculate the embedded tokens and output from rnn layer #
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        hidden = hidden.to(device)
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.randn(1, 1, self.hidden_dim, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

MAX_LENGTH = 64

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout_p=0.01, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.max_length = max_length

        # TODO: define embedding layer corresponding to given `vocab` and `embedding_dim` #
        # Think about what should be the input dimension here?
        self.embedding = nn.Embedding(self.output_dim, self.hidden_dim)

        # TODO: define attn layer to compute attention weights #
        # Think about what should be the input dimension for attn?
        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)

        # TODO: define feed-forward(linear) layer to combine information from attention layer and embedding layer #
        # Refer to the Diagram!
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = torch.nn.RNN(input_size=hidden_dim,            # The number of expected features in the input x
                      hidden_size=hidden_dim,            # The size of the hidden state vector h
                      num_layers=1,                 # Number of recurrent layers. E.g., setting num_layers=2 would stack two RNNs together
                      batch_first=True,                 # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
                      bidirectional=False,           # If True, becomes a bidirectional RNN (you can play around to see what would happen :)
                      )
        
        # TODO: define feed-forward(linear) layer to output #
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
      
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # TODO: compute attention weights #
        attn_weights = self.attn(torch.cat((embedded[0], hidden[0]), 1))
        attn_weights = F.softmax(attn_weights, dim=1)

        # TODO: multiply attention weights and contextual vector #
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.randn(1, 1, self.hidden_dim, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size, word_to_index):
        super(EncoderDecoder, self).__init__()
        # YOUR CODE HERE
        embedding_dim = 100
        self.encoder = EncoderRNN(len(word_to_index), embedding_dim, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, output_vocab_size, max_length = 64)
        self.word_to_index = word_to_index

    def forward(self, inputs, input_mask=None, targets=None):
        # YOUR CODE HERE
        encoder_hidden = self.encoder.initHidden()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        input_length = inputs.size(0)
        target_length = targets.size(0)

        encoder_outputs = torch.zeros(64, self.encoder.hidden_dim, device=device)
        decoder_outputs = torch.zeros(64, self.decoder.output_dim, device=device)
        decoder_input = torch.tensor([[self.word_to_index["<start>"]]], device=device)
        decoder_hidden = encoder_hidden

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(inputs[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[di] = decoder_output
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
        
        return decoder_outputs
