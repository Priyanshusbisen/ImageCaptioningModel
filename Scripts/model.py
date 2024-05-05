import torch
import torchvision.models as models
import torch.nn as nn


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN : bool = False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        if self.training:
            features, _ = self.inception(images)
        else:
            features = self.inception(images)

        for name, param in self.inception.named_parameters():
            #Train only the fully connectued layer and not other layers since we are finetuning
            if 'fe.weight' in name or 'fc.bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN

        features = self.dropout(self.Relu(features))
        return features 


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layer: int):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layer)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, input):
        embeddings = self.dropout(self.embedding(input))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hidden, _ = self.lstm(embeddings)
        output = self.linear(hidden)
        return output

class Seq2Seq(nn.Module):
    def __init__(self, embed_dim, vocab_size, hidden_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderRNN(vocab_size, embed_dim, hidden_size, num_layers)

    def forward(self, images, input):
        features = self.encoder(images)
        output = self.decoder(features, input)

        return output
    
    def generate_caption(self, images, vocabulary, max_len: int = 50):
        result_caption = []
        
        with torch.no_grad():
            x = self.encoder(images).unsqueeze(0)
            states = None
            for _ in range(max_len):
                hidden, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hidden.squeeze(0))
                
                prediction = output.argmax(1)

                result_caption.append(prediction.item())

                if vocabulary.itos[prediction.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]