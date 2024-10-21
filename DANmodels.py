import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sentiment_data import read_sentiment_examples, read_word_embeddings, WordEmbeddings
from torch.utils.data import Dataset
from utils import Indexer


# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, vectorizer=None, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        
        # Extract sentences and labels from the examples
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        
        # Vectorize the sentences using CountVectorizer
        if train or vectorizer is None:
            self.vectorizer = read_word_embeddings("data/glove.6B.300d-relativized.txt")
            self.embeddings = []
            for sentence in self.sentences:
                sentence_embedding = []
                for word in sentence:
                    sentence_embedding.append(self.vectorizer.get_embedding(word))
                sentence_embedding = [sum(x) / len(x) for x in zip(*sentence_embedding)]
                self.embeddings.append(sentence_embedding)

        else:
            self.vectorizer = vectorizer
            self.embeddings = []
            for sentence in self.sentences:
                sentence_embedding = []
                for word in sentence:
                    sentence_embedding.append(self.vectorizer.get_embedding(word))
                sentence_embedding = [sum(x) / len(x) for x in zip(*sentence_embedding)]
                self.embeddings.append(sentence_embedding)
        
        # Convert embeddings and labels to PyTorch tensors
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.embeddings[idx], self.labels[idx]


class RandInitDAN(Dataset):
    def __init__(self, infile, embedfile):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        self.word_idxs = Indexer()
        self.word_idxs.add_and_get_index("PAD")
        self.word_idxs.add_and_get_index("UNK")

        # Add words from the embeddings file to the indexer
        with open(embedfile, 'r') as f:
            for line in f:
                word = line.split()[0]
                self.word_idxs.add_and_get_index(word)

        # Extract sentences and labels from the examples
        self.sentence_idxs = [[self.word_idxs.index_of(word) if self.word_idxs.contains(word) else self.word_idxs.index_of("UNK") for word in ex.words] for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def collate_function(self, batch):
        features, labels = zip(*batch)
        padded_data = pad_sequence(features, batch_first=True)
        return torch.tensor(padded_data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return torch.tensor(self.sentence_idxs[idx], dtype=torch.long), self.labels[idx]


class NN2DAN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.log_softmax(x)
        return x

# Three-layer fully connected neural network with Dropout and Batch Normalization
class NN3DAN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        return self.log_softmax(x)


class RandDAN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.long()
        avg_embedding = self.embedding(x).mean(dim=1)
        x = F.relu(self.fc1(avg_embedding))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
