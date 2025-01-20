from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertModel

class BertEmbedder:
    """
    A class used to generate embeddings from text using the DistilBERT model.

    Attributes
    ----------
    device : str
        The device to run the model on ('cpu' or 'cuda').
    model_ckpt : str
        The checkpoint name of the pre-trained DistilBERT model.
    max_length : int
        The maximum length of the tokenized input sequences.
    padding : bool
        Whether to pad the input sequences to the maximum length.
    truncation : bool
        Whether to truncate the input sequences to the maximum length.
    tokenizer : DistilBertTokenizer
        The tokenizer for the DistilBERT model.
    model : DistilBertModel
        The DistilBERT model.

    Methods
    -------
    embed(texts)
        Generates embeddings for a list of texts.
    """

    def __init__(
            self, 
            device='cpu', 
            model_ckpt='distilbert-base-uncased', 
            max_length=128,
            padding=True,
            truncation=True
        ):
        """
        Constructs all the necessary attributes for the BertEmbedder object.

        Parameters
        ----------
        device : str, optional
            The device to run the model on (default is 'cpu').
        model_ckpt : str, optional
            The checkpoint name of the pre-trained DistilBERT model (default is 'distilbert-base-uncased').
        max_length : int, optional
            The maximum length of the tokenized input sequences (default is 128).
        padding : bool, optional
            Whether to pad the input sequences to the maximum length (default is True).
        truncation : bool, optional
            Whether to truncate the input sequences to the maximum length (default is True).
        """
        
        # Load attributes
        self.device = device
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # Load the model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
        self.model = DistilBertModel.from_pretrained(model_ckpt).to(self.device)

        # The model should be always in evaluation mode
        self.model.eval()

    def embed(self, texts):
        """
        Generates embeddings for a list of texts.

        Parameters
        ----------
        texts : list of str
            A list of texts to generate embeddings for.

        Returns
        -------
        torch.Tensor
            A tensor containing the embeddings for the input texts.
        """

        # Tokenize the text
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation
        )

        # Ensure that the input tensors are on the correct device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # Get model outputs and return last hidden state of the [CLS] token
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]
    

class BertDataset(Dataset):
    """
    A custom  PyTorch Dataset class for handling text data for BERT model.
    Attributes:
        texts (list): A list of text data.
    """
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx]}