import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """

        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1] # Número de características (palabras en el vocabulario), Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)


    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples
        num_samples = labels.shape[0] # numero de samples
        class_counts = torch.bincount(labels.long())  # cuántos ejemplos hay por clase ej: tensor([2, 2]) si dos clases
        class_priors: Dict[int, torch.Tensor] = { int(class_label): count.float() / num_samples for class_label, count in enumerate(class_counts)}
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # TODO: Estimate conditional probabilities for the words in features and apply smoothing
        classes = torch.unique(labels) # clases que hay unicas
        vocab_size = features.shape[1]  # Tamaño del vocabulario (número de palabras)
        num_classes = len(classes) # numero de clases

        class_word_counts: Dict[int, torch.Tensor] = {}
        for c in classes:
            class_features = features[labels == c]  # escogemos los ejemplos de la clase c
            word_counts = class_features.sum(dim=0)  # Suma cuántas veces aparece cada palabra columna por columna
            conditional_probabilities = (word_counts + delta) / (word_counts.sum() + delta * vocab_size) # calcula y asigna las probabilidades condicionales de las palabras para una clase específica
            class_word_counts[int(c)] = conditional_probabilities # guardamos las probabilidades condicionales en el diccionario
        return class_word_counts

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words
        log_posteriors: torch.Tensor = torch.zeros(len(self.class_priors))
        # log_prior + log_likelihood = Log(P(C)) + Log(P(x | C)) = sum(feature * log(P(word | C)))
        for c in self.class_priors:
            log_prior = torch.log(self.class_priors[c])
            log_likelihood = torch.sum(feature * torch.log(self.conditional_probabilities[c]))
            log_posteriors[c] = log_prior + log_likelihood
        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # TODO: Calculate log posteriors and obtain the class of maximum likelihood
        log_posteriors = self.estimate_class_posteriors(feature) 
        pred: int = torch.argmax(log_posteriors).item()  # clase con mayor probabilidad
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        log_posteriors = self.estimate_class_posteriors(feature)
        probs: torch.Tensor = torch.softmax(log_posteriors, dim=0)  # convertimos log-probabilidades en probabilidades reales
        return probs
