from copy import deepcopy
import logging
import random

from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np

from src.steering.config import SteeringConfig
from src.data.dataset import DatasetConstructor
from src.utils.helpers import mean_pooling
from src.utils.constants import SampleSelectionMethods

logger = logging.getLogger(__name__)


class ContrastiveDatasetConstructor(DatasetConstructor):

    def __init__(
        self,
        dataset: Dataset,
        steering_config: SteeringConfig,
        task: str,
        type: str = "cot",
    ):
        super().__init__(
            dataset,
            task,
            steering_config.tokenizer,
            steering_config.use_tokenized_version,
            steering_config.encoding_method,
        )
        self.num_fewshot_examples = steering_config.num_fewshot_examples
        self.n_contrastive_samples = steering_config.n_contrastive_samples
        self.add_answer = steering_config.add_answer
        self.add_question = steering_config.add_question
        self.system_prompt = steering_config.extraction_system_prompt
        self.prefix = steering_config.prefix if steering_config.add_prefix else ""
        self.add_generation_prompt = steering_config.add_generation_prompt
        self.type = type
        self.sample_selection_method = steering_config.sample_selection_method
        self.fewshot_map = {}

        # Select the first n examples
        if self.n_contrastive_samples:
            self.dataset = self.dataset.select(range(self.n_contrastive_samples))

        # Copy the dataset to avoid modifying the original one
        self.full_dataset = deepcopy(self.dataset)
        self.distances = self.compute_question_distances(8)

    def process_qa_example(self, example, fewshot_examples):
        positive, negative = "", ""

    def process_instruct_example(self, example, fewshot_examples):
        positive, negative = [], []

        # Add the system prompt
        if self.system_prompt and not "raise_exception('System role not supported')" in self.tokenizer.chat_template: # TODO: there has to be a better solution than hardcoding this
            positive.append({"role": "system", "content": self.system_prompt})
            negative.append({"role": "system", "content": self.system_prompt})

        # Add the few-shot examples
        for fewshot_example in fewshot_examples:
            question = self.question_template.format(question=fewshot_example["question"])
            positive.append({"role": "user", "content": question})
            negative.append({"role": "user", "content": question})

            positive_answer, negative_answer = self.get_contrastive_answers(fewshot_example)
            positive.append({"role": "assistant", "content": positive_answer})
            negative.append({"role": "assistant", "content": negative_answer})

        # Add the actual example
        if self.add_question:
            positive.append({"role": "user", "content": self.question_template.format(question=example["question"])})
            negative.append({"role": "user", "content": self.question_template.format(question=example["question"])})

        positive_answer = self.prefix if self.prefix else ""
        negative_answer = self.prefix if self.prefix else ""
        if self.add_answer:
            positive_answer_cont, negative_answer_cont = self.get_contrastive_answers(example)
            negative_answer += negative_answer_cont
            positive_answer += positive_answer_cont

        positive = self.tokenizer.apply_chat_template(positive, tokenize=False, add_generation_prompt=self.add_generation_prompt)
        negative = self.tokenizer.apply_chat_template(negative, tokenize=False, add_generation_prompt=self.add_generation_prompt)

        if self.prefix and not self.add_answer and self.add_generation_prompt:
            positive += self.prefix
            negative += self.prefix

        return positive, negative

    def get_contrastive_answers(self, example):
        """ Get the contrastive answers for the example. """

        if self.type == "cot":
            negative_answer = self.answer_template.format(answer=example["answer"])
            positive_answer = example["steps"] + negative_answer

        elif self.type == "contrastive":
            negative_answer = self.answer_template.format(answer=example["incorrect_answer"])
            positive_answer = self.answer_template.format(answer=example["answer"])
        else:
            raise ValueError(f"Unknown type: {self.type}")

        return positive_answer, negative_answer

    def sample_fewshot_examples(self, example):
        """
        Sample the few-shot examples from the training set.

        Args:
            task_object: The task object
            num_fewshot: The number of few-shot examples
            example: The example to avoid

        Returns:
            The few-shot examples.
        """
        if self.sample_selection_method == SampleSelectionMethods.distance:
            fewshot_examples = self.select_similar_examples(example)
            self.fewshot_map[example["id"]] = [e["id"] for e in fewshot_examples]
            return fewshot_examples

        sampled_docs = random.sample(list(self.dataset), self.num_fewshot_examples)

        # Make sure the sampled examples do not contain the actual example
        questions = [doc['question'] for doc in sampled_docs]
        while example['question'] in questions:
            sampled_docs = random.sample(list(self.dataset), self.num_fewshot_examples)
            questions = [doc['question'] for doc in sampled_docs]

        self.fewshot_map[example["id"]] = [e["id"] for e in sampled_docs]
        return sampled_docs

    def select_similar_examples(self, example):
        """
        Select the similar examples from the training set.

        Args:
            task_object: The task object
            num_fewshot: The number of few-shot examples
            example: The example to avoid
        Returns:
            The few-shot examples.
        """

        # Get the index of the example
        example_index = self.full_dataset['id'].index(example['id'])
        example_distances = self.distances[example_index]
        closest_indices = example_distances.argsort()

        # Remove itself from the list of closest indices
        closest_indices = [i for i in closest_indices if i != example_index]
        selected_indices = closest_indices[:self.num_fewshot_examples]

        return [self.full_dataset[int(i)] for i in selected_indices]

    def compute_question_distances(self, batch_size):
        """
        Compute the distances for the given questions.

        Returns:
            The question distance matrix
        """

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)

        # Check if the distance matrix is already cached
        # try:
        #     distance_matrix = np.load(f"cached_vectors/question_distances_{self.task}.npy")
        #     return distance_matrix
        # except FileNotFoundError:
        #     logger.info("Distance matrix not found, computing it...")

        all_embeddings = []
        for examples in self.dataset.iter(batch_size=batch_size):
            tokens = tokenizer(
                examples["question"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output = model(**tokens)

            sentence_embeddings = mean_pooling(output, tokens['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(sentence_embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Compute the distance matrix using cosine distance (1 - cosine similarity)
        cosine_similarity = torch.matmul(all_embeddings, all_embeddings.T)
        distance_matrix = 1 - cosine_similarity
        distance_matrix = distance_matrix.cpu().numpy()

        # Save the distance matrix to a file
        # file_name = f"cached_vectors/question_distances_{self.task}.npy"
        # np.save(file_name, distance_matrix)

        return distance_matrix
