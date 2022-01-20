import ast
from data.data_utils import get_gt_seeds_titles, raw_data_link
import nltk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
import json
import csv
import sys
from models.reco.recos_utils import index_amp


nltk.download("punkt")


class WikipediaTextDatasetParagraphsSentences(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, dataset_name, block_size, mode="train"):
        self.hparams = hparams
        cached_features_file = os.path.join(
            f"data/datasets/cached_proccessed/{dataset_name}",
            f"bs_{block_size}_{dataset_name}_{type(self).__name__}_tokenizer_{str(type(tokenizer)).split('.')[-1][:-2]}_mode_{mode}",
        )
        self.cached_features_file = cached_features_file
        os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)

        raw_data_path = self.download_raw(dataset_name)

        all_articles = self.save_load_splitted_dataset(mode, cached_features_file, raw_data_path)

        self.hparams = hparams

        max_article_len, max_sentences, max_sent_len = int(1e6), 16, 10000
        block_size = min(block_size, tokenizer.max_len_sentences_pair) if tokenizer is not None else block_size
        self.block_size = block_size
        self.tokenizer = tokenizer

        if os.path.exists(cached_features_file) and (self.hparams is None or not self.hparams.overwrite_data_cache):
            print("\nLoading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples, self.indices_map = pickle.load(handle)
        else:
            print("\nCreating features from dataset file at ", cached_features_file)

            self.examples = []
            self.indices_map = []
            max_str_len = 0

            for idx_article, article in enumerate(tqdm(all_articles)):
                this_sample_sections = []
                if len(article[1]) > max_str_len:
                    max_str_len = len(article[1])
                try:
                    title, sections = article[0], ast.literal_eval(article[1])
                except SyntaxError:
                    print("Article IDX: ", idx_article)
                    print(max_str_len, len(article[1]))
                    print(article[1])
                valid_sections_count = 0
                for section_idx, section in enumerate(sections):
                    this_sections_sentences = []
                    if section[1] == "":
                        continue
                    valid_sentences_count = 0
                    title_with_base_title = "{}:{}".format(title, section[0])
                    for sent_idx, sent in enumerate(nltk.sent_tokenize(section[1][:max_article_len])[:max_sentences]):
                        tokenized_desc = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(json.dumps(sent[:max_sent_len])))[
                            :block_size
                        ]
                        this_sections_sentences.append(
                            (
                                tokenized_desc,
                                len(tokenized_desc),
                                idx_article,
                                valid_sections_count,
                                valid_sentences_count,
                                sent[:max_sent_len],
                            ),
                        )
                        self.indices_map.append((idx_article, valid_sections_count, valid_sentences_count))
                        valid_sentences_count += 1
                    this_sample_sections.append((this_sections_sentences, title_with_base_title))
                    valid_sections_count += 1
                self.examples.append((this_sample_sections, title))

            print("\nSaving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump((self.examples, self.indices_map), handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.labels = [idx_article for idx_article, _, _ in self.indices_map]

    def save_load_splitted_dataset(self, mode, cached_features_file, raw_data_path):
        proccessed_path = f"{cached_features_file}_EXAMPLES"
        if not os.path.exists(proccessed_path):
            all_articles = self.read_all_articles(raw_data_path)
            indices = list(range(len(all_articles)))
            if mode != "test":
                train_indices = sorted(
                    np.random.choice(indices, replace=False, size=int(len(all_articles) * self.hparams.train_val_ratio))
                )
                val_indices = np.setdiff1d(list(range(len(all_articles))), train_indices)
                indices = train_indices if mode == "train" else val_indices

            articles = []
            for i in indices:
                articles.append(all_articles[i])
            all_articles = articles
            pickle.dump(all_articles, open(proccessed_path, "wb"))
            print(f"\nsaved dataset at {proccessed_path}")
        else:
            all_articles = pickle.load(open(proccessed_path, "rb"))
        setattr(self.hparams, f"{mode}_data_file", proccessed_path)
        return all_articles

    def read_all_articles(self, raw_data_path):
        csv.field_size_limit(sys.maxsize if sys.maxsize < 2147483647 else 2147483647)
        with open(raw_data_path, encoding='utf8', newline="") as f:
            reader = csv.reader(f)
            all_articles = list(reader)
        return all_articles[1:]

    def download_raw(self, dataset_name):
        raw_data_path = f"data/datasets/{dataset_name}/raw_data"
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        if not os.path.exists(raw_data_path):
            print('operating system: {}'.format(sys.platform))
            if sys.platform == 'win32':
                os.system(f"curl.exe -o {raw_data_path} {raw_data_link(dataset_name)}")
            else:
                os.system(f"wget -O {raw_data_path} {raw_data_link(dataset_name)}")
        return raw_data_path

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, item):
        idx_article, idx_section, idx_sentence = self.indices_map[item]
        sent = self.examples[idx_article][0][idx_section][0][idx_sentence]

        return (
            torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sent[0]), dtype=torch.long,)[
                : self.hparams.limit_tokens
            ],
            self.examples[idx_article][1],
            self.examples[idx_article][0][idx_section][1],
            sent[1],
            idx_article,
            idx_section,
            idx_sentence,
            item,
            self.labels[item],
        )

class WikipediaTextDatasetParagraphsSentencesTest(WikipediaTextDatasetParagraphsSentences):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, dataset_name, block_size, mode="test"):
        super().__init__(tokenizer, hparams, dataset_name, block_size, mode=mode)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        sections = []
        for idx_section, section in enumerate(self.examples[item][0]):
            sentences = []
            for idx_sentence, sentence in enumerate(section[0]):
                sentences.append(
                    (
                        torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sentence[0]), dtype=torch.long,),
                        self.examples[item][1],
                        section[1],
                        sentence[1],
                        item,
                        idx_section,
                        idx_sentence,
                        item,
                        self.labels[item],
                    )
                )
            sections.append(sentences)
        return sections


class WikipediaTextDatasetOnePlusNCoherence(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, dataset_name, block_size, mode="train", N=1):
        self.hparams = hparams
        cached_features_file = os.path.join(
            f"data/datasets/cached_proccessed/{dataset_name}_coherence",
            f"bs_{block_size}_{dataset_name}_{type(self).__name__}_tokenizer_{str(type(tokenizer)).split('.')[-1][:-2]}_mode_{mode}",
        )
        self.cached_features_file = cached_features_file
        os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)

        raw_data_path = self.download_raw(dataset_name)

        all_articles = self.save_load_splitted_dataset(mode, cached_features_file, raw_data_path)

        self.hparams = hparams

        max_article_len, max_sentences, max_sent_len = int(1e6), 16, 10000
        block_size = min(block_size, tokenizer.max_len_sentences_pair) if tokenizer is not None else block_size
        self.block_size = block_size
        self.tokenizer = tokenizer

        if os.path.exists(cached_features_file+"1") and (self.hparams is None or not self.hparams.overwrite_data_cache):
            print("\nLoading features from cached file %s", cached_features_file)
            with open(cached_features_file+"1", "rb") as handle:
                self.examples1, self.indices_map = pickle.load(handle)
            with open(cached_features_file+"2", "rb") as handle:
                self.examples2, _ = pickle.load(handle)
        else:
            print("\nCreating features from dataset file at ", cached_features_file)

            self.examples1 = []
            self.examples2 = []
            self.indices_map = []
            self.indices_map2 = []

            for idx_article, article in enumerate(tqdm(all_articles)):
                this_sample_sections1 = []
                this_sample_sections2 = []
                title, sections = article[0], ast.literal_eval(article[1])
                valid_sections_count = 0
                for section_idx, section in enumerate(sections):
                    this_sections_sentences = []
                    if section[1] == "":
                        continue
                    valid_sequences_count = 0
                    title_with_base_title = "{}:{}".format(title, section[0])
                    sentences = nltk.sent_tokenize(section[1][:max_article_len])[:max_sentences]
                    tok_sentences = []
                    for sent in sentences:
                        tok_sentences.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(json.dumps(sent[:max_sent_len])))[
                            :block_size
                        ])
                    sentence_pairs = []
                    for sent_idx, sent in enumerate(tok_sentences):
                        if sent_idx + N < len(tok_sentences):
                            sentence_pairs.append(tok_sentences[sent_idx:sent_idx+N+1])
                        else:
                            continue
                    #sentence_pairs = list(zip(sentence_pairs))
                    sentence_dict = {tuple(item[0]): item[1:] for item in sentence_pairs}
                    first_sentences = []
                    subsequent_sentences = []
                    sentence_labels = []

                    for pair_idx, (key, value) in enumerate(sentence_pairs):
                        first_sentences.append((key,
                                               len(key),
                                               idx_article,
                                               valid_sections_count,
                                               valid_sequences_count,
                                               sentences[pair_idx]))
                        subsequent_sentences.append((value,
                                               len(value),
                                               idx_article,
                                               valid_sections_count,
                                               valid_sequences_count,
                                               sentences[pair_idx:pair_idx+N]))

                        self.indices_map.append((idx_article, valid_sections_count, valid_sequences_count))
                        valid_sequences_count += 1
                    this_sample_sections1.append((first_sentences, title_with_base_title))
                    this_sample_sections2.append((subsequent_sentences, title_with_base_title))

                    valid_sections_count += 1
                self.examples1.append((this_sample_sections1, title))
                self.examples2.append((this_sample_sections2, title))

            print("\nSaving features into cached file %s", cached_features_file)
            with open(cached_features_file+"1", "wb") as handle:
                pickle.dump((self.examples1, self.indices_map), handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cached_features_file+"2", "wb") as handle:
                pickle.dump((self.examples1, self.indices_map), handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.indices_map = np.array(self.indices_map)
        self.labels = torch.tensor([int("{}{}{}".format(art_idx, sec_idx, pair_idx)) for art_idx, sec_idx, pair_idx in self.indices_map])
        # make sure that each article, section and sentence pair index is only assigned once
        assert self.labels.unique(return_counts=True)[1].sum() == len(self.labels)

    def save_load_splitted_dataset(self, mode, cached_features_file, raw_data_path):
        proccessed_path = f"{cached_features_file}_EXAMPLES"
        if not os.path.exists(proccessed_path):
            all_articles = self.read_all_articles(raw_data_path)
            indices = list(range(len(all_articles)))
            if mode != "test":
                train_indices = sorted(
                    np.random.choice(indices, replace=False, size=int(len(all_articles) * self.hparams.train_val_ratio))
                )
                val_indices = np.setdiff1d(list(range(len(all_articles))), train_indices)
                indices = train_indices if mode == "train" else val_indices

            articles = []
            for i in indices:
                articles.append(all_articles[i])
            all_articles = articles
            pickle.dump(all_articles, open(proccessed_path, "wb"))
            print(f"\nsaved dataset at {proccessed_path}")
        else:
            all_articles = pickle.load(open(proccessed_path, "rb"))
        setattr(self.hparams, f"{mode}_data_file", proccessed_path)
        return all_articles

    def read_all_articles(self, raw_data_path):
        csv.field_size_limit(sys.maxsize if sys.maxsize < 2147483647 else 2147483647)
        with open(raw_data_path, encoding='utf8', newline="") as f:
            reader = csv.reader(f)
            all_articles = list(reader)
        return all_articles[1:]

    def download_raw(self, dataset_name):
        raw_data_path = f"data/datasets/{dataset_name}/raw_data"
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        if not os.path.exists(raw_data_path):
            print('operating system: {}'.format(sys.platform))
            if sys.platform == 'win32':
                os.system(f"curl.exe -o {raw_data_path} {raw_data_link(dataset_name)}")
            else:
                os.system(f"wget -O {raw_data_path} {raw_data_link(dataset_name)}")

        return raw_data_path

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, item):
        idx_article, idx_section, idx_sentence = self.indices_map[item]
        sent1 = self.examples1[idx_article][0][idx_section][0][idx_sentence]
        sent2 = self.examples2[idx_article][0][idx_section][0][idx_sentence]
        sample1 = (
            torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sent1[0]), dtype=torch.long,)[
                : self.hparams.limit_tokens
            ],
            self.examples1[idx_article][1],
            self.examples1[idx_article][0][idx_section][1],
            sent1[1],
            idx_article,
            idx_section,
            idx_sentence,
            item,
            self.labels[item],
        )
        sample2 = (
            torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sent2[0]), dtype=torch.long,)[
                : self.hparams.limit_tokens
            ],
            self.examples2[idx_article][1],
            self.examples2[idx_article][0][idx_section][1],
            sent2[1],
            idx_article,
            idx_section,
            idx_sentence,
            item,
            self.labels[item],
        )

        return (sample1, sample2)
