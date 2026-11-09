---
license: odc-by
size_categories:
- 100K<n<1M
task_categories:
- text-generation
- question-answering
- text2text-generation
pretty_name: WildChat
dataset_info:
  features:
  - name: conversation_id
    dtype: string
  - name: model
    dtype: string
  - name: timestamp
    dtype: timestamp[s, tz=UTC]
  - name: conversation
    list:
    - name: content
      dtype: string
    - name: language
      dtype: string
    - name: redacted
      dtype: bool
    - name: role
      dtype: string
    - name: toxic
      dtype: bool
  - name: turn
    dtype: int64
  - name: language
    dtype: string
  - name: openai_moderation
    list:
    - name: categories
      struct:
      - name: harassment
        dtype: bool
      - name: harassment/threatening
        dtype: bool
      - name: hate
        dtype: bool
      - name: hate/threatening
        dtype: bool
      - name: self-harm
        dtype: bool
      - name: self-harm/instructions
        dtype: bool
      - name: self-harm/intent
        dtype: bool
      - name: sexual
        dtype: bool
      - name: sexual/minors
        dtype: bool
      - name: violence
        dtype: bool
      - name: violence/graphic
        dtype: bool
    - name: category_scores
      struct:
      - name: harassment
        dtype: float64
      - name: harassment/threatening
        dtype: float64
      - name: hate
        dtype: float64
      - name: hate/threatening
        dtype: float64
      - name: self-harm
        dtype: float64
      - name: self-harm/instructions
        dtype: float64
      - name: self-harm/intent
        dtype: float64
      - name: sexual
        dtype: float64
      - name: sexual/minors
        dtype: float64
      - name: violence
        dtype: float64
      - name: violence/graphic
        dtype: float64
    - name: flagged
      dtype: bool
  - name: detoxify_moderation
    list:
    - name: identity_attack
      dtype: float32
    - name: insult
      dtype: float32
    - name: obscene
      dtype: float32
    - name: severe_toxicity
      dtype: float32
    - name: sexual_explicit
      dtype: float32
    - name: threat
      dtype: float32
    - name: toxicity
      dtype: float32
  - name: toxic
    dtype: bool
  - name: redacted
    dtype: bool
  splits:
  - name: train
    num_bytes: 2949464505.6494355
    num_examples: 529428
  download_size: 1586548072
  dataset_size: 2949464505.6494355
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
tags:
- instruction-finetuning
---
# Dataset Card for WildChat

## Note: a newer version with 4.8 million conversations and demographic information can be found [here](https://huggingface.co/datasets/allenai/WildChat-4.8M).

## Dataset Description
 
- **Paper:** https://arxiv.org/abs/2405.01470

- **Interactive Search Tool:** https://wildvisualizer.com ([paper](https://arxiv.org/abs/2409.03753))

- **License:** [ODC-BY](https://opendatacommons.org/licenses/by/1-0/)

- **Language(s) (NLP):** multi-lingual

- **Point of Contact:** [Yuntian Deng](https://yuntiandeng.com/)

### Dataset Summary

WildChat is a collection of 650K conversations between human users and ChatGPT. We collected WildChat by offering online users free access to OpenAI's GPT-3.5 and GPT-4. The dataset contains a broad spectrum of user-chatbot interactions that are not previously covered by other instruction fine-tuning datasets: for example, interactions include ambiguous user requests, code-switching, topic-switching, political discussions, etc. WildChat can serve both as a dataset for instructional fine-tuning and as a valuable resource for studying user behaviors. Note that this version of the dataset only contains non-toxic user inputs/ChatGPT responses.

### Updates

**2024-10-17: Content Update.** Conversations flagged by [Niloofar Mireshghallah](https://homes.cs.washington.edu/~niloofar/) and her collaborators in ["Breaking News: Case Studies of Generative AI's Use in Journalism"](https://arxiv.org/abs/2406.13706) for containing PII or sensitive information have been removed from this version of the dataset.

**2024-07-22: Content Update.** All toxic conversations identified by the OpenAI Moderations API or Detoxify have been removed from this version of the dataset.

**2024-06-26: License Change.** We have updated the license of WildChat to [ODC-BY](https://opendatacommons.org/licenses/by/1-0/). This change is retroactively applied to any previous downloads under the ImpACT license.

### Languages

66 languages were detected in WildChat.

### Personal and Sensitive Information

The data has been de-identified with Microsoft Presidio and hand-written rules by the authors.

### Data Fields

- `conversation_id` (string): Each conversation has a unique id.
- `model` (string): The underlying OpenAI model, such as gpt-3.5-turbo or gpt-4.
- `timestamp` (timestamp): The timestamp of the last turn in the conversation in UTC.
- `conversation` (list): A list of user/assistant utterances. Each utterance is a dictionary containing the `role` of the speaker (user or assistant), the `content` of the utterance, the detected `language` of the utterance, whether the content of the utterance is considered `toxic`, and whether PII has been detected and anonymized (`redacted`).
- `turn` (int): The number of turns in the conversation. A turn refers to one round of user-assistant interaction.
- `language` (string): The language of the conversation. Note that this is the most frequently detected language in the utterances of the conversation.
- `openai_moderation` (list): A list of OpenAI Moderation results. Each element in the list corresponds to one utterance in the conversation.
- `detoxify_moderation` (list): A list of Detoxify results. Each element in the list corresponds to one utterance in the conversation.
- `toxic` (bool): Whether this conversation contains any utterances considered to be toxic by either OpenAI Moderation or Detoxify.
- `redacted` (bool): Whether this conversation contains any utterances in which PII is detected and anonymized.

### Empty User Inputs

This dataset includes a small subset of conversations where users submitted empty inputs, sometimes leading to hallucinated responses from the assistant. This issue, first noticed by @yuchenlin, arises from the design of our Huggingface chatbot used for data collection, which did not restrict the submission of empty inputs. As a result, users could submit without entering any text, causing the assistant to generate responses without any user prompts. This occurs in a small fraction of the dataset---12,405 out of 652,139 conversations.

### Licensing Information

WildChat is now made available under the [**ODC-BY License**](https://opendatacommons.org/licenses/by/1-0/). This change is retroactively applied to any previous downloads under the ImpACT license.

### Citation Information

Please consider citing [our paper](https://arxiv.org/abs/2405.01470) if you find this dataset useful:
```
@inproceedings{
  zhao2024wildchat,
  title={WildChat: 1M Chat{GPT} Interaction Logs in the Wild},
  author={Wenting Zhao and Xiang Ren and Jack Hessel and Claire Cardie and Yejin Choi and Yuntian Deng},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=Bl8u7ZRlbM}
}
```

```
@inproceedings{deng2024wildvis,
  title     = "{W}ild{V}is: Open Source Visualizer for Million-Scale Chat Logs in the Wild",
  author    = "Deng, Yuntian and Zhao, Wenting and Hessel, Jack and Ren, Xiang and Cardie, Claire and Choi, Yejin",
  booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
  year      = "2024",
  url       = "https://aclanthology.org/2024.emnlp-demo.50/"
}
```