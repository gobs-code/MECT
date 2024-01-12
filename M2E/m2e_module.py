# -*- coding: utf-8 -*-

import torch
import logging
import os
from typing import List, Dict
from utils import chunk_it, post_process_wikidata
from fairseq.models.bart import BARTHubInterface, BARTModel
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    XLMRobertaTokenizer,
    MBartForConditionalGeneration,
)

logger = logging.getLogger(__name__)

class _M2E_HubInterface:
    def sample(
            self,
            sentences: List[str],
            num_beams: int = 5,
            num_return_sequences=5,
            text_to_id: Dict[str, str] = None,
            marginalize: bool = False,
            **kwargs
    ) -> List[str]:
        input_args = {
            k: v.to(self.device)
            for k, v in self.tokenizer.batch_encode_plus(
                sentences, padding=True, return_tensors="pt"
            ).items()
        }

        outputs = self.generate(
            **input_args,
            min_length=0,
            max_length=1024,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )

        outputs = chunk_it(
            [
                {"text": text, "score": score, }
                for text, score in zip(
                self.tokenizer.batch_decode(
                    outputs.sequences, skip_special_tokens=True
                ),
                outputs.sequences_scores,
            )
            ],
            len(sentences),
        )

        outputs = post_process_wikidata(
            outputs, text_to_id=text_to_id, marginalize=marginalize
        )

        return outputs

    def encode(self, sentence):
        return self.tokenizer.encode(sentence, return_tensors="pt")[0]


class _M2E_HubInterface(_M2E_HubInterface, BartForConditionalGeneration):
    pass


class m_M2EHubInterface(__BIELHubInterface, MBartForConditionalGeneration):
    pass


class M2E(BartForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = M2EHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        return model


class mM2E(MBartForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = mM2EHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name_or_path)
        return model


class _M2EHubInterface:
    def sample(
            self,
            sentences: List[str],
            beam: int = 5,
            verbose: bool = False,
            text_to_id=None,
            marginalize=False,
            marginalize_lenpen=0.5,
            max_len_a=1024,
            max_len_b=1024,
            **kwargs,
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]

        batched_hypos = self.generate(
            tokenized_sentences,
            beam,
            verbose,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            **kwargs,
        )

        outputs = [
            [
                {"text": self.decode(hypo["tokens"]), "score": hypo["score"]}
                for hypo in hypos
            ]
            for hypos in batched_hypos
        ]

        outputs = post_process_wikidata(
            outputs, text_to_id=text_to_id, marginalize=marginalize
        )

        return outputs

    def generate(self, *args, **kwargs) -> List[List[Dict[str, torch.Tensor]]]:
        return super(BARTHubInterface, self).generate(*args, **kwargs)

    def encode(self, sentence) -> torch.LongTensor:
        tokens = super(BARTHubInterface, self).encode(sentence)
        tokens[
            tokens >= len(self.task.target_dictionary)
            ] = self.task.target_dictionary.unk_index
        if tokens[0] != self.task.target_dictionary.bos_index:
            return torch.cat(
                (torch.tensor([self.task.target_dictionary.bos_index]), tokens)
            )
        else:
            return tokens


class M2EHubInterface(_GENREHubInterface, BARTHubInterface):
    pass


class mM2EHubInterface(_GENREHubInterface, BARTHubInterface):
    pass


class M2E(BARTModel):
    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path,
            checkpoint_file="model.pt",
            data_name_or_path=".",
            bpe="gpt2",
            **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return M2EHubInterface(x["args"], x["task"], x["models"][0])


class mM2E(BARTModel):
    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path,
            sentencepiece_model="spm_256000.model",
            checkpoint_file="model.pt",
            data_name_or_path=".",
            bpe="sentencepiece",
            layernorm_embedding=True,
            **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            sentencepiece_model=os.path.join(model_name_or_path, sentencepiece_model),
            **kwargs,
        )
        return mM2EHubInterface(x["args"], x["task"], x["models"][0])
