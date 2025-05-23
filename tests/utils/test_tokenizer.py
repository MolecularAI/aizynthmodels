import os
import re

import pytest
import torch

from aizynthmodels.utils.tokenizer import SMILESAtomTokenizer, SMILESTokenizer


@pytest.fixture
def smiles(shared_datadir):
    with open(shared_datadir / "test_smiles.smi") as file:
        smiles_data = file.readlines()
        test_smiles = [smi[:-1] for smi in smiles_data]
    return test_smiles


@pytest.fixture
def multi_char_tokens(shared_datadir):
    with open(shared_datadir / "multi_char_atoms.txt") as file:
        tokens = [t for t in file.read().split() if not t.startswith("#")]
    return tokens


@pytest.fixture
def tokenizer(smiles):
    return SMILESTokenizer(smiles=smiles)


def test_default_arguments():
    with pytest.warns(Warning):
        tok = SMILESTokenizer()

    assert tok.vocabulary != {}
    assert tok.decoder_vocabulary != {}
    assert isinstance(tok.re, re.Pattern)


def test_one_hot_encoding():
    smiles = ["BrC[nHCl]"]

    tok = SMILESTokenizer(smiles=sorted(list(smiles[0])), encoding_type="one hot")

    ids = tok.convert_tokens_to_ids(tok.tokenize(smiles))
    one_hot = tok(smiles)

    ids_truth = [1, 4, 11, 5, 7, 10, 6, 5, 9, 8, 2]
    one_hot_truth = torch.zeros(size=(11, 12))
    one_hot_truth[torch.arange(11), ids_truth] = 1

    assert torch.equal(one_hot[0], one_hot_truth)
    assert all(id1 == id2 for id1, id2 in zip(ids[0].tolist(), ids_truth))


def test_tokenize_no_tokens(smiles):
    tok = SMILESTokenizer(smiles=smiles)

    smiles_tokenized = tok.tokenize(smiles)
    for smi, smi_tok in zip(smiles, smiles_tokenized):
        assert all(token == token_tok for token, token_tok in zip(["^"] + list(smi) + ["&"], smi_tok))

    smiles_tokenized = tok.tokenize(smiles, enclose=False)
    for smi, smi_tok in zip(smiles, smiles_tokenized):
        assert all(token == token_tok for token, token_tok in zip(list(smi), smi_tok))


def test_tokenize_multi_char_tokens():
    smiles = [
        "BrC[nHCl]",
        "Cl.Cc1(Br)ccC",
    ]

    correct_tokens = [
        ["^", "Br", "C", "[", "n", "H", "Cl", "]", "&"],
        ["^", "Cl", ".", "C", "c", "1", "(", "Br", ")", "c", "c", "C", "&"],
    ]

    tok = SMILESTokenizer(smiles=smiles, tokens=["Bq", "Br", "Cl"])

    smiles_tokenized = tok.tokenize(smiles)

    assert all(
        [
            smi_tok == corr_tok
            for smiles_tok, correct_tok in zip(smiles_tokenized, correct_tokens)
            for smi_tok, corr_tok in zip(smiles_tok, correct_tok)
        ]
    )
    assert len(tok) == 16


def test_add_already_present_token():
    smiles = [
        "BrC[nHCl]",
        "Cl.Cc1(Br)ccC",
    ]
    tok = SMILESTokenizer(tokens=["Bq", "Br", "Cl"])
    with pytest.raises(ValueError, match="already present in list of tokens"):
        tok.add_tokens(["Br"], smiles=smiles)


def test_remove_token_from_vocab():
    smiles = [
        "BrC[nHCl]",
        "Cl.Cc1(Br)ccC",
    ]
    tok = SMILESTokenizer(tokens=["Bq", "Br", "Cl"])
    tok.add_tokens(tokens=[], smiles=smiles)
    vocab1 = tok.vocabulary

    tok.remove_token_from_vocabulary("Br")
    vocab2 = tok.vocabulary

    assert len(vocab1) == len(vocab2) + 1
    assert "Br" in vocab1
    assert "Br" not in vocab2

    with pytest.raises(ValueError, match="is not in the vocabulary"):
        tok.remove_token_from_vocabulary("not-a-token")


def test_get_item():
    smiles = [
        "BrC[nHCl]",
        "Cl.Cc1(Br)ccC",
    ]
    tok = SMILESTokenizer(smiles=smiles, tokens=["Bq", "Br", "Cl"])
    assert tok["^"] == 1
    assert tok["start"] == 1
    with pytest.raises(KeyError, match="Unknown token:"):
        assert tok["not-a-token"]


def test_regex_tokens():
    smiles = [
        "NC[nHCl]",
        "C.CCCcc1(Br)cccC",
    ]

    correct_tokens = [
        ["^", "N", "C", "[nHCl]", "&"],
        ["^", "C", ".", "CCC", "cc", "1", "(", "B", "r", ")", "ccc", "C", "&"],
    ]

    tok = SMILESTokenizer(smiles=smiles, regex_token_patterns=[r"\[[^\]]+\]", "[c]+", "[C]+"])

    smiles_tokenized = tok.tokenize(smiles)
    assert all(
        [
            smi_tok == corr_tok
            for smiles_tok, correct_tok in zip(smiles_tokenized, correct_tokens)
            for smi_tok, corr_tok in zip(smiles_tok, correct_tok)
        ]
    )


# TODO Amiguity of Sc and Sn should be fixed
def test_tokenize_detokenize_inverse(tokenizer, smiles):
    tokenized_data = tokenizer.tokenize(smiles)
    detokenized_data = tokenizer.detokenize(tokenized_data)

    assert all(detok == smi for detok, smi in zip(detokenized_data, smiles))


def test_detokenize_new_lines_and_control_and_padding(tokenizer):
    smiles = ["^CN1&\n", "^cccCl&\n"]
    tokens = [
        [" ", "^", "C", "N", "1", "&"],
        ["^", "c", "c", "c", "Cl", "&", " "],
        ["^", "c", "c", "c", "Cl", "&", "Br", " "],
    ]

    smiles_raw = tokenizer.detokenize(tokens)
    smiles_control = tokenizer.detokenize(tokens, include_control_tokens=True)
    smiles_truncated = tokenizer.detokenize(tokens, include_control_tokens=False, truncate_at_end_token=True)

    assert all(len(smi_trunc) < len(smi_control) for smi_trunc, smi_control in zip(smiles_truncated, smiles_control))

    smiles_end_of_line = tokenizer.detokenize(tokens, include_end_of_line_token=True)
    smiles_all = tokenizer.detokenize(tokens, include_end_of_line_token=True, include_control_tokens=True)

    for smi, smi_detokenized in zip(smiles, smiles_raw):
        assert smi[1:-2] == smi_detokenized

    for smi, smi_detokenized in zip(smiles, smiles_control):
        assert smi[:-1] == smi_detokenized

    for smi, smi_detokenized in zip(smiles, smiles_end_of_line):
        assert smi[1:-2] + smi[-1:] == smi_detokenized

    for smi, smi_detokenized in zip(smiles, smiles_all):
        assert smi == smi_detokenized

    for smi, smi_truncated in zip(smiles, smiles_end_of_line):
        assert smi[1:-2] + smi[-1:] == smi_truncated


def test_ids_to_encoding_to_ids(tokenizer, smiles):
    # Test on List[str] input
    encoding_ids = tokenizer(smiles)

    with pytest.raises(ValueError, match="unknown choice of encoding"):
        tokenizer.convert_ids_to_encoding(encoding_ids, encoding_type="not-encoding-type")

    encoding_oh = tokenizer.convert_ids_to_encoding(encoding_ids, encoding_type="one hot")
    decoding_ids = tokenizer.convert_encoding_to_ids(encoding_oh, encoding_type="one hot")

    for encoded_id, decoded_id in zip(encoding_ids, decoding_ids):
        assert torch.equal(encoded_id, decoded_id)

    # Test on str input
    encoding_ids = tokenizer(smiles[0])
    encoding_oh = tokenizer.convert_ids_to_encoding(encoding_ids, encoding_type="one hot")
    decoding_ids = tokenizer.convert_encoding_to_ids(encoding_oh, encoding_type="one hot")
    for encoded_id, decoded_id in zip(encoding_ids, decoding_ids):
        assert torch.equal(encoded_id, decoded_id)


def test_encode_decode_encode_index(tokenizer, smiles):
    encoded_data = tokenizer(smiles)
    decoded_smiles = tokenizer.decode(encoded_data)

    for smi, smi_decoded in zip(smiles, decoded_smiles):
        assert smi == smi_decoded


def test_encode_decode_encode_one_hot(tokenizer, smiles):
    encoded_data = tokenizer(smiles, encoding_type="one hot")
    decoded_smiles = tokenizer.decode(encoded_data, encoding_type="one hot")

    for smi, smi_decoded in zip(smiles, decoded_smiles):
        assert smi == smi_decoded


def test_save_and_load(tokenizer, tmpdir):
    test_smiles = ["C.CCCcc1(Br)cccC"]
    filename = str(tmpdir / "vocab.json")

    tokenizer.save_vocabulary(filename)

    assert os.path.exists(filename)

    tokenizer2 = SMILESTokenizer(filename=filename)

    assert tokenizer(test_smiles)[0].tolist() == tokenizer2(test_smiles)[0].tolist()

    with pytest.warns(Warning):
        tokenizer3 = SMILESTokenizer()

    assert tokenizer(test_smiles)[0].tolist() != tokenizer3(test_smiles)[0].tolist()


def test_default_atom_tokens():
    with pytest.warns(Warning):
        atom_tokenizer = SMILESAtomTokenizer()

    assert atom_tokenizer.vocabulary != {}
    assert atom_tokenizer.decoder_vocabulary != {}
    assert isinstance(atom_tokenizer.re, re.Pattern)


def test_atom_tokens(smiles, multi_char_tokens):
    tokenizer = SMILESTokenizer(smiles=smiles, tokens=multi_char_tokens)
    atom_tokenizer = SMILESAtomTokenizer(smiles=smiles)
    assert tokenizer.vocabulary == atom_tokenizer.vocabulary

    for tokens, atom_tokens in zip(tokenizer(smiles), atom_tokenizer(smiles)):
        assert torch.equal(tokens, atom_tokens)
