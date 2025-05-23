from omegaconf import OmegaConf

from aizynthmodels.chemformer.data.tokenizer import (
    ChemformerTokenizer,
    ReplaceTokensMasker,
    SpanTokensMasker,
    build_masker,
)


def test_create_vocab(setup_tokenizer):
    tokenizer = setup_tokenizer()
    expected = {
        "<PAD>": 0,
        "?": 1,
        "^": 2,
        "&": 3,
        "<MASK>": 4,
        "<SEP>": 5,
        "C": 6,
        "O": 7,
        ".": 8,
        "c": 9,
        "Cl": 10,
        "(": 11,
        "=": 12,
        ")": 13,
        "Br": 14,
    }

    vocab = tokenizer.vocabulary

    assert expected == vocab
    assert tokenizer.chem_token_idxs == [6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_add_non_chem_tokens(setup_tokenizer):
    tokenizer = setup_tokenizer(tokens=["<RESERVED>"])
    expected = {
        "<PAD>": 0,
        "?": 1,
        "^": 2,
        "&": 3,
        "<MASK>": 4,
        "<SEP>": 5,
        "<RESERVED>": 6,
        "C": 7,
        "O": 8,
        ".": 9,
        "c": 10,
        "Cl": 11,
        "(": 12,
        "=": 13,
        ")": 14,
        "Br": 15,
    }

    assert expected == tokenizer.vocabulary
    assert tokenizer.chem_token_idxs == [7, 8, 9, 10, 11, 12, 13, 14, 15]


def test_save_and_load(setup_tokenizer, tmpdir):
    test_smiles = ["C.CCCcc1(Br)cccC"]
    filename = str(tmpdir / "vocab.json")
    tokenizer = setup_tokenizer()

    tokenizer.save_vocabulary(filename)

    tokenizer2 = ChemformerTokenizer(filename=filename)

    assert str(tokenizer2) == "ChemformerTokenizer"
    assert tokenizer(test_smiles)[0].tolist() == tokenizer2(test_smiles)[0].tolist()
    assert tokenizer2.chem_token_idxs == [6, 7, 8, 9, 10, 11, 12, 13, 14]


def test_mask_tokens_empty_mask(setup_masker, example_tokens):
    _, masker = setup_masker()
    masked, token_mask = masker(example_tokens, empty_mask=True)
    expected_sum = 0
    mask_sum = sum([sum(m) for m in token_mask])

    assert masked == example_tokens
    assert expected_sum == mask_sum


def test_mask_tokens_replace_zero_mask_token_prob(setup_masker, mock_random_choice, example_tokens):
    _, masker = setup_masker(ReplaceTokensMasker, {"mask_prob": 0.0})

    masked, token_mask = masker(example_tokens)

    expected_masks = [
        [False, True, False, True, False, True, False, False],
        [False, True, False, True, False, True, False],
    ]

    assert expected_masks == token_mask


def test_mask_tokens_replace(setup_masker, mock_random_choice, example_tokens):
    _, masker = setup_masker(ReplaceTokensMasker)

    masked, token_mask = masker(example_tokens)

    expected_masks = [
        [False, True, False, True, False, True, False, False],
        [False, True, False, True, False, True, False],
    ]

    assert expected_masks == token_mask


def test_mask_tokens_span(setup_masker, mock_random_choice, mocker, example_tokens):
    patched_poisson = mocker.patch("aizynthmodels.chemformer.data.tokenizer.torch.poisson")
    patched_poisson.return_value.long.return_value.item.side_effect = [3, 3, 2, 3]
    _, masker = setup_masker()
    masked, token_mask = masker(example_tokens)

    expected_masks = [
        [False, True, False, True, False],
        [False, True, True, False],
    ]

    assert expected_masks == token_mask


def test_convert_tokens_to_ids(regex_tokens, smiles_data, example_tokens):
    tokenizer = ChemformerTokenizer(smiles=smiles_data[2:3], regex_token_patterns=regex_tokens)
    ids = tokenizer.convert_tokens_to_ids(example_tokens)
    expected_ids = [[2, 6, 7, 8, 9, 10, 1, 3], [2, 6, 6, 5, 6, 11, 3]]

    assert expected_ids == [item.tolist() for item in ids]


def test_tokenize_one_sentence(setup_tokenizer, smiles_data):
    tokenizer = setup_tokenizer()
    tokens = tokenizer.tokenize(smiles_data)
    expected = [
        ["^", "C", "C", "O", ".", "C", "c", "c", "&"],
        ["^", "C", "C", "Cl", "C", "Cl", "&"],
        ["^", "C", "(", "=", "O", ")", "C", "Br", "&"],
    ]

    assert expected == tokens


def test_build_masker(regex_tokens, smiles_data):
    tokenizer = ChemformerTokenizer(smiles=smiles_data[2:3], regex_token_patterns=regex_tokens)

    masker_config = OmegaConf.create({"type": "SpanTokensMasker", "arguments": [{"mask_prob": 0.1}]})
    masker = build_masker({"masker": masker_config}, tokenizer)

    assert isinstance(masker, SpanTokensMasker)

    masker_config = OmegaConf.create({"type": "ReplaceTokensMasker", "arguments": [{"mask_prob": 0.7}]})
    masker = build_masker({"masker": masker_config}, tokenizer)
    assert isinstance(masker, ReplaceTokensMasker)

    masker = build_masker({}, tokenizer)
    assert masker is None
