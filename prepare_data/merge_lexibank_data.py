from pycldf.dataset import Dataset
from pycldf import Wordlist, StructureDataset
#from panphon import data as panphon_data
import pandas as pd
import json
import os


# dataset = Dataset.from_metadata("lundgrenomagoa/cldf/cldf-metadata.json")
# print(dataset)

# forms = list(dataset["FormTable"])
# derp_dict = dict(forms[0])

# print(derp_dict)

# read in the IPA data from panphon
ipa_all = pd.read_csv("panphon_data/expanded/all_ipa_symbols.csv", index_col="Unnamed: 0", dtype=str)
all_ipa_symbols = list(ipa_all.index)


def lookahead(word, start_index, end_index_from_behind, pattern):
    """
    recursive function to find the longest match of character sequences in a string

    @param word: the word to be tokenized
    @type word: str
    @param start_index: the start index of the substring that is checked for a match
    @type start_index: int
    @param end_index_from_behind: the end index of the substring that is checked, counted from the end of the word
    @type end_index_from_behind: int
    @param pattern: the list of symbols that provide information about which substrings are valid and which not
    @type pattern: list[str]

    @return the longest match with its start and end index
    @rtype str, int, int
    """
    # only lookahead for max length of IPA symbol
    substring = ""
    if end_index_from_behind == 0:
        substring = word[start_index:]
    else:
        substring = word[start_index:-end_index_from_behind]
    if substring in pattern or len(substring) == 1:
        return substring, start_index, (len(word) - end_index_from_behind)
    else:
        return lookahead(word, start_index, end_index_from_behind+1, pattern)


def tokenize_greedily(form):
    """
    a greedy tokenizer for IPA sequences. Tokenization is based on the panphon IPA chart.

    @param form: the form to be tokenized
    @type form: str

    @return the tokenized string
    @rtype list[str]
    """
    i = 0
    tokens = []
    for char in form:
        token, start_index, end_index = lookahead(form, i, 0, all_ipa_symbols)
        tokens.append(token)
        i += (end_index - start_index)
        if i >= len(form):
            break
    return tokens


def make_inventory(forms):
    """
    make a phoneme inventory from a given database, counting the number of occurences of each phoneme by language.
    differentiates between phonemes that are represented in panphon and those which are not.

    @param forms: all the forms (i.e. the dataset) that should be inventorized
    @type forms: list[dict[str:str]]

    @return the phoneme inventory (for phonemes that are in panphon) and a list of phonemes
        that are not represented in panphon
    @rtype dict[int:dict[str:int]], list[str]
    """
    inventory = {}
    undefined_segments = []
    for form in forms:
        lang = form["Language_ID"]
        segments = form["Segments"]
        for seg in segments:
            if seg in all_ipa_symbols:
                if seg in inventory:
                    inventory[seg]["total"] += 1
                    if lang in inventory[seg]:
                        inventory[seg][lang] += 1
                    else:
                        inventory[seg][lang] = 1
                else:
                    inventory[seg] = {
                        "total": 1,
                        lang: 1
                    }
            else:
                if seg not in undefined_segments:
                    undefined_segments.append(seg)
    return inventory, undefined_segments


def filter_forms(forms, inventory, undesired_symbols=None, threshold_one_lang=100, threshold_two_langs=50):
    """
    filter out forms that contain symbols that either don't match to panphon or have too few occurrences

    @param forms: the list of forms
    @type forms: list[dict[str:str]]
    @param inventory: the given phoneme inventory
    @type inventory: dict[int:dict[str:int]]
    @param undesired_symbols: symbols that need to be filtered out
    @type undesired_symbols: list[str]
    @param threshold_one_lang: the threshold for minimum occurrences of a phoneme in only one language
    @type threshold_one_lang: int
    @param threshold_two_langs: the threshold for minimum occurrences of a phoneme in at least two languages
    @type threshold_two_langs: int

    @return the filtered forms
    @rtype list[dict[str:str]]
    """
    if undesired_symbols is None:
        undesired_symbols = []
    for symbol in inventory:
        occurrences = inventory[symbol]
        l1 = [lang for lang in occurrences if occurrences[lang] <= threshold_one_lang]
        l2 = [lang for lang in occurrences if occurrences[lang] <= threshold_two_langs]
        if len(l1) < 1 and len(l2) < 2:
            undesired_symbols.append(symbol)
    filtered_forms = [form for form in forms if (len(set(undesired_symbols) & set(form["Segments"])) == 0)]
    return filtered_forms


def dump_inventory_to_json(inventory, filepath):
    """
    dumps a collected phoneme inventory in a json file.

    @param inventory: the inventory
    @type inventory: dict or list
    @param filepath: the desired location of the file
    @type filepath: str
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=4)


def read_json(filepath):
    """
    reads a json file

    @param filepath: the file to be read
    @type filepath: str

    @return the content of the json file
    @rtype list or dict
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_data(data, input_dir, greedy_tokenization=False):
    """
    read a CLDF dataset, normalize it, tokenize the raw IPA, and return it as a list of forms

    @param data - the dataset to be read

    @return the dataset as a list of forms
    @rtype list[dict[str:str]
    """
    dataset = Dataset.from_metadata(input_dir + "/" + data + "/cldf/cldf-metadata.json")
    forms = list(dataset["FormTable"])
    params = list(dataset["ParameterTable"])
    langs = list(dataset["LanguageTable"])

    if greedy_tokenization:
        standardized_forms = []
        relevant_keys = ["ID", "Language_ID", "Parameter_ID", "Form", "Segments", "Source"]
        for form in forms:
            standardized_dict = {key: value for (key, value) in form.items() if key in relevant_keys}
            standardized_dict["Own_Segments"] = standardized_dict["Segments"]
            standardized_dict["Segments"] = tokenize_greedily(form["Form"])
            standardized_forms.append(standardized_dict)
        forms = standardized_forms

    return forms, params, langs


def write_data(forms, params, langs, output_dir):
    """
    write data in a CLDF format

    @param forms - the forms that the database should include
    """
    dataset = Wordlist.in_dir(output_dir + "/cldf")
    dataset.add_component("ParameterTable")
    dataset.add_component("LanguageTable")
    #dataset.add_component("FormTable")
    dataset.write(FormTable=forms, ParameterTable=params, LanguageTable=langs)


if __name__ == '__main__':
    all_forms = []
    all_params = []
    all_langs = []

    #databases = ["bdpa", "bodtkhobwa", "castrosui", "chenhmongmien", "lundgrenomagoa", "naganorgyalrongic",
    #            "northeuralex", "suntb", "yanglalo"]

    lexibank_source_dir = "./lexibank_data/source_databases"
    databases = [name for name in os.listdir(lexibank_source_dir)
                 if os.path.isdir(os.path.join(lexibank_source_dir, name))]
    target_dir = "./lexibank_data/merged_dataset"

    for database in databases:
        forms, params, langs = read_data(database, input_dir=lexibank_source_dir)
        all_forms += forms
        all_params += params
        all_langs += langs

    inventory, unknown_symbols = make_inventory(all_forms)
    forms_filtered = filter_forms(all_forms, inventory, unknown_symbols)
    write_data(forms_filtered, all_params, all_langs, output_dir=target_dir)
