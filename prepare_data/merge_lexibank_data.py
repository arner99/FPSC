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
# TODO obsolete
ipa_all = pd.read_csv("panphon_data/expanded/all_ipa_symbols.csv", index_col="Unnamed: 0", dtype=str)
all_ipa_symbols = list(ipa_all.index)

# read supported base symbols
base_symbols = []
with open("etinen_symbol_data/base_ipa_symbols.csv") as f:
    for line in f:
        symbol = line.split(",")[0]
        if symbol != "":
            base_symbols.append(symbol)

# read supported diacritics
diacritics = []
with open("etinen_symbol_data/diacritic_rules.csv") as f:
    for line in f:
        symbol = line.split("\t")[0]
        if symbol != "":
            diacritics.append(symbol)

# read vowels (for polyphthong processing)
vowels = []
with open("etinen_symbol_data/vowel_dimensions.csv") as f:
    for line in f:
        symbol = line.split()[0]
        if symbol != "":
            vowels.append(symbol)


def symbol_is_etinen_compatible(symbol):
    if symbol in base_symbols:
        return True
    elif symbol == "":
        return False
    else:
        if symbol[-1] in diacritics:
            return symbol_is_etinen_compatible(symbol[:-1])
        elif symbol[0] in diacritics:
            return symbol_is_etinen_compatible(symbol[1:])
        else:
            return False


def is_polyphthong(symbol):
    vowel_count = 0
    for c in symbol:
        if c in vowels:
            vowel_count += 1

    return vowel_count == 2 or vowel_count == 3


def reorder_polyphthong(symbol):
    polyphthong = ""
    other_symbols = ""
    for c in symbol:
        if c in vowels:
            polyphthong += c
        else:
            other_symbols += c

    base_symbols.append(polyphthong)
    return polyphthong + other_symbols


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


def filter_forms_for_etinen(forms):
    occurrences_per_unknown_symbol = {}
    filtered_forms = []

    for form in forms:
        processable = True
        segments = form["Segments"]
        for i, seg in enumerate(segments):
            # remove arrow symbols and stress markers that EtInEn can't process
            seg = seg.replace("→", "").replace("←", "").replace("ˈ", "").replace(":", "ː")
            # handle slash notations. right hand side usually corresponds to canonical IPA notation.

            if "/" in seg:
                try:
                    left, right = seg.split("/")

                    # polypthong handling
                    if is_polyphthong(left):
                        left = reorder_polyphthong(left)
                    if is_polyphthong(right):
                        right = reorder_polyphthong(right)

                    if symbol_is_etinen_compatible(right):
                        seg = right
                    elif symbol_is_etinen_compatible(left):
                        seg = left
                    else:
                        processable = False
                        if seg in occurrences_per_unknown_symbol:
                            occurrences_per_unknown_symbol[seg] += 1
                        else:
                            occurrences_per_unknown_symbol[seg] = 1
                except:
                    print("More than one '/' in symbol " + seg)
                    processable = False
                    if seg in occurrences_per_unknown_symbol:
                        occurrences_per_unknown_symbol[seg] += 1
                    else:
                        occurrences_per_unknown_symbol[seg] = 1
            else:
                # polyphthong handling
                if is_polyphthong(seg):
                    seg = reorder_polyphthong(seg)

                if not symbol_is_etinen_compatible(seg):
                    processable = False
                    if seg in occurrences_per_unknown_symbol:
                        occurrences_per_unknown_symbol[seg] += 1
                    else:
                        occurrences_per_unknown_symbol[seg] = 1

            segments[i] = seg

        while "" in segments:
            segments.remove("")

        if processable:
            form["Segments"] = segments
            filtered_forms.append(form)

    return filtered_forms, occurrences_per_unknown_symbol


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

    try:
        cognates = list(dataset["CognateTable"])
    except KeyError:
        cognates = []

    if greedy_tokenization:
        standardized_forms = []
        relevant_keys = ["ID", "Language_ID", "Parameter_ID", "Form", "Segments", "Source"]
        for form in forms:
            standardized_dict = {key: value for (key, value) in form.items() if key in relevant_keys}
            standardized_dict["Own_Segments"] = standardized_dict["Segments"]
            standardized_dict["Segments"] = tokenize_greedily(form["Form"])
            standardized_forms.append(standardized_dict)
        forms = standardized_forms

    return forms, params, langs, cognates


def write_data(forms, params, langs, cognates, output_dir):
    """
    write data in a CLDF format

    @param forms - the forms that the database should include
    """
    dataset = Wordlist.in_dir(output_dir + "/cldf")
    dataset.add_component("ParameterTable")
    dataset.add_component("LanguageTable")
    dataset.add_component("CognateTable")
    dataset.write(FormTable=forms, ParameterTable=params, LanguageTable=langs, CognateTable=cognates)


if __name__ == '__main__':
    all_forms = []
    all_params = []
    all_langs = []
    all_cognates = []

    # store how many concepts each database contains, and which languages (by glottocode) are contained in which dataset.
    # if glottocodes overlap, remove forms of the database that contains i. less varities under that glottocode and
    # ii. less concepts.
    num_params_per_database = {}
    glottocode_coverage_by_database = {}

    lexibank_source_dir = "./lexibank_data/source_databases_cog"
    databases = [name for name in os.listdir(lexibank_source_dir)
                 if os.path.isdir(os.path.join(lexibank_source_dir, name))]
    target_dir = "./lexibank_data/merged_dataset_cog"

    for database in databases:
        forms, params, langs, cognates = read_data(database, input_dir=lexibank_source_dir)

        num_params_per_database[database] = len(params)
        for lang in langs:
            glottocode = lang["Glottocode"]
            if glottocode in glottocode_coverage_by_database:
                glotto_dict = glottocode_coverage_by_database[glottocode]
                if database in glotto_dict:
                    glotto_dict[database].append(lang["ID"])
                else:
                    glotto_dict[database] = [lang["ID"]]
            else:
                glottocode_coverage_by_database[glottocode] = {database: [lang["ID"]]}

        all_forms += forms
        all_params += params
        all_langs += langs
        all_cognates += cognates

    lang_ids_to_remove = [lang["ID"] for lang in all_langs if lang["Glottocode"] == ""]

    for glottocode, sources in glottocode_coverage_by_database.items():
        if len(sources) > 1:
            most_varieties = 1
            databases_with_most_varieties = []
            for source_database, varieties in sources.items():
                if len(varieties) > most_varieties:
                    most_varieties = len(varieties)
                    databases_with_most_varieties = [source_database]
                elif len(varieties) == most_varieties:
                    databases_with_most_varieties.append(source_database)

            if len(databases_with_most_varieties) > 1:
                database_to_keep = max(databases_with_most_varieties, key=num_params_per_database.get)
            else:
                database_to_keep = databases_with_most_varieties[0]

            for source_database, varieties in sources.items():
                if source_database != database_to_keep:
                    lang_ids_to_remove += [varieties]

    all_forms = [form for form in all_forms if form["Language_ID"] not in lang_ids_to_remove]
    all_langs = [lang for lang in all_langs if lang["ID"] not in lang_ids_to_remove]

    forms_filtered, unknown_sybols = filter_forms_for_etinen(all_forms)

    #inventory, unknown_symbols = make_inventory(all_forms)
    #forms_filtered = filter_forms(all_forms, inventory, unknown_symbols)
    write_data(forms_filtered, all_params, all_langs, all_cognates, output_dir=target_dir)

    with open(f"{target_dir}/unknown_sybols.tsv", "w") as f:
        for symbol, count in unknown_sybols.items():
            f.write(f"{symbol}\t{count}\n")
