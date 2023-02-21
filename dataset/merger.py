from pycldf.dataset import Dataset
from pycldf import Wordlist, StructureDataset
from features.ipa_feature_table import IPAFeatureTable
import json
import os

feature_table = IPAFeatureTable()


def lookahead(word, start_index, end_index_from_behind):
    """
    recursive function to find the longest match of character sequences in a string

    @param word: the word to be tokenized
    @type word: str
    @param start_index: the start index of the substring that is checked for a match
    @type start_index: int
    @param end_index_from_behind: the end index of the substring that is checked, counted from the end of the word
    @type end_index_from_behind: int

    @return the longest match with its start and end index
    @rtype str, int, int
    """
    substring = ""
    if end_index_from_behind == 0:
        substring = word[start_index:]
    else:
        substring = word[start_index:-end_index_from_behind]
    if substring in feature_table or len(substring) == 1:
        return substring, start_index, (len(word) - end_index_from_behind)
    else:
        return lookahead(word, start_index, end_index_from_behind + 1)


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
        token, start_index, end_index = lookahead(form, i, 0)
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
            if seg in feature_table:
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
    DEPRECATED.
    filter out forms that contain symbols outside a given inventory or have too few occurrences

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

                    # check whether symbols are compatible with the feature table.
                    # prefer the right one over the left one in slash notations, since Lexibank seems to
                    # put a more universal notation on the right hand side (opposed to more specific, non-standard
                    # symbols / symbol usages on the left hand side)
                    if right in feature_table:
                        seg = right
                    elif left in feature_table:
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
                if seg not in feature_table:
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

    :param data: the dataset to be read
    :param input_dir: the directory where the dataset is located
    :param greedy_tokenization: apply greedy tokenization if true, use tokenization from database otherwise.

    @return the dataset as a list of forms
    @rtype list[dict[str:str]]

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


def merge_datasets(lexibank_source_dir, target_dir):
    all_forms = []
    all_params = []
    all_langs = []
    all_cognates = []

    # store how many concepts each database contains, and which languages (by glottocode) are contained in which dataset.
    # if glottocodes overlap, remove forms of the database that contains i. fewer varieties under that glottocode and
    # ii. fewer concepts.
    num_params_per_database = {}
    glottocode_coverage_by_database = {}

    # get names of all databases in the source directory
    databases = [name for name in os.listdir(lexibank_source_dir)
                 if os.path.isdir(os.path.join(lexibank_source_dir, name))]

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

    lang_ids_to_remove = [lang["ID"] for lang in all_langs if not lang["Glottocode"]]

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
                    lang_ids_to_remove += varieties

    all_forms = [form for form in all_forms if form["Language_ID"] not in lang_ids_to_remove]
    all_langs = [lang for lang in all_langs if lang["ID"] not in lang_ids_to_remove]

    forms_filtered, unknown_symbols = filter_forms_for_etinen(all_forms)

    # inventory, unknown_symbols = make_inventory(all_forms)
    # forms_filtered = filter_forms(all_forms, inventory, unknown_symbols)
    write_data(forms_filtered, all_params, all_langs, all_cognates, output_dir=target_dir)

    with open(f"{target_dir}/unknown_symbols.tsv", "w") as f:
        for symbol, count in unknown_symbols.items():
            f.write(f"{symbol}\t{count}\n")
