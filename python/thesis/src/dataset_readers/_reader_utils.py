def process_multiword_tokens(annotation):
    """
    Taken from: https://github.com/Hyperparticle/udify/blob/master/udify/dataset_readers/universal_dependencies.py

    Processes CoNLLU annotations for multi-word tokens.
    If the token id returned by the conllu library is a tuple object (either a multi-word token or an elided token),
    then the token id is set to None so that the token won't be used later on by the model.
    """
    
    for i in range(len(annotation)):
        conllu_id = annotation[i]["id"]
        if type(conllu_id) == tuple:
            if "-" in conllu_id:
                conllu_id = str(conllu_id[0]) + "-" + str(conllu_id[2])
                annotation[i]["multi_id"] = conllu_id
                annotation[i]["id"] = None
            elif "." in conllu_id:
                annotation[i]["id"] = None
                annotation[i]["multi_id"] = None
        else:
            annotation[i]["multi_id"] = None
    
    return annotation



def min_edit_script(source, target, allow_copy=False):
    """
    Taken from: https://github.com/Hyperparticle/udify/blob/master/udify/dataset_readers/lemma_edit.py

    Finds the minimum edit script to transform the source to the target
    """
    a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(0, len(source) + 1):
        for j in range(0, len(target) + 1):
            if i == 0 and j == 0:
                a[i][j] = (0, "")
            else:
                if allow_copy and i and j and source[i - 1] == target[j - 1] and a[i-1][j-1][0] < a[i][j][0]:
                    a[i][j] = (a[i-1][j-1][0], a[i-1][j-1][1] + "→")
                if i and a[i-1][j][0] < a[i][j][0]:
                    a[i][j] = (a[i-1][j][0] + 1, a[i-1][j][1] + "-")
                if j and a[i][j-1][0] < a[i][j][0]:
                    a[i][j] = (a[i][j-1][0] + 1, a[i][j-1][1] + "+" + target[j - 1])
    return a[-1][-1][1]


def gen_lemma_rule(form, lemma, allow_copy=False):
    """
    Taken from: https://github.com/Hyperparticle/udify/blob/master/udify/dataset_readers/lemma_edit.py

    Generates a lemma rule to transform the source to the target
    """
    form = form.lower()

    previous_case = -1
    lemma_casing = ""
    for i, c in enumerate(lemma):
        case = "↑" if c.lower() != c else "↓"
        if case != previous_case:
            lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case, i if i <= len(lemma) // 2 else i - len(lemma))
        previous_case = case
    lemma = lemma.lower()

    best, best_form, best_lemma = 0, 0, 0
    for l in range(len(lemma)):
        for f in range(len(form)):
            cpl = 0
            while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl] == lemma[l + cpl]: cpl += 1
            if cpl > best:
                best = cpl
                best_form = f
                best_lemma = l

    rule = lemma_casing + ";"
    if not best:
        rule += "a" + lemma
    else:
        rule += "d{}¦{}".format(
            min_edit_script(form[:best_form], lemma[:best_lemma], allow_copy),
            min_edit_script(form[best_form + best:], lemma[best_lemma + best:], allow_copy),
        )
    return rule
