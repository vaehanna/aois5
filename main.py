import re
import numpy
import itertools, functools

import re
import numpy
from enum import Enum



class TYPE_OF_FUNC(Enum):
    CONJUNCTIVE = 0
    DISJUNCTIVE = 1

class ImplicantIsNecessary(Exception):
    pass

class ImplicantIsUnnecessary(Exception):
    pass

def translate_to_implicant(pair, entrance):
    surrounding_translator = [
        [('~a', '~b'), ('~a', '~c'), ('~b', '~c'), ('~b'), ('~c'), ('~a', '~b', '~c')],
        [('~a', 'c'), ('~a', '~b'), ('~b', 'c'), ('c'), ('~b'), ('~a', '~b', 'c')],
        [('~a', 'b'), ('~a', 'c'), ('b', 'c'), ('b'), ('c'), ('~a', 'b', 'c')],
        [('~a', '~c'), ('~a', 'b'), ('b', '~c'), ('~c'), ('b'), ('~a', 'b', '~c')],
        [('a', '~b'), ('a', '~c'), ('~b', '~c'), ('~b'), ('~c'), ('a', '~b', '~c')],
        [('a', 'c'), ('a', '~b'), ('~b', 'c'), ('c'), ('~b'), ('a', '~b', 'c')],
        [('a', 'b'), ('a', 'c'), ('b', 'c'), ('b'), ('c'), ('a', 'b', 'c')],
        [('a', '~c'), ('a', 'b'), ('b', '~c'), ('~c'), ('b'), ('a', 'b', '~c')],
        ]
    index = 0
    for row in range(2):
        for col in range(4):
            if pair == (row, col):
                return surrounding_translator[index][entrance]
            index += 1
    raise Exception("Index is out of range!")


def find_type_of_function(function):
    result_flag = re.search(r"^~?(\w|True|False)(\*~?(\w|True|False))*(\s\+\s~?(\w|True|False)(\*~?(\w|True|False))*)*$", function)
    type_of_func = TYPE_OF_FUNC.DISJUNCTIVE
    if result_flag is None:
        result_flag = re.search(r"^\(?~?\w(\+~?\w\)?)*(\s\*\s\(?~?\w(\+~?\w\)?)*)*$", function)
        type_of_func = TYPE_OF_FUNC.CONJUNCTIVE
    if result_flag is None:
        raise Exception("Invalid input!")
    return type_of_func

#gets array of implicants as an input
def convert_to_eval(implicants, function_type):
        ##WARN! Could break if you'll change the Enum "TYPE_OF_FUNC"
        #concatenate expression to one string
        if (function_type == TYPE_OF_FUNC.CONJUNCTIVE):
            implicants = [" or ".join(impl) for impl in implicants]
            eval_string = "(" + ") and (".join(implicants) + ")"
        else:
            #convert individual implicants to strings
            implicants = [" and ".join(impl) for impl in implicants]
            eval_string = "(" + ") or (".join(implicants) + ")"
        eval_string = re.sub('~', ' not ', eval_string)
        return eval_string

def convert_to_human(implicants, function_type):
        ##WARN! Could break if you'll change the Enum "TYPE_OF_FUNC"
        #concatenate expression to one string
        if (function_type == TYPE_OF_FUNC.CONJUNCTIVE):
            implicants = ["+".join(impl) for impl in implicants]
            eval_string = "(" +") * (".join(implicants) + ")"
        else:
            #convert individual implicants to strings
            implicants = ["*".join(impl) for impl in implicants]
            eval_string = "(" + ") + (".join(implicants)  + ")"
        eval_string = re.sub(' not ', '~', eval_string)
        return eval_string


def represent_in_values(function, implicant, first_value, second_value):
    char_1 = re.search(r'\w', implicant[0])
    char_1 = char_1.group()
    function = re.sub(char_1, str(first_value), function)
    char_2 = re.search(r'\w', implicant[1])
    char_2 = char_2.group()
    function = re.sub(char_2, str(second_value), function)
    function = re.sub(r'~0', 'True', function)
    function = re.sub(r'~1', 'False', function)
    function = re.sub(r'1', 'True', function)
    function = re.sub(r'0', 'False', function)
    return function


def split_function(function):
    type_of_func = find_type_of_function(function)
    function_splitted = []

    if type_of_func == TYPE_OF_FUNC.DISJUNCTIVE:
        function = function.split(' + ')
        for elem in function:
            function_splitted.append(elem.split('*'))
    else:
        function = function.replace('(', '')
        function = function.replace(')', '')
        function = function.split(' * ')
        for elem in function:
            function_splitted.append(elem.split('+'))
    return function_splitted


def joining_rule(function):
    type_of_func = find_type_of_function(function)
    function_splitted = split_function(function)

    result = set()
    first_elem_index = 0
    while first_elem_index < len(function_splitted):
        second_elem_index = 0
        while second_elem_index < len(function_splitted):
            first_el = function_splitted[first_elem_index]
            second_el = function_splitted[second_elem_index]
            if len(set(first_el) ^ set(second_el)) == 2:
                sum = set(first_el) & set(second_el)
                function_splitted.remove(first_el)
                function_splitted.remove(second_el)
                if type_of_func == TYPE_OF_FUNC.DISJUNCTIVE:
                    result.add("*".join(sum))
                else:
                    result.add("+".join(sum))
                first_elem_index = 0
                second_elem_index = 0
                continue
            second_elem_index += 1
        first_elem_index += 1
    if type_of_func == TYPE_OF_FUNC.DISJUNCTIVE:
        result = " + ".join(result)
    else:
        result = " * ".join(result)
    for remaining in function_splitted:
        if type_of_func == TYPE_OF_FUNC.DISJUNCTIVE:
            result += ' + ' + "*".join(remaining)
        else:
            result += ' * ' + "+".join(remaining)
    return result


def find_kernel(perfect_form):
    type_of_func = find_type_of_function(perfect_form)
    simple_form = joining_rule(perfect_form)
    perfect_form_splitted = split_function(perfect_form)
    simple_form_splitted = split_function(simple_form)

    table = numpy.zeros(shape=(len(simple_form_splitted),
                               len(perfect_form_splitted)))

    for constituent in range(len(perfect_form_splitted)):
        for implicant in range(len(simple_form_splitted)):
            if set(simple_form_splitted[implicant]).issubset(perfect_form_splitted[constituent]):
                table[implicant][constituent] = 1
    kernel = []
    kernel_search_result = numpy.count_nonzero(table == 1, axis=0)
    for index in range(len(kernel_search_result)):
        if kernel_search_result[index] == 1:
            for element in range(len(table[:, index])):
                if table[element][index] == 1:
                    if simple_form_splitted[element] not in kernel:
                        kernel.append(simple_form_splitted[element])
    kernel_result = ""
    if type_of_func == TYPE_OF_FUNC.DISJUNCTIVE:
        for implicant in range(len(kernel)):
            kernel[implicant] = "*".join(kernel[implicant])
            kernel_result += kernel[implicant] + ' + '
    else:
        for implicant in range(len(kernel)):
            kernel[implicant] = "+".join(kernel[implicant])
            kernel_result += '(' + kernel[implicant] + ')' + ' * '
    kernel_result = kernel_result[: len(kernel_result) - 3]
    return kernel_result





def find_surrounding(KMap, position):
    surrounding = numpy.zeros(shape=(2, 3))
    surrounding[position[0]][1] = KMap[position]
    if position[0] == 0:
        if position[1] == 0:
            surrounding[0][0] = KMap[position[0]][KMap.shape[1] - 1]
            surrounding[0][2] = KMap[position[0]][position[1] + 1]
            surrounding[1][0] = KMap[position[0] + 1][KMap.shape[1] - 1]
            surrounding[1][1] = KMap[position[0] + 1][position[1]]
            surrounding[1][2] = KMap[position[0] + 1][position[1] + 1]
        elif position[1] == KMap.shape[1] - 1:
            surrounding[0][0] = KMap[position[0]][position[1] - 1]
            surrounding[0][2] = KMap[position[0]][0]
            surrounding[1][0] = KMap[position[0] + 1][position[1] - 1]
            surrounding[1][1] = KMap[position[0] + 1][position[1]]
            surrounding[1][2] = KMap[position[0] + 1][0]
        else:
            surrounding[0][0] = KMap[position[0]][position[1] - 1]
            surrounding[0][2] = KMap[position[0]][position[1] + 1]
            surrounding[1][0] = KMap[position[0] + 1][position[1] - 1]
            surrounding[1][1] = KMap[position[0] + 1][position[1]]
            surrounding[1][2] = KMap[position[0] + 1][position[1] + 1]
    elif position[0] == 1:
        if position[1] == 0:
            surrounding[0][0] = KMap[position[0] - 1][KMap.shape[1] - 1]
            surrounding[0][1] = KMap[position[0] - 1][position[1]]
            surrounding[0][2] = KMap[position[0] - 1][position[1] + 1]
            surrounding[1][0] = KMap[position[0]][KMap.shape[1] - 1]
            surrounding[1][2] = KMap[position[0]][position[1] + 1]
        elif position[1] == KMap.shape[1] - 1:
            surrounding[0][0] = KMap[position[0] - 1][position[1] - 1]
            surrounding[0][1] = KMap[position[0] - 1][position[1]]
            surrounding[0][2] = KMap[position[0] - 1][0]
            surrounding[1][0] = KMap[position[0]][position[1] - 1]
            surrounding[1][2] = KMap[position[0]][0]
        else:
            surrounding[0][0] = KMap[position[0] - 1][position[1] - 1]
            surrounding[0][1] = KMap[position[0] - 1][position[1]]
            surrounding[0][2] = KMap[position[0] - 1][position[1] + 1]
            surrounding[1][0] = KMap[position[0]][position[1] - 1]
            surrounding[1][2] = KMap[position[0]][position[1] + 1]
    return surrounding


def to_end_form(function):
    print("Joining rule: ", joining_rule(function))
    print("minimize computation: ", str(find_odd(function)))
    print("minimize Quine", minimize_Quine(function))

def find_odd(function):
    function = joining_rule(function)
    function_splitted = split_function(function)
    function_type = find_type_of_function(function)

    # Determining the number of variables in the function (by searching for every unique letter)
    function_variables = set(re.findall(r'[a-z]', function))

    for implicant in function_splitted:
        try:
            # find all variables used by an implicant
            implicant_variables = set(re.findall(r'[a-z]', "".join(implicant)))
            function_remainder = function_splitted[:]
            function_remainder.remove(implicant)

            # generate valuesets for the implicant to
            implicant_combinations = itertools.product(range(2), repeat=len(implicant_variables))

            # convert them to a dict for eval (looks like {"var1": value_1, "var2": value_2} )
            # implicant_valuesets = []
            # for combination in implicant_combinations:
            #     implicant_valuesets.insert
            implicant_valuesets = [{list(implicant_variables)[ind]: val for ind, val in enumerate(combination)} for combination in implicant_combinations]

            # finding the valueset when implicant is equal to 0|1
            control_valueset = {}
            for valueset in implicant_valuesets:
                if (eval(convert_to_eval([implicant], function_type), valueset) == function_type.value):
                    control_valueset = valueset

            # if the remaining function is still 0|1 on this valueset, it means that the implicant is unnecessary
            # let's complete the valueset we're checking to all combinations of variables in the remaining function
            variables_to_complete = function_variables - implicant_variables
            # generate combinations
            combinations_for_completion = itertools.combinations(range(2), len(variables_to_complete))

            # for each combination of remaining variables, create a valueset of control_valueset (implicant) + variables_to_complete combination
            combinations_for_completion = [{list(variables_to_complete)[ind]: val for ind, val in enumerate(valueset)} for valueset in combinations_for_completion]

            # valueset = completion variables, control_valueset = valueset when implicant is equal to 1|0, | is the "+" for dicts, basically (merge)
            for valueset in combinations_for_completion:
                # if on any valueset we generated the value is not the same as on implicant, the implicant is NOT unnecessary
                if (eval(convert_to_eval(function_remainder, function_type), {**valueset, **control_valueset}) is not function_type.value):
                    raise ImplicantIsNecessary()

            # this code is unaccessable unless the previous check shown that implicant is UNnecessary
            function_splitted.remove(implicant)

        except ImplicantIsNecessary:
            continue
    return convert_to_human(function_splitted, function_type)


def minimize_Quine(perfect_form):
    type_of_func = find_type_of_function(perfect_form)
    simple_form = joining_rule(perfect_form)
    perfect_form_splitted = split_function(perfect_form)
    simple_form_splitted = split_function(simple_form)

    table = numpy.zeros(shape=(len(simple_form_splitted),
                               len(perfect_form_splitted)))

    for constituent in range(len(perfect_form_splitted)):
        for implicant in range(len(simple_form_splitted)):
            if set(simple_form_splitted[implicant]).issubset(perfect_form_splitted[constituent]):
                table[implicant][constituent] = 1
    kernel = find_kernel(perfect_form)
    kernel_splitted = split_function(kernel)
    arr = numpy.count_nonzero(table == 1, axis=1)
    result = []
    for index in range(len(arr)):
        if arr[index] == 1 or simple_form_splitted[index] in kernel_splitted:
            result.append(simple_form_splitted[index])
    minimal_form = ""
    if type_of_func == TYPE_OF_FUNC.DISJUNCTIVE:
        for implicant in range(len(result)):
            result[implicant] = "*".join(result[implicant])
            minimal_form += result[implicant] + ' + '
    else:
        for implicant in range(len(result)):
            result[implicant] = "+".join(result[implicant])
            minimal_form += '(' + result[implicant] + ')' + ' * '
    return minimal_form[: len(minimal_form) - 3]

def string_formula(formula):
    arg_length = 3
    inside = '*'
    outside = '+'
    substring = []
    for i in range(len(formula)):
        args = []
        for j in range(len(formula[i])):
            if formula[i][j] == 0:
                args.append('!x' + str(j + 1))
            if formula[i][j] == 1:
                args.append('x' + str(j + 1))
        substring.append(inside.join(args))
        if len(substring[-1]) > arg_length:
            substring[-1] = '(' + substring[-1] + ')'
    output = outside.join(substring)
    return output


def PDNF(table, arguments_number):
    formula = []
    arguments = create_dictionary(arguments_number)
    for i in range(len(table)):
        if table[i] == 1:
            bracket = []
            for arg_index in range(1, arguments_number + 1):
                bracket.append(arguments['x' + str(arg_index)][i])
            formula.append(bracket)
    return formula


def create_dictionary(arguments_number):
    dictionary = []
    for i in range(arguments_number):
        index = i + 1
        same = 2 ** (arguments_number - index)
        array = [0 for j in range(same)]
        array += [1 for j in range(same)]
        while len(array) < 2 ** (arguments_number):
            array += array
        dictionary.append(['x' + str(index), array])
    dictionary = dict(dictionary)
    return dictionary


def summator_table():
    arguments_number = 3
    arguments = create_dictionary(arguments_number)
    b = [0 for i in range(len(arguments['x1']))]
    d = b.copy()
    for i in range(len(arguments['x1'])):
        sum = arguments['x1'][i] + arguments['x2'][i] + arguments['x3'][i]
        if sum >= 2:
            b[i] = 1
            sum -= 2
        if sum == 1:
            d[i] = 1
    return d, b


def is_mergable(constit1, constit2, arg_index, arguments_number):
    mergability = True
    for i in range(arguments_number):
        if i != arg_index and constit1[i] != constit2[i]:
            mergability = False
            break
        if i == arg_index and constit1[i] == constit2[i]:
            mergability = False
            break
    return mergability


def merge(formula, arguments_number):
    merged = []
    unmerged = []
    used = [False for i in range(len(formula))]
    for i in range(arguments_number):
        for j in range(len(formula) - 1):
            for k in range(j + 1, len(formula)):
                if is_mergable(formula[j], formula[k], i, arguments_number):
                    used[j] = True
                    used[k] = True
                    merged.append(formula[j].copy())
                    merged[-1].pop(i)
                    merged[-1].insert(i, -1)
                    break
    for i in range(len(used)):
        if not (used[i]):
            unmerged.append(formula[i])
    return merged, unmerged


def substitute(values, formula):
    for i in range(len(values)):
        if values[i] == -1:
            missed_value = i
    for i in range(len(formula)):
        if formula[i] != -1 and i != missed_value:
            existing_arg = i
    res = []
    res.append(formula[missed_value])
    if formula[existing_arg] == values[existing_arg]:
        res.append(1)
    else:
        res.append(0)
    return res


def delete_excess(formula):
    new_formula = formula.copy()
    no_change = 1
    i = 0
    while i < len(new_formula):
        res = []
        for other in new_formula:
            if new_formula[i] != other:
                sub = substitute(new_formula[i], other)
                if sub[1] == no_change:
                    res.append(sub[0])
        pos, neg = False, False
        for arg in res:
            if arg == 0: neg = True
            if arg == 1: pos = True
        if pos and neg:
            new_formula.pop(i)
        else:
            i += 1
    return new_formula


def delete_identical(formula):
    i = 0
    while i < len(formula) - 1:
        same = False
        for j in range(i + 1, len(formula)):
            if formula[i] == formula[j]:
                same = True
        if same:
            formula.pop(i)
        else:
            i += 1
    return formula


def simplify(formula):
    arguments_number = len(formula[0])
    i = arguments_number
    simplified = []
    merged = formula
    while i > 1:
        merged, unmerged = merge(merged, arguments_number)
        merged = delete_excess(merged)
        simplified += unmerged
        i -= 1
    simplified += merged
    formula = delete_identical(simplified)
    return simplified
V_index = 3
arguments_number = 3

def transition_table():
    arguments = create_dictionary(arguments_number+1)
    table = []
    for q in arguments.values():
        table.append(q)
    table += current_q(arguments)
    h = [[0 for m in range(len(arguments['x1']))] for n in range(arguments_number)]
    for i in range(len(h)):
        for j in range(len(h[i])):
            if table[i][j] != table[i+V_index+1][j]:
                h[i][j] = 1
    table += h
    return table

def current_q(arguments):
    arguments_number = 3
    q = [[0 for m in range(len(arguments['x1']))] for n in range(arguments_number)]
    for i in range(len(arguments['x1'])):
        V = [0 for j in range(arguments_number-1)]+[arguments['x'+str(arguments_number+1)][i]]
        index = arguments_number
        plusone = 0
        while index > 0:
            sum = arguments['x'+str(index)][i] + V[index-1] + plusone
            plusone = 0
            if sum >= 2:
                sum -= 2
                plusone = 1
            q[index-1][i] = sum
            index -= 1
    return q

table = transition_table()
string_names = ['q3*','q2*','q1*','V', 'q3','q2','q1','h3','h2','h1']
print('Двоичный счетчик накапливающего типа на 8 внутренних состояний в базисе НЕ И-ИЛИ и Т-триггер')
for i in range(len(table)):
    print(string_names[i].ljust(5) + ' '.join([str(el) for el in table[i]]))

h_index = 7
for i in range(h_index, len(table)):
    h_pdnf = PDNF(table[i], arguments_number+1)
    h_simplified = simplify(h_pdnf)
    h_pdnf = string_formula(h_pdnf)
    h_pdnf = h_pdnf.replace('x', 'q')
    h_pdnf = h_pdnf.replace('q4', 'V')
    h_pdnf = h_pdnf.replace('1', '4')
    h_pdnf = h_pdnf.replace('3', '1')
    h_pdnf = h_pdnf.replace('4', '3')
    h_simplified = string_formula(h_simplified)
    h_simplified = h_simplified.replace('x', 'q')
    h_simplified = h_simplified.replace('q4', 'V')
    h_simplified = h_simplified.replace('1', '4')
    h_simplified = h_simplified.replace('3', '1')
    h_simplified = h_simplified.replace('4', '3')
    print('\nPDNF(h'+str(len(table)-i)+'): ' + h_pdnf)
    print('PDNF(h'+str(len(table)-i)+'): ' + h_simplified)
    logism_grammar = h_simplified.replace('!', '~')
    logism_grammar = logism_grammar.replace('*', '&')
    print('PDNF(h'+str(len(table)-i)+') for logism: ' + logism_grammar)