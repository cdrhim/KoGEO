import os

import glob

import csv
import json

import numpy as np

from utils import solver
solvers = solver()

def isfloat(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def preprocess(datapath,output_root=None):
    # Preprocess B.K's dataset for GEO and TM-generation model
    answers = []

    for path in glob.glob(datapath):
        if 'orig' in path:
            continue

        if 'fold' in path:
            num_valid = path.split('/')[-1].split('_')[1].replace('fold','')
            out_dir = os.path.join(output_root,'fold{}'.format(num_valid))
        else:
            validnum = str(0)
            out_dir = os.path.join(output_root,'fold{}'.format(validnum))

        out_token = os.path.join(out_dir, 'tokens')
        os.makedirs(out_token, exist_ok=True)

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t")
            if 'train' in path:
                datatype = 'train.tsv'
            elif 'test' in path:
                datatype = 'test.tsv'
            else:
                datatype = 'dev.tsv'
            output_path = os.path.join(out_token,datatype)

            with open(output_path, "w",encoding="utf-8-sig") as output_file:
                for n, line in enumerate(reader):
                    print("{} is completed.".format(n))

                    line = json.loads(line[0])
                    index = int(line['id'].split('_')[-1].split('-')[-1])+1

                    question = line['text'].split(' ')
                    numbers = line['numbers']

                    # generate number dictionary
                    initial = 0
                    nnum = 0
                    number_dict = {}
                    for nn,number in enumerate(numbers):
                        if '.' in number['value']:
                            number['value'] = str(round(float(number['value']),2))
                            if str(round(float(number['value']),2))[-2:] == '.0':
                                number['value'] = str(int(float(number['value'])))

                        number_dict['N{}'.format(nn)] = number['value']

                        if number['is_text'] == False:
                            question[initial + number['token'][0]] = number['value']

                        question.insert(initial + number['token'][0] + 1, 'N{}'.format(nnum))

                        nnum += 1
                        initial += 1

                    # generate number dictionary
                    output_expression = ''
                    expressions = line['expr']
                    for en, expr in enumerate(expressions):
                        if en != len(expressions) - 1:
                            output_expression += ' ' + expr[1]
                        else:
                            if expr[1] not in answers:
                                answers.append(expr[1])

                            output_expression += ' EOS{}'.format(answers.index(expr[1]))

                    output_expression = output_expression.replace('N_','N').replace('X_','X').replace('C_','C').replace('_','.')\
                        .replace('divide','/').replace('multiply','*').replace('subtract','-').replace('add','+').rstrip().lstrip()

                    exp_list = [str(e+'=').rstrip().lstrip().split(' ') for ne,e in enumerate(output_expression.replace('C','').split('=')[:-1])]
                    infix_exp_lists = [solvers.getInfix(post) for post in exp_list]
                    for ni,inexp in enumerate(infix_exp_lists):
                        for key in number_dict:
                            infix_exp_lists[ni] = infix_exp_lists[ni].replace(key,number_dict[key])

                    variables = list(set([tok for tok in output_expression.split(' ') if 'X' in tok]))
                    solutions = solvers.solve(infix_exp_lists,variables)
                    solutions = [str(solution) for solution in solutions]
                    if solutions == False or len(solutions)==0:
                        solutions = [-100]

                    expressions = output_expression.split(' ')
                    out = []
                    numout = []
                    init = 0

                    for en, expr in enumerate(expressions):
                        if expr == '':
                            continue
                        init += 1
                        out.append(expr)
                        if 'N' not in expr:
                            numout.append(expr)

                    # assign each token to each operator
                    op = 0
                    op_element = 0

                    expression_token_opindex = []
                    opdict = {}

                    for en, expr in enumerate(expressions[::-1]):
                        if expr == '':
                            continue

                        if ('EOS' not in expr) and ('N' not in expr) and ('C' not in expr) and ('X' not in expr):
                            op += 1
                            op_element = 0
                            opdict[op] = expr
                        else:
                            if op != 0:
                                op_element += 1

                        if op_element > 2:
                            expression_token_opindex.append([expr, op - (op_element - 2)])
                            continue

                        if op > 0:
                            if op_element == 2 and opdict[op] in ['+','*']:
                            # X2 X1 + -> X1 X2 +
                                if 'X' in expr and 'X' in expressions[::-1][en - 1]:
                                    if int(expr.replace('X', '')) > int(expressions[::-1][en - 1].replace('X', '')):
                                        expression_token_opindex[-1] = [expr, op]
                                        expression_token_opindex.append([expressions[::-1][en - 1], op])
                                        continue
                        expression_token_opindex.append([expr, op])

                    output_expression = [ch[0] for ch in expression_token_opindex[::-1]]
                    output_opidx = [ch[1] for ch in expression_token_opindex[::-1]]


                    operators = ['-', '/', '*', '+','EOS', '=','^']
                    groupdiff = []
                    implicit_pairs = []

                    for n1,number1 in enumerate(numbers):
                        groupdiff_temp = [-1]
                        ids = np.where(np.array(output_expression) == 'N{}'.format(n1))

                        if len(ids[0]) == 0:
                            implicit_pairs.append('0')
                        else:
                            implicit_operators = []
                            implicit_variables = []

                            for d1 in ids[0]:
                                if d1 > 0:
                                    if 'X' in output_expression[d1-1] or 'X' in output_expression[d1+1]:
                                        implicit_variables.append(True)
                                        continue

                                else:
                                    if 'X' in output_expression[d1 + 1]:
                                        implicit_variables.append(True)
                                        continue

                                for ni, oid in enumerate(output_expression[d1:]):
                                    if (oid in operators):
                                        implicit_operators.append(operators.index(oid))
                                        continue

                            if True in implicit_variables:
                                implicit_pairs.append("1")
                            else:
                                implicit_pairs.append(str(implicit_operators[0] + 2))

                        for n2, number2 in enumerate(numbers):
                            if n2 <= n1:
                                continue

                            if len(ids[0]) == 0:
                                groupdiff_temp.append(0)
                                continue

                            ids2 = np.where(np.array(output_expression) == 'N{}'.format(n2))

                            if len(ids2[0]) ==0:
                                groupdiff_temp.append(0)
                                continue

                            gp_temp = []
                            for d1 in ids[0]:
                                opd1 = output_opidx[d1]

                                for d2 in ids2[0]:
                                    opd2 = output_opidx[d2]
                                    gp_temp.append(abs(opd2-opd1))

                            groupdiff_temp.append(int(np.min(gp_temp)))

                        if len(groupdiff_temp) > 1:
                            groupdiff.extend([str(gd) for gd in groupdiff_temp])

                    output_file.write('\t'.join([str(index), ' '.join(question),' '.join(output_expression),
                                                 ' '.join(groupdiff),' '.join(solutions),' '.join([number_dict[key] for key in number_dict]),
                                                 ' '.join(implicit_pairs)]) + '\n')
            f.close()

if __name__ == "__main__":
    # datapath = r'dataset/mawps/*.jsonl'
    # datapath = r'dataset/Math23k/*.jsonl'
    datapath = r'dataset/alg514/*.jsonl'
    # datapath = r'dataset/draw/*.jsonl'
    # datapath = r'/home/dg/PycharmProjects/Deep-Reinforcement-Learning-Hands-On/Chapter18/dataset/oalg514/raw/*.jsonl'
    # datapath = r'dataset/IL/*.jsonl'
    outputfolder = r'dataset/alg514'
    preprocess(datapath,outputfolder)
