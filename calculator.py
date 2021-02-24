from sympy import *
import signal
from contextlib import contextmanager
from multiprocessing import Pool

class TimeoutException(Exception):
    """
    Timeout Exception to handle this type of exception
    """
    pass
def time_limit(timeout, func, *args, **kwargs):
    """ Run func with the given timeout. If func didn't finish running
        within the timeout, raise TimeLimitExpired
    """
    with Pool(1) as pool:
        result = pool.apply_async(func, args=args, kwds=kwargs)
        try:
            return result.get(timeout=timeout)
        finally:
            pool.terminate()
# @contextmanager
# def time_limit(seconds):
#     """
#     [Context] limiting time for computing an expression
#     :param seconds: maximum amount of spent time for computation
#     """
#     def signal_handler(signum, frame):
#         raise TimeoutException("Timed out!")
#     signal.signal(signal.SIGALRM, signal_handler)
#     signal.alarm(seconds)
#     try:
#         yield
#     finally:
#         signal.alarm(0)

class solver(object):
    def __init__(self):
        self.precedence = {}
        self.precedence['^'] = 5
        self.precedence['='] = 4
        self.precedence['*'] = 3
        self.precedence['/'] = 3
        self.precedence['+'] = 2
        self.precedence['-'] = 2
        self.precedence['('] = 1
        self.precedence[')'] = 1
        self.seconds = 10
        self.infix_to_postfix = infix_to_postfix()

    def solve(self,exp_list,symbols,simplification =False):
        answers = []
        if not simplification:
            equation_list = [Eq(simplify(exp.replace('=', '-').replace('1.NEG', '(-1)'))) for exp in exp_list]
            symbols = [Symbol(symbol) for symbol in symbols]
            answer = solve(equation_list, symbols)

            for key in answer:
                try:
                    answers.append(round(float(answer[key]), 2))
                except:
                    answers.append(round(float(key[0]), 2))
        else:
            answer = simplify(exp_list)
            for key in answer:
                answers.append(round(float(key), 2))
        return answers


    def getInfix(self,exp_list):
        exp_list = [e for e in exp_list.split(' ') if e != '']
        stack=[]
        for j in range(len(exp_list)) :
            if self.operand(exp_list[j]) :
                stack.append(exp_list[j])
            else:
                operator1=stack.pop()
                try:
                    operator2 = stack.pop()
                except:
                    if exp_list[j] in ['-','+']:
                        operator2 = '0'
                    else:
                        operator2 = '1'
                stack.append("(" + operator2 + exp_list[j] + operator1 + ")")
        return stack.pop()

    def operand(self,char) :
        if char not in self.precedence:
            return True
        return False

    def toPostfixEquations(self,exp_list, symbol):
        # symbol_list = symbols(symbol)
        out_list = []
        for ne,exp in enumerate(exp_list):
            exp = simplify(exp.replace('=','-'))
            exp = str(exp)
            for e in self.precedence:
                exp = exp.replace(e,' {} '.format(e))
            expr = [e for e in exp.split(' ') if e != '']
            posfixexp = self.infix_to_postfix.infixtopostfix(expr)
            out_list.append(posfixexp)
        return out_list
class infix_to_postfix:
    precedence = {'^': 5, '*': 4, '/': 4, '+': 3, '-': 3, '(': 2, ')': 1}

    def __init__(self):
        self.items = []
        self.size = -1

    def push(self, value):
        self.items.append(value)
        self.size += 1

    def pop(self):
        if self.isempty():
            return 0
        else:
            self.size -= 1
            return self.items.pop()

    def isempty(self):
        if (self.size == -1):
            return True
        else:
            return False

    def seek(self):
        if self.isempty():
            return false
        else:
            return self.items[self.size]

    def isOperand(self, i):
        if i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return True
        else:
            return False

    def infixtopostfix(self, expr):
        postfix = ""
        print('postfix expression after every iteration is:')
        for i in expr:

            if (i in '+-*/^'):
                while (len(self.items) and self.precedence[i] <= self.precedence[self.seek()]):
                    postfix += self.pop()
                self.push(i)
            elif i is '(':
                self.push(i)
            elif i is ')':
                o = self.pop()
                while o != '(':
                    postfix += o
                    o = self.pop()
            else:
                postfix += i
            print(postfix)
            # end of for
        while len(self.items):
            if (self.seek() == '('):
                self.pop()
            else:
                postfix += self.pop()
        return postfix