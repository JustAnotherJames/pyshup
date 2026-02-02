from __future__ import annotations
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from enum import Enum
import string
import random
import copy

from typing import TypeVar, overload, cast

'''
program -> rootblock

rootblock -> block | function

function -> Function(...)
            
block -> stmt*

stmt -> assign | raw | comment | control | Print | ExprStmt

assign -> variable '=' expr

raw -> Raw(...)

comment -> Comment(...)

control -> if_clause | while_clause

if_clause -> 'if' test 'then' block else_part 'fi'
           | 'if' test 'then' block 'fi'
           
else_part -> 'elif' expr 'then' block else_part
           | 'else' block

while_clause -> 'while' test 'do' block 'done'

test ->  '[' expr '-eq' 0 ']'

ExprStmt -> expr

expr -> variable
      | literal
      | command
      | binaryOperation
      | unaryOperation
      | functionCall
      | group_expr
    
variable -> Variable(...)

literal -> Literal()

command -> 'capture' expr param_list variable variable variable

param_list -> '(' expr* ')'

binaryOperation -> ArithmeticOperation | ComparisonOperation

ArithmeticOperation -> expr ArithmeticOperator expr

ArithmeticOperator -> '+' | '-' | '*' | '/' | '%'
    
ComparisonOperation -> expr ComparisonOperator expr

ComparisonOperator -> '<' | '<=' | '>' | '>=' | '==' | '~='
                    | 'and' | 'or'

unaryOperation -> unop expr

unop -> '-'

functionCall -> expr param_list
    
group_expr -> '(' expr ')'

'''

capture = """capture() {
    outvar=$1
    errvar=$2
    statusvar=$3
    shift 3

    stdout_tmp=$(mktemp "${TMPDIR:-/tmp}/cap.out.XXXXXX") || return 1
    stderr_tmp=$(mktemp "${TMPDIR:-/tmp}/cap.err.XXXXXX") || {
        rm -f "$stdout_tmp"
        return 1
    }

    "$@" >"$stdout_tmp" 2>"$stderr_tmp"
    status=$?

    stdout=$(cat "$stdout_tmp")
    stderr=$(cat "$stderr_tmp")

    rm -f "$stdout_tmp" "$stderr_tmp"

    eval "$outvar=\\$stdout"
    eval "$errvar=\\$stderr"
    eval "$statusvar=\\$status"

    return 0
}"""

class Environment:
    def __init__(self):
        self.needs_capture = False

T = TypeVar("T", bound="Node")

class Node(ABC):
    children = []
    
    def __init__(self):
        self.parent: Node | None = None
        #self.children: list[Node] = []
    
    @overload
    def adopt(self, node: None) -> None: ...
    
    @overload
    def adopt(self, node: T) -> T: ...
    
    @overload
    def adopt(self, node: list[T]) -> list[T]: ...
    
    def adopt(self, node):
        if node is None:
            return None
        
        if isinstance(node, list):
            for n in node:
                if not n.parent:
                    n.parent = self
                #self.children.append(n)
        else:
            if not node.parent:
                node.parent = self
            #if node != self:
            #    self.children.append(node)
        return node
        

class CoreNode(Node):
    pass

class RootBlock(CoreNode):
    children = ['functions', 'statements']
    def __init__(self, statements: list[Statement | Function] | None = None):
        super().__init__()
        self.statements: list[Statement] = []
        self.functions: list[Function] = []
        self.parent = self.adopt(self)
        
        if statements:
            for x in statements:
                if isinstance(x, Function):
                    self.functions.append(self.adopt(x))
                elif isinstance(x, Statement):
                    self.statements.append(self.adopt(x))
        
    
    def add(self, statement: Statement | Function):
        if isinstance(statement, Function):
            self.functions.append(self.adopt(statement))
        elif isinstance(statement, Statement):
            self.statements.append(self.adopt(statement))
        else:
            raise Exception(f'Cannot add {statement} to root block')
        return self
    
class Function(CoreNode):
    children = ['functionBody']
    def __init__(self, functionName: str, functionBody: Block):
        super().__init__()
        self.functionName = functionName
        self.functionBody = self.adopt(functionBody)

class Block(CoreNode):
    children = ['statements']
    def __init__(self, statements: list[Statement] | None = None):
        super().__init__()
        self.statements = self.adopt(statements or [])
    
    def add(self, statement: Statement | Function):
        if isinstance(statement, Function):
            raise Exception('cannot add Function to non-root block')
        self.statements.append(self.adopt(statement))
        return self
    
class Statement(CoreNode):
    pass

class Assign(Statement):
    children = ['lhs', 'rhs']
    def __init__(self, variable: Variable, rhs: Expression):
        super().__init__()
        self.lhs = self.adopt(variable)
        self.rhs = self.adopt(rhs)
    
class Raw(Statement):
    children = []
    def __init__(self, sh: str):
        super().__init__()
        self.sh = sh

class Comment(Statement):
    children = []
    def __init__(self, comment: str):
        super().__init__()
        self.comment = comment

class Control(Statement):
    pass

class If(Control):
    children = ['ifcondition', 'then', 'elseifconditions', 'elifthens', 'elsethen']
    def __init__(self, condition: Test, then: Block | None = None, elseifs: list[tuple[Test, Block]] | None = None, elsethen: Block | None = None):
        super().__init__()
        self.ifcondition = self.adopt(condition)
        self.then = self.adopt(then or Block())
        
        self.elseifconditions: list[Test] = []
        self.elifthens: list[Block] = []
        if elseifs:
            for (elifcond, elifthen) in elseifs:
                self.elseifconditions.append(self.adopt(elifcond))
                self.elifthens.append(self.adopt(elifthen))
        
        self.elsethen: Block | None = self.adopt(elsethen)

class While(Control):
    children = ['condition', 'then']
    def __init__(self, condition: Test, then: Block | None = None):
        super().__init__()
        self.condition = self.adopt(condition)
        self.then = self.adopt(then or Block())
        
class Test(CoreNode):
    def __init__(self, expression: Expression):
        super().__init__()
        self.expression = expression
        
class Print(Statement):
    def __init__(self, printExpression: Expression):
        super().__init__()
        self.printExpression = self.adopt(printExpression)
       
class ExpressionStatement(Statement):
    children = ['expression']
    def __init__(self, expression):
        super().__init__()
        self.expression = self.adopt(expression)
    
class Expression(CoreNode):
    pass

class Variable(Expression):
    children = []
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        
class Literal(Expression):
    children = []
    def __init__(self, value: int | str | bool):
        super().__init__()
        if isinstance(value, str):
            value = f'"{value}"'
        self.value = value
        
class Command(Expression):
    children = ['command', 'args', 'defined_stdout', 'defined_stderr', 'defined_returnCode']
    def __init__(self, command: Expression | None = None, args: list[Expression] | None = None, name: str | None = None):
        super().__init__()
        self.command = self.adopt(command)
        self.args = self.adopt(args)
        
        characters = string.ascii_letters + string.digits
        self.name = name or f"TMP_{''.join(random.choices(characters, k=5))}"
        
        self.defined_stdout = None
        self.defined_stderr = None
        self.defined_returnCode = None

    @property
    def stdout(self) -> Variable:
        if not self.defined_stdout:
            self.defined_stdout = self.adopt(Variable(f'{self.name}_stdout'))    
        return self.defined_stdout
    
    @property
    def stderr(self) -> Variable:
        if not self.defined_stderr:
            self.defined_stderr = self.adopt(Variable(f'{self.name}_stderr'))
        return self.defined_stderr
    
    @property
    def returnCode(self) -> Variable:
        if not self.defined_returnCode:
            self.defined_returnCode = self.adopt(Variable(f'{self.name}_returnCode'))
        return self.defined_returnCode
        
class BinaryOperationType(Enum):
    plus = '+'
    minus = '-'
    multiply = '*'
    divide = '/'
    equal = '=='
    notEqual = '!='
    lessThan = '<'
    greaterThan = '>'
    greaterThanEqual = '>='
    lessThanEqual = '<='
    
class BinaryOperation(Expression):
    children = ['lhs', 'rhs']
    def __init__(self, lhs: Expression, operator: BinaryOperationType, rhs: Expression):
        super().__init__()
        self.lhs = self.adopt(lhs)
        self.operator = operator
        self.rhs = self.adopt(rhs)
        
    plus=BinaryOperationType.plus
    minus=BinaryOperationType.minus
    multiply=BinaryOperationType.multiply
    divide=BinaryOperationType.divide
    equal=BinaryOperationType.equal
    notEqual=BinaryOperationType.notEqual
    lessThan=BinaryOperationType.lessThan
    greaterThan=BinaryOperationType.greaterThan
    greaterThanEqual=BinaryOperationType.greaterThanEqual
    lessThanEqual=BinaryOperationType.lessThanEqual
    
class UnaryOperationType(Enum):
    negate = '-'
    
class UnaryOperation(Expression):
    children = ['expr']
    def __init__(self, expr: Expression, operator: UnaryOperationType):
        super().__init__()
        self.expr = self.adopt(expr)
        self.operator = operator

    negate=UnaryOperationType.negate

class FunctionCall(Expression):
    children = ['args']
    def __init__(self, functionName: str, args: list[Expression] | None):
        super().__init__()
        self.functionName = functionName
        self.args = self.adopt(args or [])
        
class GroupExpression(Expression):
    children = ['expression']
    def __init__(self, expression: Expression):
        super().__init__()
        self.expression = self.adopt(expression)


class SugarNode(Node):
    pass

# contrived example of a sugar node since it's just a raw passthrough - same with the visitor
# this is done because of the complexity of Command due to not currently having a real variable table
class Echo(SugarNode, Command):
    def __init__(self, expr: Expression, name: str | None = None):
        super().__init__(Literal('echo'), [expr], name)
    



class Phase(ABC):
    def __init__(self, environment: Environment):
        self.environment = environment
    
    @singledispatchmethod
    def visit(self, node: Node):
        raise NotImplementedError(f"no visit implementation for {node}")
        
    def visit_all(self, rootNode: Node, *args, **kwargs) -> Node:
        return self.visit_traverse(rootNode)
    
    def visit_one(self, node: Node) -> Node:
        return self.visit(node)
    
    def visit_traverse(self, rootNode: Node) -> Node:
        rootNodeCopy = copy.deepcopy(rootNode)
        root = self.visit_one(rootNodeCopy)
        # a little hacky, but works for now. Mind the traversal order - can be wonky.
        stack = [root]
        while stack:
            curr = stack.pop()
            childrenNames = curr.children
            for childName in childrenNames:
                child: Node | list[Node] = getattr(curr, childName)
                if not child: continue
                try:
                    if isinstance(child, list):
                        children = child
                        for i in range(len(children)):
                            children[i] = self.visit_one(children[i])
                            stack.append(children[i])
                    else:
                        setattr(curr, childName, self.visit_one(child))
                        stack.append(child)
                except Exception as e: # throws on nodes who's children cannot be set (i.e Command's stdout/stderr/returnCode)
                    continue
        return root


class EnvironmentAssignment(Phase):
    @singledispatchmethod
    def visit(self, node: Node):
        return node
    
    @visit.register
    def _(self, node: Variable):
        parent = node.parent
        if isinstance(parent, Command):
            if parent.defined_stderr and parent.defined_stdout:
                self.environment.needs_capture = True
        return node
        
# Phase Emitter
class ShellRenderer:
    def __init__(self, environment, indent='\t'):
        self.indent: str = indent
        self.level: int = 0
        self.environment = environment
        
    def line(self, line: str) -> str:
        return f'{self.indent*(self.level - 1)}{line}\n'
        
    @singledispatchmethod
    def visit(self, node: CoreNode):
        raise NotImplementedError(f"No visit method for {type(node).__name__}")
    
    def visit_all(self, node: CoreNode) -> str:
        return self.visit(node)
    
    # Root Block
    @visit.register
    def _(self, node: RootBlock):
        lines = ''
        self.level += 1
        for function in node.functions:
            lines += self.visit(function)
            
        for stmt in node.statements:
            lines += self.visit(stmt)
        self.level -= 1
        return lines
        
    # Function
    @visit.register
    def _(self, node: Function):
        return \
            self.line(f'{node.functionName}()' + '{') + \
            self.visit(node.functionBody) + \
            self.line('}')
    
    # Block
    @visit.register
    def _(self, node: Block):
        lines = ''
        self.level += 1
        for stmt in node.statements:
            lines += self.visit(stmt)
        self.level -= 1
        return lines
        
    # Assign
    @visit.register
    def _(self, node: Assign):
        return self.line(f'{node.lhs.name}={self.visit(node.rhs)}')    
    
    # Raw
    @visit.register
    def _(self, node:Raw):
        return self.line(node.sh)
    
    # Comment
    @visit.register
    def _(self, node: Comment):
        return self.line(f'# {node.comment}')
        
    # If
    @visit.register
    def _(self, node: If):
        lines = ''
        if node.ifcondition:
            lines += self.line(f'if {self.visit(node.ifcondition)}; then')
            lines += self.visit(node.then) if node.then else self.line(':')

        for (condition, then) in zip(node.elseifconditions, node.elifthens):
            lines += self.line(f'elif {self.visit(condition)}; then')
            lines += self.visit(then)
                
        if node.elsethen:
            lines += self.line('else')
            lines += self.visit(node.elsethen)
                
        lines += self.line(f'fi')
        return lines
    
    # While
    @visit.register
    def _(self, node: While):
        return \
            self.line(f'while {self.visit(node.condition)}; do') + \
            self.visit(node.then) + \
            self.line('done')
            
    # Test
    @visit.register
    def _(self, test: Test) -> str:
        return f'[ {self.visit(test.expression)} -ne 0 ]'
    
    # Print
    @visit.register
    def _(self, node: Print) -> str:
        return self.line(f'echo {self.visit(node.printExpression)}')
        
    # ExpressionStatement
    @visit.register
    def _(self, node: ExpressionStatement):
        return self.line(self.visit(node.expression))
        
    # Variable
    @visit.register
    def _(self, node: Variable) -> str:
        return f'"${node.name}"'
        
    # Literal
    @visit.register
    def _(self, node: Literal) -> str:
        return f'{node.value}'
    
    # Command
    @visit.register
    def _(self, node: Command):
        text = ''
                
        args = f'{' '.join([self.visit(arg) for arg in (node.args or [])])}'
        
        if node.defined_stdout and node.defined_stderr:
            text += (f'capture '
                f'{node.stdout.name} {node.stderr.name} {node.returnCode.name if node.defined_returnCode else "_" } '
                f'{self.visit(node.command)} '
                f'{args}'
            )
        elif node.defined_stdout: # just stdout
            text += f'{node.stdout.name}={self.visit(node.command)} {args}'
        elif node.defined_stderr: # just stderr
            text += f'{node.stderr.name}={self.visit(node.command)} {args} 2>&1 >/dev/null'
        else:
            text += f'{self.visit(node.command)} {args}'
            
        if node.defined_returnCode and not (node.defined_stderr and node.defined_stdout):
            text += f'{node.returnCode.name}=$?'
        
        return text
        
    # BinaryOperation
    @visit.register
    def _(self, node: BinaryOperation) -> str:
        return f'$(({self.visit(node.lhs)} {node.operator.value} {self.visit(node.rhs)}))'
    
    # UnaryOperation
    @visit.register
    def _(self, node:UnaryOperation) -> str:
        return f'$(({node.operator.value}{self.visit(node.expr)}))'
    
    #FunctionCall
    @visit.register
    def _(self, node: FunctionCall):
        return f'$({node.functionName} {' '.join([self.visit(arg) for arg in node.args])})'
        
    #GroupExpression
    @visit.register
    def _(self, node: GroupExpression):
        return f'({self.visit(node.expression)})'
    
class Transpiler:
    def __init__(self):
        self.environment = Environment()
        self.phases: list[Phase] = [
            EnvironmentAssignment(self.environment),
        ]
        self.emitter = ShellRenderer(self.environment)
    
    def transpile(self, rootBlock: RootBlock) -> str:
        node = rootBlock
        for phase in self.phases:
            node = phase.visit_all(node)
        
        assert(isinstance(node, RootBlock))
        if self.environment.needs_capture:
            node.statements.insert(0, Raw(capture))
        
        return self.emitter.visit_all(node)