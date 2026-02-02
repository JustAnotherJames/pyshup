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
                n.parent = self
                #self.children.append(n)
        else:
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
    def __init__(self, name):
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
    children = ['command', 'args', 'stdout', 'stderr', 'returnCode']
    def __init__(self, command: Expression | None = None, args: list[Expression] | None = None, name: str | None = None):
        super().__init__()
        self.command = self.adopt(command)
        self.args = self.adopt(args)
        
        characters = string.ascii_letters + string.digits
        self.name = name or f"TMP_{''.join(random.choices(characters, k=5))}"
        if self.command:
            Command.variable_map[self.name] = {
                'stdout': self.adopt(Variable(f'{self.name}_stdout')),
                'stderr': self.adopt(Variable(f'{self.name}_stderr')),
                'returnCode':  self.adopt(Variable(f'{self.name}_returnCode'))
            }
            
    # this is a bad hack to allow captures from one line of the AST to be addressed in other lines
    # this is somewhat necessary because we can only assign 1 value at a time, but Command returns 3 values
    #i.e
    # [
    #  Command('myVar1', ...cmd...)
    #  Assign(Variable('v1'), Command(None, None, 'myVar1').stdout),
    #  Assign(Variable('v2'), Command(None, None, 'myVar1').stderr),
    #  Assign(Variable('v3'), Command(None, None, 'myVar1').returnCode) 
    # ]
    # TODO: move variable map into actual variable table
    variable_map = {}
        
    @property
    def stdout(self) -> Variable:
        return Command.variable_map.get(self.name, {}).get('stdout', None)
    
    @property
    def stderr(self) -> Variable:
        return Command.variable_map.get(self.name, {}).get('stderr', None)
    
    @property
    def returnCode(self) -> Variable:
        return Command.variable_map.get(self.name, {}).get('returnCode', None)
        
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
    @abstractmethod
    def visit(cls, rootNode: Node, *args, **kwargs) -> Node:
        pass
    
    @abstractmethod
    def visit_one(self, node: Node) -> Node:
        pass
    
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
        
# First Phase: Validate - (this is just an example for now to make sure parents and children are set correctly)
class Validator(Phase):
    @singledispatchmethod
    def visit_node(self, node: Node) -> Node:
        # confirm AST was built correctly
        assert(node.parent)
        #assert(node in node.parent.children or node.parent == node)
        return node
    
    def visit_one(self, node: Node):
        return self.visit_node(node)
    
    def visit(self, rootNode: Node) -> Node:
        return self.visit_traverse(rootNode)
    
# Second Phase: Desugaring - desugar SugarNodes into lower CoreNodes in AST
class Desugarer(Phase):
    @singledispatchmethod
    def visit_sugarNode(self, node: Node):
        assert(isinstance(node, CoreNode) and not isinstance(node, SugarNode))
        return node
    
    def visit(self, rootNode: Node) -> CoreNode:
        #no non-core nodes will exist, safe to cast
        return cast(CoreNode, self.visit_traverse(rootNode))
    
    def visit_one(self, node: Node) -> Node:
        return self.visit_sugarNode(node)
    
    @visit_sugarNode.register
    def _(self, node: Echo):
        return Command(Literal('echo'), [node.command] if node.command else None, node.name)

# Third Phase: assign Variables() to actual variable table to keep track of their names and usage for future phase optimization
#TODO:
class VariableTable(Phase):
    pass
    
# Fourth Phase: Optimize - remove uncessary nodes (i.e echo with no capture semantics should just result in echo)
#TODO:
class Optimizer(Phase):
    @singledispatchmethod
    def visit_node(self, node: Node):
        return node
    
    @visit_node.register
    def _(self, node: Command):
        return node
        
# Last Phase Emitter
class ShellRenderer:
    def __init__(self, indent='\t'):
        self.indent: str = indent
        self.level: int = 0
        
    def line(self, line: str) -> str:
        return f'{self.indent*(self.level - 1)}{line}\n'
        
    @singledispatchmethod
    def visit_coreNode(self, node: CoreNode):
        raise NotImplementedError(f"No visit_coreNode method for {type(node).__name__}")
    
    def visit(self, node: CoreNode) -> str:
        return self.visit_coreNode(node)
    
    # Root Block
    @visit_coreNode.register
    def _(self, node: RootBlock):
        lines = ''
        self.level += 1
        for function in node.functions:
            lines += self.visit_coreNode(function)
            
        for stmt in node.statements:
            lines += self.visit_coreNode(stmt)
        self.level -= 1
        return lines
        
    # Function
    @visit_coreNode.register
    def _(self, node: Function):
        return \
            self.line(f'{node.functionName}()' + '{') + \
            self.visit_coreNode(node.functionBody) + \
            self.line('}')
    
    # Block
    @visit_coreNode.register
    def _(self, node: Block):
        lines = ''
        self.level += 1
        for stmt in node.statements:
            lines += self.visit_coreNode(stmt)
        self.level -= 1
        return lines
        
    # Assign
    @visit_coreNode.register
    def _(self, node: Assign):
        return self.line(f'{node.lhs.name}={self.visit_coreNode(node.rhs)}')    
    
    # Raw
    @visit_coreNode.register
    def _(self, node:Raw):
        return self.line(node.sh)
    
    # Comment
    @visit_coreNode.register
    def _(self, node: Comment):
        return self.line(f'# {node.comment}')
        
    # If
    @visit_coreNode.register
    def _(self, node: If):
        lines = ''
        if node.ifcondition:
            lines += self.line(f'if {self.visit_coreNode(node.ifcondition)}; then')
            lines += self.visit_coreNode(node.then) if node.then else self.line(':')

        for (condition, then) in zip(node.elseifconditions, node.elifthens):
            lines += self.line(f'elif {self.visit_coreNode(condition)}; then')
            lines += self.visit_coreNode(then)
                
        if node.elsethen:
            lines += self.line('else')
            lines += self.visit_coreNode(node.elsethen)
                
        lines += self.line(f'fi')
        return lines
    
    # While
    @visit_coreNode.register
    def _(self, node: While):
        return \
            self.line(f'while {self.visit_coreNode(node.condition)}; do') + \
            self.visit_coreNode(node.then) + \
            self.line('done')
            
    # Test
    @visit_coreNode.register
    def _(self, test: Test) -> str:
        return f'[ {self.visit_coreNode(test.expression)} -ne 0 ]'
    
    # Print
    @visit_coreNode.register
    def _(self, node: Print) -> str:
        return self.line(f'echo {self.visit_coreNode(node.printExpression)}')
        
    # ExpressionStatement
    @visit_coreNode.register
    def _(self, node: ExpressionStatement):
        return self.line(self.visit_coreNode(node.expression))
        
    # Variable
    @visit_coreNode.register
    def _(self, node: Variable) -> str:
        return f'"${node.name}"'
        
    # Literal
    @visit_coreNode.register
    def _(self, node: Literal) -> str:
        return f'{node.value}'
    
    # Command
    @visit_coreNode.register
    def _(self, node: Command):
        return (f'capture '
            f'{node.stdout.name} {node.stderr.name} {node.returnCode.name} '
            f'{self.visit_coreNode(node.command)} '
            f'{' '.join([self.visit_coreNode(arg) for arg in (node.args or [])])}'
        )
        
    # BinaryOperation
    @visit_coreNode.register
    def _(self, node: BinaryOperation) -> str:
        return f'$(({self.visit_coreNode(node.lhs)} {node.operator.value} {self.visit_coreNode(node.rhs)}))'
    
    # UnaryOperation
    @visit_coreNode.register
    def _(self, node:UnaryOperation) -> str:
        return f'$(({node.operator.value}{self.visit_coreNode(node.expr)}))'
    
    #FunctionCall
    @visit_coreNode.register
    def _(self, node: FunctionCall):
        return f'$({node.functionName} {' '.join([self.visit_coreNode(arg) for arg in node.args])})'
        
    #GroupExpression
    @visit_coreNode.register
    def _(self, node: GroupExpression):
        return f'({self.visit_coreNode(node.expression)})'
    
class Transpiler:
    def __init__(self):
        self.phases: list[Phase] = [
            Validator(),
            Desugarer(),
        ]
        self.emitter = ShellRenderer()
    
    def transpile(self, node: Node) -> str:
        for phase in self.phases:
            node = phase.visit(node)
        
        assert(isinstance(node, CoreNode))
        return self.emitter.visit(node)