from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from enum import Enum
from functools import singledispatchmethod
from typing import TypeVar

"""
program -> rootblock

rootblock -> block | function

function -> Function( param_list )

param_list -> Variable*
            
block -> stmt*

stmt -> assign
      | raw 
      | comment 
      | control 
      | Print 
      | ExprStmt

# note: assignmust must be a statement in our grammar due to python's grammar.
# COULD make it an expression, but DSL would only be able to use it as a statement
# Ex: "if a = 10:" is illegal in python.
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
      | group_expr
    
variable -> Variable(...)

literal -> Literal()

command -> 'capture' variable variable variable expr args 

args -> '(' expr* ')'

binaryOperation -> ArithmeticOperation | ComparisonOperation

ArithmeticOperation -> expr ArithmeticOperator expr

ArithmeticOperator -> '+' | '-' | '*' | '/' | '%'
    
ComparisonOperation -> expr ComparisonOperator expr

ComparisonOperator -> '<' | '<=' | '>' | '>=' | '==' | '~='
                    | 'and' | 'or'

unaryOperation -> unop expr

unop -> '-'
    
group_expr -> '(' expr ')'

"""

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
    class EnvType(Enum):
        Variable = "variables"
        Function = "functions"

    def __init__(self, parent_environment: Environment | None):
        self.root: Environment
        self.parent_environment = None
        if parent_environment is None:
            self.root = self
        else:
            self.parent_environment = parent_environment
            self.root = self.parent_environment.root
            self.parent_environment.child_env.append(self)
        assert self.parent_environment is not None or self.root == self
        assert self.root is not None
        self.child_env: list[Environment] = []

        self.needs_capture = False
        self.functions: dict[str, Function] = {}
        self.variables: dict[str, Statement] = {}

    def _setEnv(self, type: EnvType, name: str, value: Node) -> bool:
        if self._getEnv(type, name) is not None:
            return False
        getattr(self, type.value)[name] = value
        return True

    def _getEnv(self, type: EnvType, name: str) -> Node | None:
        storageType = getattr(self, type.value)
        env: Environment | None = self
        while True:
            if name in storageType:
                t = storageType[name]
                assert t is None or isinstance(t, Node)
                return t
            if env and env.parent_environment:
                env = env.parent_environment
            else:
                break
        return None

    def setFunction(self, name: str, value: Function) -> bool:
        return self._setEnv(self.EnvType.Function, name, value)

    def getFunction(self, name: str) -> Function | None:
        node = self._getEnv(self.EnvType.Function, name)
        assert node is None or isinstance(node, Function)
        return node

    def setVariable(self, name: str, value: Variable) -> bool:
        return self._setEnv(self.EnvType.Variable, name, value)

    def getVariable(self, name: str) -> Variable | None:
        node = self._getEnv(self.EnvType.Variable, name)
        assert node is None or isinstance(node, Variable)
        return node


T = TypeVar("T", bound="Node")


class Node(ABC):
    children: list[str] = []


class CoreNode(Node):
    pass


class RootBlock(CoreNode):
    children = ["functions", "statements"]

    def __init__(self, statements: list[Statement | Function] = []):
        super().__init__()
        self.statements: list[Statement] = []
        self.functions: list[Function] = []

        for x in statements:
            if isinstance(x, Function):
                self.functions.append(x)
            elif isinstance(x, Statement):
                self.statements.append(x)

    def add(self, statement: Statement | Function) -> RootBlock:
        if isinstance(statement, Function):
            self.functions.append(statement)
        elif isinstance(statement, Statement):
            self.statements.append(statement)
        else:
            raise Exception(f"Cannot add {statement} to root block")
        return self


class Function(CoreNode):
    children = ["functionBody"]

    def __init__(
        self,
        functionName: str,
        functionParams: list[Variable] | None = [],
        functionBody: Block | None = None,
    ):
        super().__init__()
        self.functionName = functionName
        self.functionParams = functionParams
        self.functionBody = functionBody or Block()
        self.returnStatement: Return | None = None


class Block(CoreNode):
    children = ["statements"]

    def __init__(self, statements: list[Statement] = []):
        super().__init__()
        self.statements = statements or []

        self.parent: Node | None = None

    def insert(self, index: int, statement: Statement) -> Block:
        self.statements.insert(index, statement)
        return self

    def add(self, statement: Statement) -> Block:
        self.statements.append(statement)
        return self


class Statement(CoreNode):
    pass


class Assign(Statement):
    children = ["lhs", "rhs"]

    def __init__(self, variable: Variable, rhs: Expression):
        super().__init__()
        self.lhs = variable
        self.rhs = rhs


class Return(Statement):
    children = ["expression"]

    def __init__(
        self, expression: Expression | None = None, returnMethod: str = "echo"
    ):
        super().__init__()
        self.returnMethod = returnMethod
        self.expression = expression

        self.function: Function | None = None


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


class While(Control):
    children = ["condition", "then"]

    def __init__(self, condition: Condition, then: Block | None = None):
        super().__init__()
        self.condition = condition
        self.then = then or Block()


class If(Control):
    children = ["ifcondition", "then", "elseifconditions", "elifthens", "elsethen"]

    def __init__(
        self,
        condition: Condition,
        then: Block | None = None,
        elseifs: list[tuple[Condition, Block]] = [],
        elsethen: Block | None = None,
    ):
        super().__init__()
        self.ifcondition = condition
        self.then = then or Block()

        self.elseifconditions: list[Condition] = []
        self.elifthens: list[Block] = []
        for elifcond, elifthen in elseifs:
            self.elseifconditions.append(elifcond)
            self.elifthens.append(elifthen)

        self.elsethen: Block | None = elsethen


class Condition(CoreNode):
    children = ["expression"]

    def __init__(self, expression: Expression):
        super().__init__()
        self.expression = expression


class Print(Statement):
    children = ["args"]

    def __init__(self, args: list[Expression] = [], sep: str = " ", end: str = "\\n"):
        super().__init__()
        self.args = args
        self.sep = sep
        self.end = end


class ExpressionStatement(Statement):
    children = ["expression"]

    def __init__(self, expression: Expression):
        super().__init__()
        self.expression = expression


class Expression(CoreNode):
    pass


class Variable(Expression):
    children = []

    def __init__(self, name: str):
        super().__init__()
        self.name = name


class CommandVariable(Variable):
    children = []

    def __init__(self, name: str):
        super().__init__(name)
        self.command: Command


class Literal(Expression):
    children = []

    def __init__(self, value: int | str | bool | None, raw: bool = False):
        super().__init__()
        self.value = value
        self.type = type(self.value)
        self.raw = raw


class Command(Expression):
    children = ["command", "args"]

    def __init__(
        self,
        command: Expression | None = None,
        args: list[Expression] | None = [],
        name: str | None = None,
    ):
        super().__init__()
        self.command = command
        self.args = args

        characters = string.ascii_letters + string.digits
        self.name = name or f"TMP_{''.join(random.choices(characters, k=5))}"

        self.defined_stdout: CommandVariable | None = None
        self.defined_stderr: CommandVariable | None = None
        self.defined_returnCode: CommandVariable | None = None

    @property
    def stdout(self) -> CommandVariable:
        if not self.defined_stdout:
            self.defined_stdout = CommandVariable(f"{self.name}_stdout")
            self.defined_stdout.command = self
        return self.defined_stdout

    @property
    def stderr(self) -> CommandVariable:
        if not self.defined_stderr:
            self.defined_stderr = CommandVariable(f"{self.name}_stderr")
            self.defined_stderr.command = self
        return self.defined_stderr

    @property
    def returnCode(self) -> CommandVariable:
        if not self.defined_returnCode:
            self.defined_returnCode = CommandVariable(f"{self.name}_returnCode")
            self.defined_returnCode.command = self
        return self.defined_returnCode


class BinaryOperationType(Enum):
    plus = "+"
    minus = "-"
    multiply = "*"
    divide = "/"
    modulo = "%"
    equal = "=="
    notEqual = "!="
    lessThan = "<"
    greaterThan = ">"
    greaterThanEqual = ">="
    lessThanEqual = "<="
    And = "&&"
    Or = "||"
    binaryAnd = "&"
    binaryOr = "|"
    binaryXor = "^"
    binaryShiftLeft = "<<"
    binaryShiftRight = ">>"


class BinaryOperation(Expression):
    children = ["lhs", "rhs"]

    def __init__(self, lhs: Expression, operator: BinaryOperationType, rhs: Expression):
        super().__init__()
        self.lhs = lhs
        self.operator = operator
        self.rhs = rhs


class UnaryOperationType(Enum):
    negate = "-"
    positive = "+"
    binaryInvert = "~"


class UnaryOperation(Expression):
    children = ["expr"]

    def __init__(self, expr: Expression, operator: UnaryOperationType):
        super().__init__()
        self.expr = expr
        self.operator = operator

    negate = UnaryOperationType.negate
    positive = UnaryOperationType.positive
    binaryInvert = UnaryOperationType.binaryInvert


class GroupExpression(Expression):
    children = ["expression"]

    def __init__(self, expression: Expression):
        super().__init__()
        self.expression = expression


class SugarNode(Node):
    pass


# contrived example of a sugar node since it's just a raw passthrough
# real sugarNodes should be handled in another sugarnode-specific phase
class Echo(SugarNode, Command):
    def __init__(self, args: list[Expression] = [], name: str | None = None):
        super().__init__(Literal("echo"), args, name)


def visit_traverse(dispatcher: Phase, node: Node) -> Node:
    for childName in node.children:
        child = getattr(node, childName)
        if not child:
            continue
        elif isinstance(child, Node):
            setattr(node, childName, dispatcher.visit(child, node))
        elif isinstance(child, list):
            children = child
            for i, child in enumerate(children):
                children[i] = dispatcher.visit(child, node)
        else:
            raise Exception("child is not type node or list")
    return node


class Phase:
    @singledispatchmethod
    @abstractmethod
    def visit(self, node: Node, parent: Node) -> Node:
        pass


# for debugging
class Print_AST(Phase):
    def __init__(self) -> None:
        self.string = ""
        self.indent = 0

    @singledispatchmethod
    def visit(self, node: Node, parent: Node) -> Node:
        self.string += "\t" * self.indent + node.__repr__() + "\n"
        self.indent += 1
        visit_traverse(self, node)
        self.indent -= 1
        return node

    def print(self, node: Node) -> str:
        visit_traverse(self, node)
        return self.string


# associates nested blocks, performs variable renaming
# detects when "capture" function needs to be injected
class PrePass(Phase):
    def __init__(self, environment: Environment):
        self.environment = environment

    @singledispatchmethod
    def visit(self, node: Node, parent: Node) -> Node:
        return visit_traverse(self, node)

    # Command
    @visit.register
    def _(self, node: Command, parent: Node) -> Node:
        if node.defined_stdout and node.defined_stderr:
            self.environment.root.needs_capture = True
        return node

    # Block
    @visit.register
    def _(self, node: Block, parent: Node) -> Node:
        node.parent = parent
        return visit_traverse(self, node)

    # Return
    @visit.register
    def _(self, node: Return, parent: Node) -> Node:
        assert isinstance(parent, Block) and parent.parent is not None
        function = parent.parent
        if not isinstance(function, Function):
            raise Exception("Return statement cannot appear outside of Functions")
        function.returnStatement = node
        node.function = function
        if node.expression:
            expression = visit_traverse(self, node.expression)
            assert isinstance(expression, Expression)
            node.expression = expression
        return node

    # Function
    @visit.register
    def _(self, node: Function, parent: Node) -> Node:
        self.environment.setFunction(node.functionName, node)
        for variable in node.functionParams or []:
            self.environment.setVariable(variable.name, variable)
            variable.name = f"{variable.name}_{node.functionName}"
        node.functionBody.parent = node
        functionBody = visit_traverse(self, node.functionBody)
        assert isinstance(functionBody, Block)
        node.functionBody = functionBody
        return node

    # Variable
    @visit.register
    def _(self, node: Variable, parent: Node) -> Node:
        # check if we've been previously registered & mangled
        existingVariable = self.environment.getVariable(node.name)
        if existingVariable:
            node.name = existingVariable.name
            self.environment.setVariable(
                node.name, node
            )  # both names now refer to variable
            return node
        self.environment.setVariable(node.name, node)
        return node


# Phase Emitter
class ShellRenderer:
    def __init__(self, environment: Environment, indent: str = "\t"):
        self.indent: str = indent
        self.level: int = 0
        self.environment = environment

    def line(self, line: str) -> str:
        return f"{self.indent * (self.level - 1)}{line}\n"

    @singledispatchmethod
    def visit(self, node: CoreNode) -> str:
        raise NotImplementedError(f"No visit method for {type(node).__name__}")

    # Root Block
    @visit.register
    def _(self, node: RootBlock) -> str:
        lines: str = "#!/bin/sh\n"
        self.level += 1
        for function in node.functions:
            lines += self.visit(function)

        prev_env = self.environment
        self.environment = Environment(prev_env)
        for stmt in node.statements:
            lines += self.visit(stmt)
        self.environment = prev_env
        self.level -= 1
        return lines

    # Function
    @visit.register
    def _(self, node: Function) -> str:
        # move parameters into function body with proper assignments
        if node.functionParams is not None:
            for i, param in enumerate(node.functionParams):
                node.functionBody.insert(i, Assign(param, Literal(f"${i + 1}")))

        return (
            self.line(f"{node.functionName}()" + "{ ")
            + self.visit(node.functionBody)
            + self.line("}")
        )

    # Return
    @visit.register
    def _(self, node: Return) -> str:
        if not node.expression:
            return ""
        assert node.function is not None

        if node.returnMethod == "echo":
            return self.line(f"echo {self.visit(node.expression)}")
        else:
            raise NotImplementedError(
                f"Error, returnMethod of type {node.returnMethod} is not supported"
            )

    # Block
    @visit.register
    def _(self, node: Block) -> str:
        lines = ""
        self.level += 1
        prev_env = self.environment
        self.environment = Environment(prev_env)
        for stmt in node.statements:
            lines += self.visit(stmt)
        self.environment = prev_env
        self.level -= 1
        return lines

    # Assign
    @visit.register
    def _(self, node: Assign) -> str:
        return self.line(f"{node.lhs.name}={self.visit(node.rhs)}")

    # Raw
    @visit.register
    def _(self, node: Raw) -> str:
        return self.line(node.sh)

    # Comment
    @visit.register
    def _(self, node: Comment) -> str:
        return self.line(f"# {node.comment}")

    # If
    @visit.register
    def _(self, node: If) -> str:
        lines = ""
        if node.ifcondition:
            lines += self.line(f"if {self.visit(node.ifcondition)}; then")
            lines += self.visit(node.then) if node.then else self.line(":")

        for condition, then in zip(node.elseifconditions, node.elifthens):
            lines += self.line(f"elif {self.visit(condition)}; then")
            lines += self.visit(then)

        if node.elsethen:
            lines += self.line("else")
            lines += self.visit(node.elsethen)

        lines += self.line("fi")
        return lines

    # While
    @visit.register
    def _(self, node: While) -> str:
        return (
            self.line(f"while {self.visit(node.condition)}; do")
            + self.visit(node.then)
            + self.line("done")
        )

    # Condition
    @visit.register
    def _(self, test: Condition) -> str:
        return f"[ {self.visit(test.expression)} -ne 0 ]"

    # Print
    @visit.register
    def _(self, node: Print) -> str:
        printStr = 'printf "'
        printStr += node.sep.join(["%s"] * (len(node.args) - 1) + [f"%s{node.end}"])
        printStr += '" ' + " ".join([self.visit(arg) for arg in (node.args or [])])
        return self.line(printStr)

    # ExpressionStatement
    @visit.register
    def _(self, node: ExpressionStatement) -> str:
        return self.line(self.visit(node.expression))

    # Variable
    @visit.register
    def _(self, node: Variable) -> str:
        return f'"${node.name}"'

    # Literal
    @visit.register
    def _(self, node: Literal) -> str:
        if node.raw and isinstance(node.value, str):
            return f"{node.value}"
        if node.type == int | float:
            return f"{node.value}"
        return f'"{node.value}"'

    # Command
    @visit.register
    def _(self, node: Command) -> str:
        text = ""

        args = f"{' '.join([self.visit(arg) for arg in (node.args or [])])}"
        if args:
            args = " " + args

        if node.defined_stdout and node.defined_stderr:
            text += (
                f"capture "
                f"{node.stdout.name} "
                f"{node.stderr.name} "
                f"{node.returnCode.name if node.defined_returnCode else '_'} "
                f"{self.visit(node.command)}"
                f"{args}"
            )
        elif node.defined_stdout:  # just stdout
            text += f"{node.stdout.name}=$({self.visit(node.command)}{args})"
        elif node.defined_stderr:  # just stderr
            text += f"{node.stderr.name}=$({self.visit(node.command)}{args} "
            text += "2>&1 >/dev/null)"
        else:
            text += f"{self.visit(node.command)}{args} "
            text += ">/dev/null 2>&1"

        if node.defined_returnCode and not (
            node.defined_stderr and node.defined_stdout
        ):
            text += f"{node.returnCode.name}=$?"

        return text

    # BinaryOperation
    @visit.register
    def _(self, node: BinaryOperation) -> str:
        return (
            f"$(({self.visit(node.lhs)} {node.operator.value} {self.visit(node.rhs)}))"
        )

    # UnaryOperation
    @visit.register
    def _(self, node: UnaryOperation) -> str:
        return f"$(({node.operator.value}{self.visit(node.expr)}))"

    # GroupExpression
    @visit.register
    def _(self, node: GroupExpression) -> str:
        return f"({self.visit(node.expression)})"


class Transpiler:
    def __init__(self) -> None:
        self.globalEnvironment = Environment(None)
        self.prepass = PrePass(self.globalEnvironment)
        self.emitter = ShellRenderer(self.globalEnvironment)

    def transpile(self, rootBlock: Node) -> str:
        node = rootBlock
        # return Print_AST().print(node)
        node = self.prepass.visit(node, None)
        assert isinstance(node, RootBlock)
        if self.globalEnvironment.needs_capture:
            node.functions.insert(0, Raw(capture))  # type: ignore
        return self.emitter.visit(node)
