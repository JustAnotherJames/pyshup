from __future__ import annotations
import tempfile
import subprocess
from pathlib import Path
from contextvars import ContextVar
from . import AST

#TODO: wrap Function, FunctionCall, Raw, Comment, Command, 

def get_script():
    try:
        return Script.current_script.get()
    except LookupError:
        raise RuntimeError(
            "No active Script context. "
            "Use 'with Script():'"
        )

class Script():
    current_script = ContextVar("current_script")
    
    def __init__(self):
        self.transpiler = AST.Transpiler()
        self.root: AST.RootBlock = AST.RootBlock()
        self.builder = ScriptBuilder(self)
        self._token = None
            
            
    def render(self) -> str:
        return self.transpiler.transpile(self.root)
    
    def write(self, path: Path | str):
        with open(path, "w") as f:
            f.write(self.render())
            
    def execute(self, check=True, env=None, capture_output=True):
        with tempfile.NamedTemporaryFile(mode="w+") as f:
            f.write(self.render())
            f.flush()
            f.seek(0)
            proc = subprocess.run(
                ["sh", f.name],
                capture_output=capture_output,
                text=True,
                check=check,
                env=env,
            )
            return proc
    
    def __enter__(self):
        self._token = Script.current_script.set(self)
        return self.builder
    
    def __exit__(self, *_):
        if self._token:
            Script.current_script.reset(self._token)
        else:
            raise Exception('Error: Script context not entered - Cannot exit')

class ScriptBuilder:
    def __init__(self, script: Script):
        self.script = script
        self.root: AST.RootBlock = self.script.root
        self.scopes: list[AST.RootBlock | AST.Block] = [self.root]
    
    @property
    def current_block(self) -> AST.RootBlock | AST.Block:
        return self.scopes[-1]

    def add(self, stmt: AST.Statement):
        self.current_block.add(stmt)
    
    def start_block(self, block: AST.Block | None = None) -> AST.Block:
        block = block or AST.Block()
        #self.current_block.add(block)
        self.scopes.append(block)
        return block
    
    def end_block(self) -> AST.RootBlock | AST.Block:
        if len(self.scopes) == 1:
            raise Exception('Cannot close root block')
        return self.scopes.pop()

class DSLNode:
    def __init__(self):
        pass
    
    @property
    def script(self) -> Script:
        return get_script()
        
class For(DSLNode):
    def __init__(self, initialization: AST.Assign, condition: AST.Expression, incrementation: AST.Assign):
        super().__init__()
        self.initialization = initialization
        self.condition = condition
        self.incrementaiton = incrementation
        
        self.ast_while: AST.While = AST.While(AST.Test(self.condition))
        self.forThen: ForThen = ForThen(self)
        
        self.script.builder.add(initialization)
        self.script.builder.add(self.ast_while)
    
    def Then(self, block: AST.Block | None = None):
        if block:
            for stmt in block.statements:
                self.forThen.add(stmt)
        return self.forThen

class ForThen(DSLNode):
    def __init__(self, dsl_for: For):
        self.dsl_for = dsl_for
        self.ast_while: AST.While = dsl_for.ast_while
        self.thenBlock = self.ast_while.then
    
    def add(self, statement: AST.Statement):
        if not self.thenBlock:
            raise Exception('Cannot add statement without Then Context')
        self.thenBlock.add(statement)
        
    def __enter__(self):
        self.script.builder.start_block(self.thenBlock)
        return self
    
    def __exit__(self, *_):
        self.script.builder.add(self.dsl_for.incrementaiton)
        self.script.builder.end_block()
        
class While(DSLNode):
    def __init__(self, condition: AST.Expression):
        self.condition = condition
        self.whileThen = WhileThen(self)
        self.ast_while = AST.While(AST.Test(self.condition))
        self.script.builder.add(self.ast_while)
        
    def Then(self, block: AST.Block | None = None):
        if block:
            for stmt in block.statements:
                self.whileThen.add(stmt)
        return self.whileThen
    
class WhileThen(DSLNode):
    def __init__(self, dsl_while: While):
        self.dsl_while = dsl_while
        self.ast_while = self.dsl_while.ast_while
        self.thenBlock = self.ast_while.then
    
    def add(self, statement: AST.Statement):
        if not self.thenBlock:
            raise Exception('Cannot add statement without Then Context')
        self.thenBlock.add(statement)
        
    def __enter__(self):
        self.script.builder.start_block(self.thenBlock)
        return self
    
    def __exit__(self, *_):
        self.script.builder.end_block()
    
        
        
class If(DSLNode):
    def __init__(self, condition: AST.Expression):
        super().__init__()
        self.condition = condition
        self.ast_if = AST.If(AST.Test(self.condition))
        self.script.builder.add(self.ast_if)
        
    def Then(self, block: AST.Block | None = None):
        builder = IfChainBuilder(self.ast_if)
        builder.start_block(self.ast_if.then)
        if block:
            for stmt in block.statements:
                builder.add(stmt)
        return builder
    
class IfChainBuilder(DSLNode):
    def __init__(self, ast_if: AST.If):
        self.ast_if: AST.If = ast_if
        self.current_block: AST.Block | None = None
        
    def start_block(self, thenBlock: AST.Block):
        self.current_block = thenBlock
        self.script.builder.start_block(self.current_block)
        
    def end_block(self):
        self.current_block = None
        self.script.builder.end_block()
        
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.end_block()
            
    def add(self, statement: AST.Statement):
        if not self.current_block:
            raise Exception('Cannot add statement without Then Context')
        self.current_block.add(statement)
            
    def ElseIf(self, condition: AST.Expression):
        if not self.ast_if:
            raise Exception('ElseIf needs If context')
        
        block = AST.Block()
        self.ast_if.elseifconditions.append(AST.Test(condition))
        self.ast_if.elifthens.append(block)
        return IfChainBuilderThen(self, block)
    
    def Else(self):
        if self.ast_if.elsethen is not None:
            raise Exception('Else Block already defined')
        
        block = AST.Block()
        self.ast_if.elsethen = block
        return IfChainBuilderThen(self, block)
    
class IfChainBuilderThen:
    def __init__(self, ifChainBuilder: IfChainBuilder, thenBlock: AST.Block):
        self.ifChainBuilder = ifChainBuilder
        self.thenBlock = thenBlock
        
    def Then(self, block: AST.Block | None = None):
        self.ifChainBuilder.start_block(self.thenBlock)
        if block:
            for stmt in block.statements:
                self.ifChainBuilder.add(stmt)
        return self.ifChainBuilder
    
class DSLStatement(DSLNode):
    def __init__(self, value: AST.Statement | None = None):
        super().__init__()
        if value:
            self._v = value
    
    @classmethod
    def makeASTStatement(cls, value):
        if isinstance(value, DSLStatement):
            statement = value._v
        elif isinstance(value, AST.Statement):
            statement = value
        else:
            raise Exception(f'Error: {value} cannot be converted to AST Statement')
        return statement
    
class Print(DSLStatement):
    def __init__(self, expr):
        expr = DSLExpression.makeASTExpression(expr)
        super().__init__(AST.Print(expr))
        self.script.builder.add(self._v)

class DSLExpression(DSLNode):
    def __init__(self, value = None):
        super().__init__()
        if not value:
            return
        if isinstance(value, AST.Expression):
            self._v = value
        else:
            self._v: AST.Expression = DSLExpression.makeASTExpression(value)
            
    @classmethod
    def makeASTExpression(cls, value) -> AST.Expression: # convert to Expression
        if isinstance(value, AST.Expression):
            expression = value
        elif isinstance(value, DSLExpression):
            expression = value._v
        elif isinstance(value, int) or isinstance(value, str):
            expression = AST.Literal(value)
        else:
            raise Exception(f'Error: {value} cannot be converted to AST Expression')
        return expression
        
    def __add__(self, other):
        other = DSLExpression.makeASTExpression(other)
        return AST.BinaryOperation(self._v, AST.BinaryOperation.plus, other)
    
    def __sub__(self, other):
        other = DSLExpression.makeASTExpression(other)
        return AST.BinaryOperation(self._v, AST.BinaryOperation.minus, other)
    
    def __eq__(self, other): # type: ignore
        other = DSLExpression.makeASTExpression(other)
        return AST.BinaryOperation(self._v, AST.BinaryOperationType.equal, other)
    
    def __lt__(self, other):
        other = DSLExpression.makeASTExpression(other)
        return AST.BinaryOperation(self._v, AST.BinaryOperationType.lessThan, other)
    
    def __gt__(self, other):
        other = DSLExpression.makeASTExpression(other)
        return AST.BinaryOperation(self._v, AST.BinaryOperationType.greaterThan, other)
    
    def __lte__(self, other):
        other = DSLExpression.makeASTExpression(other)
        return AST.BinaryOperation(self._v, AST.BinaryOperationType.lessThanEqual, other)

    def __gte__(self, other):
        other = DSLExpression.makeASTExpression(other)
        return AST.BinaryOperation(self._v, AST.BinaryOperationType.greaterThan, other)
    
class Command(DSLExpression):
    def __init__(self, value = None, args: list | None = None, manual_variable: str | None = None):
        value = DSLExpression.makeASTExpression(value) if value else None
        args = [DSLExpression.makeASTExpression(arg) for arg in (args or [])]
        self._v = AST.Command(value, args, manual_variable)
        
        super().__init__()
        if value:
            self.script.builder.add(AST.ExpressionStatement(self._v))
            
    @property
    def stdout(self):
        assert(isinstance(self._v, AST.Command))
        return self._v.stdout
    @property
    def stderr(self):
        assert(isinstance(self._v, AST.Command))
        return self._v.stderr
    @property
    def returnCode(self):
        assert(isinstance(self._v, AST.Command))
        return self._v.returnCode
    
class Echo(Command):
    def __init__(self, value, name: str | None = None):
        value = DSLExpression.makeASTExpression(value)
        super().__init__("echo", [value], name)

class VariableDescriptor:
    def __get__(self, obj, objtype):
        return obj._v
    
    def __set__(self, obj: Variable, other=None):
        if not isinstance(obj._v, AST.Variable):
            raise Exception('Error: Cannot assign to non-variable type')
        other = DSLExpression.makeASTExpression(other)
        obj.script.builder.add(AST.Assign(obj._v, other))
        
class Variable(DSLExpression):
    v = VariableDescriptor()
    
    def __init__(self, name, initial_value):
        self._v: AST.Variable = AST.Variable(name) # type: ignore
        initial_value = DSLExpression.makeASTExpression(initial_value)
        
        super().__init__()
        if initial_value:
            self.script.builder.add(AST.Assign(self._v, initial_value))
    
    def set(self, other):
        other = DSLExpression.makeASTExpression(other)
        return AST.Assign(self._v, other)