import pytest
from pyshup.AST import *

def test_AST():
    
    root = RootBlock([
        Function('test', None, Block([ExpressionStatement(Literal('hi'))])),
        
        ExpressionStatement(Print([Literal('hi')])),
        
        Assign(Variable('v1'), Literal(10)),
        Assign(Variable('v2'), Literal(20)),
        Assign(Variable('v1'), Variable('v2')),
        Assign(Variable('v1'), BinaryOperation(Literal(1), BinaryOperationType.plus, Literal(2))),
        
        If(Condition(BinaryOperation(Variable('v1'), BinaryOperationType.equal, Variable('v2'))), then=Block([
            Assign(Variable('v2'), Variable('v1')),
        ])),
        
        While(Condition(BinaryOperation(Variable('v1'), BinaryOperationType.lessThan, Literal(10))), Block([
            Assign(Variable('v2'), Variable('v1'))
        ])),
        
        ExpressionStatement(Command(Literal('echo'), [Literal('Hello World')], 'v_command')),
        
        Assign(Variable('v3'), Command(None, None, 'v_command').returnCode),
        
        
        ExpressionStatement(Command(Literal('test'), [Variable('one'), Variable('two'), Variable('three')])),
        
        Assign(Variable('v4'), GroupExpression(BinaryOperation(Literal(1), BinaryOperationType.plus, Literal(2)))),
        
        Comment('this is a comment'),
    ])
    
    transpiler = Transpiler()
    code = transpiler.transpile(root)
    print(code)
    
if __name__ == "__main__":
    test_AST()