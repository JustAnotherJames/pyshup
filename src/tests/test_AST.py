import pytest
from pyshup.AST import *

def test_AST():
    
    root = RootBlock([
        ExpressionStatement(Echo(Literal('hi'))),
        
        Assign(Variable('v1'), Literal(10)),
        Assign(Variable('v2'), Literal(20)),
        Assign(Variable('v1'), Variable('v2')),
        Assign(Variable('v1'), BinaryOperation(Literal(1), BinaryOperation.plus, Literal(2))),
        
        If(Test(BinaryOperation(Variable('v1'), BinaryOperationType.equal, Variable('v2'))), then=Block([
            Assign(Variable('v2'), Variable('v1')),
        ])),
        
        While(Test(BinaryOperation(Variable('v1'), BinaryOperationType.lessThan, Literal(10))), Block([
            Assign(Variable('v2'), Variable('v1'))
        ])),
        
        ExpressionStatement(Command(Literal('echo'), [Literal('Hello World')], 'v_command')),
        
        Assign(Variable('v3'), Command(None, None, 'v_command').returnCode),
        
        Function('test', Block([ExpressionStatement(Literal('hi'))])),
        
        ExpressionStatement(FunctionCall('test', [Variable('one'), FunctionCall('two', []), Literal('three')])),
        
        Assign(Variable('v4'), GroupExpression(BinaryOperation(Literal(1), BinaryOperation.plus, Literal(2)))),
        
        Comment('this is a comment'),
    ])
    
    transpiler = Transpiler()
    code = transpiler.transpile(root)
    print(code)