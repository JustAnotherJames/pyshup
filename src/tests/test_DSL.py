import pytest
from pyshup import *

def test_DSL():
    
    script = Script(inject_capture=True)
    with script as s:
        s1 = Variable('s1', 10)
        s2 = Variable('s2', -10)
        
        s1.v = s1 + s2
        
        c = Echo(s1).returnCode
        s2.v = c
        
        Echo(s2, "v0")
        Echo(Command(None, None, "v0").stderr)
        Echo(Command(None, None, "v0").stdout)
        Echo(c)
        
        with If(s1 == s2).Then() as c:
            Echo("return code and value are equal, capture of echo return code is 0:")
            Echo("s1: ")
            Echo(s1)
            Echo("return code: ")
            Echo(s2)
            Echo("return code v2:")
            c.add(AST.ExpressionStatement(AST.Echo(AST.Command(None, None, "v0").returnCode)))
        with c.Else().Then():
            Echo("return code and value are NOT equal, capture of echo return code is NOT 0:")
            Echo("s1: ")
            Echo(s1)
            Echo("return code: ")
            Echo(s2)
            Echo("return code v2:")
            c.add(AST.ExpressionStatement(AST.Echo(AST.Command(None, None, "v0").returnCode)))
        
        with For(s1.set(s2), s1 < 10, s1.set(s1 + 1)).Then():
            Echo(s1)
    
    print(script.render())
    #script.write('./test.sh')
    #res = script.execute()
    #print(res)