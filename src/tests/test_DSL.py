from pyshup import Command, Echo, For, Function, If, Print, Return, Script, Variable
from tests.conftest import shellcheck


def test_DSL() -> None:
    script = Script()
    with script:
        with Function("testFunction", Variable("p"), Variable("q")).Then():
            # this would become part of the return value since it is printed to stdout
            # Print(Variable('p'))
            Return(Variable("q") + 10, "echo")

        v1 = Variable("v1", 10)
        v2 = Variable("v2", 20)
        v3 = Variable("v3", 0)

        v3.v = v1 + v2

        with If(v3 == 30).Then() as c:
            Print("They're Equal!")
        with c.Else().Then() as c:
            Print("Error, your cpu is bugged!", "good luck")

        with For(v3.set(0), v3 < v1 + v2, v3.set(v3 + 1)).Then():
            Print(v3)
        t = Echo("this is a test")
        Print(t.stdout)
        Print(t.stderr)
        Print(t.returnCode)

        v3.v = Command("testFunction", [10, 20]).stdout
        Print(v3)

        v1.v = "one"
        v2.v = "two"
        Print(v1, v2, v3)

    code = script.render()
    print(code)
    shellcheck(code)


if __name__ == "__main__":
    test_DSL()
