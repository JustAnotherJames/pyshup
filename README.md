# Pyshup

**Pyshup** (pronounced “push-up”) is a Python embedded DSL for generating shell scripts.  
Write structured Python code with Pythonic constructs that transpiles into portable, readable shell scripts.

---

## Features

- Embedded DSL in Python for shell scripting  
- Declarative control flow (`If`, `Else`, loops)  
- Variables, commands, and output handling  
- Generates clean, executable shell scripts  
- Composable and structured, making scripts easier to maintain  

---

## Usage

```python
from pyshup import Script, Variable, If, For, Command, Print

script = Script(inject_capture=True)
with script as s:
    v1 = Variable('v1', 10)
    v2 = Variable('v2', 20)
    v3 = Variable('v3', 0)

    v3.v = v1 + v2

    with If(v3 == 30).Then() as c:
        Print("They're Equal!")
    with c.Else().Then() as c:
        Print("Error, your cpu is bugged")
    
    with For(v3.set(0), v3 < v1 + v2, v3.set(v3 + 1)).Then():
        Print(v3)
    
script.write('./test.sh')
```
And the resulting generated shell:
```bash
v1=10
v2=20
v3=0
v3=$(("$v1" + "$v2"))
if [ $(("$v3" == 30)) -ne 0 ]; then
	echo "They're Equal!"
else
	echo "Error, your cpu is bugged!"
fi
v3=0
while [ $(("$v3" < $(("$v1" + "$v2")))) -ne 0 ]; do
	echo "$v3"
	v3=$(("$v3" + 1))
done
```

---

## Installation

You can install Pyshup via pip:

```bash
pip install pyshup