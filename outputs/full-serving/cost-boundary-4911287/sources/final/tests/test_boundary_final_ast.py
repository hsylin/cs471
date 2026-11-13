"""Protect the existing server/data/client path while adding admission guards."""
import ast
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def test_native_functions_unchanged(self):
        before=ast.parse((ROOT/"full_serving/boundary_experiment.py").read_text())
        after=ast.parse((ROOT/"full_serving/boundary_final_experiment.py").read_text())
        a={x.name:x for x in before.body if isinstance(x,ast.FunctionDef)}
        b={x.name:x for x in after.body if isinstance(x,ast.FunctionDef)}
        for name in ("server_arguments","copy_inputs","matched_targets","clean"):
            self.assertEqual(ast.dump(a[name]),ast.dump(b[name]),name)
        def client(tree):
            return [ast.dump(n) for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AugAssign))
                    and ((isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="client" for t in n.targets))
                         or (isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=="client"))]
        self.assertEqual(client(a["run"]),client(b["run"]))
if __name__=="__main__":unittest.main()
