import ast
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def test_server_inputs_and_http_client_unchanged(self):
        trees=[ast.parse((ROOT/"full_serving"/name).read_text()) for name in ("boundary_control_experiment.py","boundary_transition_experiment.py")]
        functions=[{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)} for t in trees]
        for name in ("server_arguments","copy_inputs","matched_targets","clean"):
            self.assertEqual(ast.dump(functions[0][name]),ast.dump(functions[1][name]),name)
        def client_statements(tree):
            result=[]
            for n in ast.walk(tree):
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="client" for t in n.targets):
                    result.append(ast.dump(n))
                if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=="client":
                    result.append(ast.dump(n))
            return result
        self.assertEqual(client_statements(functions[0]["run"]),client_statements(functions[1]["run"]))

    def test_phase_marker_has_no_future_lengths(self):
        tree=ast.parse((ROOT/"full_serving/boundary_transition_experiment.py").read_text())
        writes=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="write_json":
                if n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=="tmp":
                    writes.append(n)
        self.assertEqual(len(writes),1)
        keys=[x.value for x in writes[0].args[1].keys]
        self.assertEqual(set(keys),{"phase","pilot","candidate_seal","dataset","wall_ns"})
if __name__=="__main__":unittest.main()
