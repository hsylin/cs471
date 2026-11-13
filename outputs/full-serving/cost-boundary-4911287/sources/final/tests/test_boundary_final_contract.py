import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from full_serving.boundary_final_contract import (
    EVIDENCE, TOTALS, TREATMENTS, performance_environment, validate_admission,
    validate_targets, verify_launch, verify_unprofiled, diagnostic_environment_keys)


class Tests(unittest.TestCase):
    def fixture(self, root):
        records = {}
        for key in EVIDENCE:
            p = root / (key + ".json")
            data = dict(status="passed")
            if key == "selection_review":
                data.update(selected_fix="exact_unchanged_map_commit_v1", policy_preserving=True,
                            shared_transport_changed=False, final_measurements_used_for_selection=False)
            p.write_text(json.dumps(data))
            records[key] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        protocol = dict(schema="cs471_boundary_final_v1", status="frozen_final",
                        selected_fix="exact_unchanged_map_commit_v1", groups=1,
                        requests_per_dataset=512, output_totals=TOTALS, treatments=list(TREATMENTS))
        pp = root / "protocol.json"; pp.write_text(json.dumps(protocol))
        files={}
        for name in ("boundary_noop_commit", "monitoring_gate_plugin", "logical_history_gpu", "client"):
            path=root/"full_serving"/(name+".py");path.parent.mkdir(exist_ok=True)
            path.write_text(name)
            files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = dict(admitted_runtime_files=files,status="passed", source_revision="a"*40,
                       protocol_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),
                       selected_fix=protocol["selected_fix"], evidence=records)
        rp = root / "admission.json"; rp.write_text(json.dumps(receipt))
        return pp, rp, receipt

    def test_all_treatments_require_review(self):
        with tempfile.TemporaryDirectory() as d:
            pp, rp, receipt = self.fixture(Path(d))
            for treatment in TREATMENTS:
                validate_admission(rp, pp, "a"*40, treatment, Path(d))
            del receipt["evidence"]["cost_controls"]; rp.write_text(json.dumps(receipt))
            with self.assertRaises(AssertionError): validate_admission(rp, pp, "a"*40, "off", Path(d))

    def test_changed_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            pp, rp, receipt = self.fixture(Path(d))
            Path(receipt["evidence"]["noop_correctness"]["path"]).write_text('{"status":"failed"}')
            with self.assertRaises(AssertionError): validate_admission(rp, pp, "a"*40, "b4_noop", Path(d))

    def test_wrong_source_or_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            pp, rp, receipt = self.fixture(Path(d))
            with self.assertRaises(AssertionError): validate_admission(rp, pp, "b"*40, "b4", Path(d))
            pp.write_text(pp.read_text()+" ")
            with self.assertRaises(AssertionError): validate_admission(rp, pp, "a"*40, "b4", Path(d))

    def test_selection_cannot_use_final_results(self):
        with tempfile.TemporaryDirectory() as d:
            pp, rp, receipt = self.fixture(Path(d))
            p = Path(receipt["evidence"]["selection_review"]["path"])
            item = json.loads(p.read_text()); item["final_measurements_used_for_selection"] = True
            p.write_text(json.dumps(item));receipt["evidence"]["selection_review"]["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
            rp.write_text(json.dumps(receipt))
            with self.assertRaises(AssertionError): validate_admission(rp, pp, "a"*40, "b4_noop", Path(d))

    def test_no_diagnostic_environment_leaks(self):
        dirty=dict(CS471_GATE_SHADOW="1", FULL_SERVING_BOUNDARY_FIX_SHADOW="1",
                   FULL_SERVING_BOUNDARY_OBSERVATION="gate_only",
                   FULL_SERVING_BOUNDARY_PROBE="shapes", FULL_SERVING_PAYBACK_BRANCH="skip",
                   FULL_SERVING_PROPOSAL_PILOT="1", PATH="/bin")
        for treatment, (arm, fix) in TREATMENTS.items():
            env=performance_environment(dirty,treatment)
            self.assertEqual(env["FULL_SERVING_BOUNDARY_FIX"],fix)
            self.assertEqual(env["FULL_SERVING_BOUNDARY_FIX_SHADOW"],"0")
            self.assertEqual(env["FULL_SERVING_BOUNDARY_PROBE"],"off")
            self.assertEqual(env["FULL_SERVING_PROPOSAL_PILOT"],"0")
            self.assertNotIn("CS471_GATE_SHADOW",env)
            self.assertNotIn("FULL_SERVING_PAYBACK_BRANCH",env)
            self.assertNotIn("FULL_SERVING_BOUNDARY_OBSERVATION",env)

    def test_exact_frozen_token_totals(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for ds,total in TOTALS.items():
                vals={str(i):total//512+(i<total%512) for i in range(512)}
                (root/(ds+"-measured-lengths.json")).write_text(json.dumps(vals))
            validate_targets(root)
            p=root/"longbench-measured-lengths.json"; vals=json.loads(p.read_text());vals["0"]+=1;p.write_text(json.dumps(vals))
            with self.assertRaises(AssertionError):validate_targets(root)

    def test_changed_candidate_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            pp,rp,receipt=self.fixture(Path(d))
            (Path(d)/"full_serving/boundary_noop_commit.py").write_text("changed")
            with self.assertRaises(AssertionError):
                validate_admission(rp,pp,"a"*40,"b4_noop",Path(d))

    def test_final_launch_rejects_shadows(self):
        for treatment,(arm,fix) in TREATMENTS.items():
            launch=dict(command=["vllm","serve"],profiling_isolation=verify_unprofiled(["vllm","serve"],{}),
                        arm=arm,pilot=False,map_commit=fix,map_shadow="0",host_probe="off",
                        history_shadow=False,payback_branch=None,
                        plugin=None if arm=="off" else "cs471_sustained")
            verify_launch(launch,treatment)
            for key,bad in [("map_shadow","1"),("history_shadow",True),("host_probe","host"),("payback_branch","skip")]:
                changed=dict(launch,**{key:bad})
                with self.assertRaises(AssertionError):verify_launch(changed,treatment)


    def test_profile_settings_removed_but_native_startup_unchanged(self):
        dirty = {"VLLM_TRACE_FUNCTION": "1", "VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1",
                 "VLLM_NVTX_SCOPES_FOR_PROFILING": "1", "CUDA_LAUNCH_BLOCKING": "1",
                 "VLLM_TORCH_PROFILER_DIR": "/private/trace",
                 "VLLM_PROTON_PROFILER_DIR": "/private/proton",
                 "VLLM_PROFILER_CONFIG": "torch",
                 "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
                 "CUDA_VISIBLE_DEVICES": "0,1,2,3", "PATH": "/bin"}
        for treatment in TREATMENTS:
            env = performance_environment(dirty, treatment)
            self.assertEqual(diagnostic_environment_keys(env), [])
            for key in ("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
                        "CUDA_VISIBLE_DEVICES", "PATH"):
                self.assertEqual(env[key], dirty[key])
            verify_unprofiled(["vllm", "serve", "model"], env)

    def test_profiler_cli_variants_and_config_indirection_rejected(self):
        for option in ("--profiler-config", "--profiler-config.profiler=torch",
                       "--profiler_config.profiler=torch",
                       "--profiler-config.torch_profiler_dir=/private/trace",
                       "--profiler-config={}", "--config", "--config=server.yaml"):
            with self.subTest(option=option), self.assertRaises(AssertionError):
                verify_unprofiled(["vllm", "serve", "model", option], {})
        with self.assertRaises(AssertionError):
            verify_unprofiled(["nsys", "profile", "vllm", "serve"], {})

    def test_node_rejects_trace_environment_reintroduced_after_controller(self):
        for key in ("VLLM_TRACE_FUNCTION", "VLLM_CUSTOM_SCOPES_FOR_PROFILING",
                    "VLLM_NVTX_SCOPES_FOR_PROFILING", "CUDA_LAUNCH_BLOCKING",
                    "VLLM_TORCH_PROFILER_DIR", "VLLM_PROTON_PROFILER_DIR",
                    "VLLM_PROFILER_CONFIG"):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                verify_unprofiled(["vllm", "serve", "model"], {key: "1"})

    def test_launch_audit_requires_proof_and_rechecks_command(self):
        launch = dict(arm="off",pilot=False,map_commit="native",map_shadow="0",
                      host_probe="off",history_shadow=False,payback_branch=None,
                      plugin=None,command=["vllm","serve","model"],
                      profiling_isolation=verify_unprofiled(["vllm","serve","model"],{}))
        verify_launch(launch, "off")
        bad=copy.deepcopy(launch);del bad["profiling_isolation"]
        with self.assertRaises(KeyError):verify_launch(bad,"off")
        bad=copy.deepcopy(launch);bad["profiling_isolation"]["checked"]=False
        with self.assertRaises(AssertionError):verify_launch(bad,"off")
        bad=copy.deepcopy(launch);bad["command"].append("--profiler-config.profiler=torch")
        with self.assertRaises(AssertionError):verify_launch(bad,"off")

if __name__=="__main__":unittest.main()
