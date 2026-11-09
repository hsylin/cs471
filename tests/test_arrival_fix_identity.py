"""Torch and NVML must identify the same devices without weakening the pin."""
import unittest
from arrival_fix.common import canonical_gpu_uuid, GPU_UUIDS


class IdentityTests(unittest.TestCase):
    def test_torch_and_nvml_uuid_formats_match(self):
        for identity in GPU_UUIDS:
            self.assertEqual(canonical_gpu_uuid(identity), identity)
            self.assertEqual(canonical_gpu_uuid(identity[4:]), identity)
        self.assertNotEqual(canonical_gpu_uuid(GPU_UUIDS[0]), canonical_gpu_uuid(GPU_UUIDS[1]))

    def test_malformed_or_mig_identity_rejected(self):
        for value in ('', '0', 'MIG-' + GPU_UUIDS[0][4:], None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_gpu_uuid(value)
