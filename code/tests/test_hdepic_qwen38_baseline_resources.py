"""Meaningful resource-policy regressions without GPU/tool/process mutation."""
import copy
from pathlib import Path
import tempfile
import unittest
from code.hdepic_qwen38_baseline import resources as r


def idle_snapshot():
    return {'gpus': [{'index': i, 'uuid': uuid, 'name': '48GB test GPU',
                     'memory_total_mib': 49140, 'memory_used_mib': 2,
                     'memory_free_mib': 48508, 'utilization_percent': 0}
                    for i, uuid in r.AUTHORIZED_GPUS.items()], 'compute_processes': []}


class ResourceTests(unittest.TestCase):
    def test_idle_pair_passes_and_pins_full_uuids(self):
        result = r.assess_idle_pair(idle_snapshot(), environ={})
        self.assertTrue(result['passed'])
        self.assertEqual(result['cuda_visible_devices'], ','.join(r.AUTHORIZED_GPUS.values()))

    def test_occupied_pid_blocks_even_zero_utilization(self):
        snapshot = idle_snapshot()
        snapshot['compute_processes'] = [{'gpu_uuid': r.AUTHORIZED_GPUS[1], 'pid': 42,
                                         'process_name': 'other job', 'used_gpu_memory_mib': 1}]
        with self.assertRaisesRegex(r.ResourceBlocked, 'GPU 1 occupied.*42'):
            r.assess_idle_pair(snapshot, environ={})

    def test_unauthorized_selection_rejected(self):
        with self.assertRaisesRegex(r.ResourceBlocked, 'Only the complete authorized'):
            r.assess_idle_pair(idle_snapshot(), selected_indices=(0, 2), environ={})

    def test_partial_scheduler_allocation_rejected(self):
        for environment in ({'CUDA_VISIBLE_DEVICES': '0'}, {'SLURM_JOB_GPUS': '1'},
                            {'CUDA_VISIBLE_DEVICES': ''}):
            with self.subTest(environment=environment), self.assertRaisesRegex(r.ResourceBlocked, 'does not allocate both'):
                r.assess_idle_pair(idle_snapshot(), environ=environment)

    def test_wrong_uuid_blocks(self):
        snapshot = idle_snapshot(); snapshot['gpus'][0]['uuid'] = 'GPU-unexpected'
        with self.assertRaisesRegex(r.ResourceBlocked, 'UUID mismatch'):
            r.assess_idle_pair(snapshot, environ={})

    def test_low_memory_and_activity_block(self):
        snapshot = idle_snapshot(); snapshot['gpus'][0]['memory_free_mib'] = 6400
        snapshot['gpus'][0]['utilization_percent'] = 100
        with self.assertRaisesRegex(r.ResourceBlocked, '42 GiB'):
            r.assess_idle_pair(snapshot, environ={})

    def test_reversed_uuid_scheduler_pair_passes(self):
        value = ','.join(reversed(list(r.AUTHORIZED_GPUS.values())))
        self.assertTrue(r.assess_idle_pair(idle_snapshot(), environ={'CUDA_VISIBLE_DEVICES': value})['passed'])

    def test_raw_csv_parsers_keep_process_without_memory(self):
        raw = 'GPU-test, 17, occupied job, [N/A]\n'
        self.assertEqual(r.parse_process_csv(raw)[0]['pid'], 17)
        self.assertIsNone(r.parse_process_csv(raw)[0]['used_gpu_memory_mib'])

    def test_disk_gate_preserves_headroom(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(r.ResourceBlocked, 'Disk has'):
                r.require_disk_space(Path(temporary), cache=Path(temporary), free_bytes=r.WEIGHT_BYTES)
            self.assertTrue(r.require_disk_space(Path(temporary), cache=Path(temporary),
                free_bytes=r.WEIGHT_BYTES + r.DISK_HEADROOM_BYTES)['passed'])


if __name__ == '__main__':
    unittest.main()
