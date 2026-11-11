import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from tensorbalance.cli import main
from tensorbalance.input import load_input
from tensorbalance.export_csv import export_csv

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = np.ones((2, 2, 2))

    def test_binned_metadata_formats_and_resolutions(self):
        for width in (25000, 1000000):
            metadata = dict(genomic_bin_width=width, genomic_bins=[4, 7], chromosome='chr21')
            paths = []
            path = self.root/f'raw_{width}.json'
            path.write_text(json.dumps(dict(raw=self.raw.tolist(), metadata=metadata)))
            paths.append(path)
            path = self.root/f'raw_{width}.npz'
            np.savez(path, raw=self.raw, genomic_bin_width=width, genomic_bins=[4, 7])
            paths.append(path)
            path = self.root/f'raw_{width}.zip'
            buffer = io.BytesIO(); np.save(buffer, self.raw)
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('raw.npy', buffer.getvalue())
                archive.writestr('metadata.json', json.dumps(metadata))
            paths.append(path)
            for source in paths:
                out = self.root/(source.name+'_result')
                self.assertEqual(main([str(source), '--out', str(out)]), 0)
                saved = json.loads((out/'metadata.json').read_text())
                self.assertEqual(saved['genomic_bin_width'], width)
                self.assertEqual(saved['genomic_bins'], [4, 7])
                self.assertEqual(np.load(out/'balanced_tensor.npy').shape, self.raw.shape)
                self.assertEqual(main([str(source), '--bin-size', '10000']), 2)
                csv_out = self.root/(source.name+'_csv')
                export_csv(out, csv_out)
                self.assertEqual((out/'metadata.json').read_bytes(), (csv_out/'metadata.json').read_bytes())

    def test_sidecar_unknown_resolution_and_conflicts(self):
        source = self.root/'raw.npy'; np.save(source, self.raw)
        unknown = {}
        load_input(source, metadata=unknown)
        self.assertEqual(unknown, {})
        sidecar = source.with_suffix('.metadata.json')
        sidecar.write_text(json.dumps(dict(bin_size_bp=25000)))
        metadata = {}; load_input(source, metadata=metadata)
        self.assertEqual(metadata, dict(bin_size_bp=25000))
        for invalid in [dict(bin_size_bp=25000, genomic_bin_width=1000000), dict(genomic_bins=[0]), dict(genomic_bins=[1,0])]:
            sidecar.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                load_input(source)

    @unittest.skipIf(pa is None, 'Optional pyarrow dependency absent')
    def test_segment_parquet_requires_choice_and_changes_binning(self):
        source = self.root/'segments.parquet'
        data = {}
        for segment in 'ABC':
            data[f'seg{segment}_chr'] = ['chr21', 'chr21']
            data[f'seg{segment}_start'] = [0, 25000]
            data[f'seg{segment}_end'] = [2, 25002]
        pq.write_table(pa.table(data), source)
        self.assertEqual(main([str(source)]), 2)
        for width, n in [(25000, 2), (1000000, 1)]:
            out = self.root/f'result_{width}'
            args = [str(source), '--out', str(out), '--bin-size', str(width), '--chromosome', 'chr21', '--unobserved-are-zero']
            self.assertEqual(main(args), 0)
            self.assertEqual(np.load(out/'bias_vector.npy').shape, (n,))
            self.assertEqual(json.loads((out/'metadata.json').read_text())['genomic_bin_width'], width)

    @unittest.skipIf(pa is None, 'Optional pyarrow dependency absent')
    def test_binned_parquet_preserved_and_incomplete_rejected(self):
        indices = np.indices((2,2,2)).reshape(3,-1)
        source = self.root/'binned.parquet'
        table = pa.table(dict(i=indices[0], j=indices[1], k=indices[2], count=self.raw.ravel()))
        table = table.replace_schema_metadata({b'tensorbalance': json.dumps(dict(genomic_bin_width=25000, genomic_bins=[4,7])).encode()})
        pq.write_table(table, source)
        out = self.root/'binned_result'
        self.assertEqual(main([str(source), '--out', str(out)]), 0)
        self.assertEqual(json.loads((out/'metadata.json').read_text())['genomic_bins'], [4,7])
        self.assertEqual(main([str(source), '--bin-size', '1000000']), 2)
        pq.write_table(table.slice(0, 7), source)
        with self.assertRaises(ValueError):
            load_input(source)
