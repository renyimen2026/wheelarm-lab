"""Column projection must retain duplicates, order, and subset index semantics."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from datasets import Dataset
from smolvla_dataset_projection import project_delta_columns


class ProjectionTests(unittest.TestCase):
    def dataset(self, mapping=None):
        hf=Dataset.from_dict({'action':[[0.,1.],[2.,3.],[4.,5.]],
                              'unneeded_image_column':['a','b','c']})
        transformed=[]
        def transform(batch):
            transformed.append(set(batch))
            return {key:[torch.tensor(value) for value in values] if key=='action' else values
                    for key,values in batch.items()}
        hf.set_transform(transform)
        d=SimpleNamespace(hf_dataset=hf,meta=SimpleNamespace(video_keys=['video']),
                          _absolute_to_relative_idx=mapping,
                          _ensure_hf_dataset_loaded=lambda:None,
                          _query_hf_dataset=lambda _:None)
        return project_delta_columns(d),transformed

    def test_only_requested_column_reaches_transform_preserving_order_and_repeats(self):
        data,columns=self.dataset()
        result=data._query_hf_dataset({'action':[2,0,2],'video':[0]})
        self.assertEqual(set(result),{'action'})
        self.assertTrue(torch.equal(result['action'],torch.tensor([[4.,5.],[0.,1.],[4.,5.]])))
        self.assertEqual(columns,[{'action'}])

    def test_absolute_subset_mapping_and_padding_duplicates_are_preserved(self):
        data,_=self.dataset({50:0,51:1,90:2})
        result=data._query_hf_dataset({'action':[90,51,51,50]})['action']
        self.assertTrue(torch.equal(result,torch.tensor([[4.,5.],[2.,3.],[2.,3.],[0.,1.]])))
        with self.assertRaises(KeyError):data._query_hf_dataset({'action':[100]})

    def test_reinstall_rejected_and_missing_columns_are_not_silently_omitted(self):
        data,_=self.dataset()
        with self.assertRaises(ValueError):project_delta_columns(data)
        with self.assertRaises(ValueError):data._query_hf_dataset({'missing':[0]})


if __name__=='__main__':unittest.main(verbosity=2)
