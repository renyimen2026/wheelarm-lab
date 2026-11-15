import argparse,json,time,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smolvla_dataset_projection import project_delta_columns
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from datasets import Image
parser=argparse.ArgumentParser(description='Check full input parity and PNG decoding cost of delta-column projection')
parser.add_argument('dataset',type=Path,help='Converted LeRobot training split')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise ValueError('Output exists')
root=args.dataset.resolve(strict=True)
torch.set_num_threads(1)
def make(episodes=None):return LeRobotDataset('wheelarm-lab/grasp_hold',root=root,episodes=episodes,delta_timestamps={'action':[i/10 for i in range(50)]})
a=make();b=project_delta_columns(make())
a._ensure_hf_dataset_loaded()
eps=a.meta.episodes
indices=set([0,51,262,len(a)-1])
for ep in range(a.num_episodes):
 row=eps[ep];start=int(row['dataset_from_index']);end=int(row['dataset_to_index'])
 indices.update([start,min(start+49,end-1),max(start,end-50),end-1])
indices=sorted(indices)
original=Image.decode_example;counts=[0]
def counted(self,*args,**kw):
 counts[0]+=1;return original(self,*args,**kw)
Image.decode_example=counted
old_seconds=new_seconds=0;old_decodes=new_decodes=0
for index in indices:
 counts[0]=0;t=time.perf_counter();x=a[index];old_seconds+=time.perf_counter()-t;old_decodes+=counts[0]
 counts[0]=0;t=time.perf_counter();y=b[index];new_seconds+=time.perf_counter()-t;new_decodes+=counts[0]
 assert x.keys()==y.keys()
 for key in x:
  if isinstance(x[key],torch.Tensor):assert torch.equal(x[key],y[key]),(index,key)
  else:assert x[key]==y[key],(index,key)
# Exercise the absolute-to-relative mapping in a nonzero episode subset too.
c=make([1,3]);d=project_delta_columns(make([1,3]))
for index in [0,1,len(c)-2,len(c)-1]:
 x,y=c[index],d[index]
 for key in x:
  if isinstance(x[key],torch.Tensor):assert torch.equal(x[key],y[key]),('subset',index,key)
  else:assert x[key]==y[key]
report={'status':'passed','full_sample_count':len(indices),'subset_sample_count':4,'all_keys_bitwise_equal':True,'includes':'current RGB, state, action[50,5], action_is_pad, task text and indices','old_seconds':old_seconds,'projected_seconds':new_seconds,'old_png_decodes':old_decodes,'projected_png_decodes':new_decodes,'speedup':old_seconds/new_seconds,'sources':['https://github.com/huggingface/lerobot/issues/4392','https://github.com/huggingface/lerobot/issues/2895']}
with args.output.open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps(report),flush=True)
