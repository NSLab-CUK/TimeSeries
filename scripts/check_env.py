"""Verify the installed environment and the four requested GPUs without training."""
import json,os,sys,platform
from pathlib import Path
import torch,numpy,pandas,openpyxl
def main():
 ids=[int(x) for x in os.environ.get('GPU_IDS','0,1,2,3').split(',')]
 assert len(ids)==len(set(ids))==4,'GPU_IDS must contain four different GPU IDs'
 assert torch.__version__.startswith('2.5.1'),torch.__version__
 assert torch.version.cuda=='12.4',torch.version.cuda
 assert torch.cuda.is_available(),'CUDA unavailable: check NVIDIA driver and visible GPUs'
 records=[]
 for i in ids:
  assert i<torch.cuda.device_count(),f'GPU {i} unavailable; visible count={torch.cuda.device_count()}'
  value=torch.arange(16,device=f'cuda:{i}').sum().item(); assert value==120
  records.append({'id':i,'name':torch.cuda.get_device_name(i),'memory_bytes':torch.cuda.get_device_properties(i).total_memory})
 report={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':str(torch.__version__),
  'cuda_runtime':torch.version.cuda,'numpy':numpy.__version__,'pandas':pandas.__version__,'openpyxl':openpyxl.__version__,
  'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'GPUs':records,'passed':True,'training_started':False}
 if len(sys.argv)>1: Path(sys.argv[1]).write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
