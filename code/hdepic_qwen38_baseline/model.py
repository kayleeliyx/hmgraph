"""Pinned exact official model metadata; full weights only after idle GPU gate."""
from pathlib import Path
import hashlib,json,os,urllib.request

ROOT=Path('/data/yaxuanli/ActGround')
RUN=ROOT/'runs/hdepic_qwen38_27b_baseline_v1'
REPO='Qwen/Qwen3.8-27B'
REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
CACHE=Path('/tmp/hdepic-qwen38-27b-hf')
METADATA=('config.json','generation_config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja','model.safetensors.index.json','README.md')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metadata():
    folder=RUN/'model_metadata';folder.mkdir(parents=True,exist_ok=True)
    api=folder/'official_model_api.json'
    if not api.exists():
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}',timeout=60) as r:
            api.write_bytes(r.read())
    info=json.loads(api.read_text());assert info['sha']==REVISION and info['id']==REPO
    for name in METADATA:
        path=folder/name
        if path.exists():continue
        with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',timeout=60) as r:
            temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(r.read());temporary.replace(path)
    cfg=json.loads((folder/'config.json').read_text())
    assert cfg['model_type']=='qwen3_5'
    identity={'repo_id':REPO,'revision':REVISION,'kind':'Official released post-trained multimodal checkpoint; no ActGround adapter',
       'architecture':cfg['architectures'],'metadata_path':str(folder),
       'metadata_sha256':{name:sha(folder/name) for name in METADATA},
       'weight_shards':sorted({*json.loads((folder/'model.safetensors.index.json').read_text())['weight_map'].values()}),
       'remote_safetensors':info.get('safetensors'),'remote_used_storage_bytes':info.get('usedStorage'),
       'weights_downloaded':False,'no_substitute_checkpoint':True,
       'source_url':f'https://huggingface.co/{REPO}/tree/{REVISION}'}
    destination=RUN/'model_identity.json'
    if destination.exists():assert json.loads(destination.read_text())==identity,'Model identity changed'
    else:destination.write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity,indent=2))

def fetch_weights():
    from huggingface_hub import snapshot_download
    # Caller must have already passed the authorized idle-GPU and disk gates.
    identity=json.loads((RUN/'model_identity.json').read_text())
    folder=snapshot_download(REPO,revision=REVISION,cache_dir=str(CACHE),
                             allow_patterns=[*METADATA,*identity['weight_shards']],max_workers=2)
    return Path(folder)

if __name__=='__main__':metadata()
