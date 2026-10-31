import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('prepare_assets',Path(__file__).resolve().parents[1]/'scripts/prepare_assets.py')
assets=importlib.util.module_from_spec(spec);spec.loader.exec_module(assets)

def test_existing_asset_is_not_downloaded(tmp_path,monkeypatch):
    path=tmp_path/'model.pt';path.write_bytes(b'verified upstream content')
    def unexpected(*args,**kwargs):raise AssertionError('No download for an existing file')
    monkeypatch.setattr(assets.urllib.request,'urlopen',unexpected)
    assets.fetch('https://example.invalid/model',path)
    assert path.read_bytes()==b'verified upstream content'

def test_failed_download_cleans_partial_file(tmp_path,monkeypatch):
    path=tmp_path/'asset.zip';partial=path.with_suffix('.zip.part');partial.write_bytes(b'old interrupted download')
    def broken(*args,**kwargs):raise OSError('offline')
    monkeypatch.setattr(assets.urllib.request,'urlopen',broken)
    with pytest.raises(OSError):assets.fetch('https://example.invalid/model',path)
    assert not partial.exists() and not path.exists()
