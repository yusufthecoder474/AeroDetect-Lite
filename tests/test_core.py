from pathlib import Path
import csv
import cv2
import numpy as np
import pytest
from src.core import *

@pytest.fixture
def picture(tmp_path):
    path=tmp_path/'image.png'; cv2.imwrite(str(path),np.zeros((40,80,3),np.uint8)); return path

def test_image(picture): assert load_image(picture).shape==(40,80,3)
def test_missing(tmp_path):
    with pytest.raises(FileNotFoundError): load_image(tmp_path/'none.jpg')
def test_format(tmp_path):
    p=tmp_path/'a.gif'; p.write_bytes(b'GIF89')
    with pytest.raises(ValueError): load_image(p)
def test_signature(tmp_path):
    p=tmp_path/'a.png'; p.write_text('not an image')
    with pytest.raises(ValueError): load_image(p)
def test_size(picture):
    with pytest.raises(ValueError): load_image(picture,max_bytes=1)
def test_discovery(picture): assert discover(picture.parent)==[picture]
def test_preprocess(picture):
    tensor,t=preprocess(load_image(picture),64); assert tensor.shape==(1,3,64,64); validate_input(tensor,64)
    np.testing.assert_allclose(restore(np.array([[0,16,64,48]]),t),[[0,0,80,40]])
def test_rgb():
    im=np.zeros((32,32,3),np.uint8); im[:,:,2]=255
    t,_=preprocess(im,32); assert t[0,0,0,0]==1 and t[0,2,0,0]==0
@pytest.mark.parametrize('size',[0,33,-32])
def test_invalid_size(size):
    with pytest.raises(ValueError): preprocess(np.zeros((10,10,3),np.uint8),size)
@pytest.mark.parametrize('tensor',[np.zeros((1,3,32,32),np.float64),np.zeros((3,32,32),np.float32),np.full((1,3,32,32),np.nan,np.float32),np.full((1,3,32,32),2,np.float32)])
def test_bad_input(tensor):
    with pytest.raises(ValueError): validate_input(tensor,32)
def test_nms():
    d=np.array([[0,0,10,10,.9,2],[0,0,10,10,.8,2],[0,0,10,10,.7,0]])
    assert len(nms(d))==2
def test_confidence():
    raw=np.zeros((1,84,2),np.float32); raw[0,:4,:]=np.array([[16],[16],[8],[8]]); raw[0,6,:]=[.9,.1]
    assert len(decode(raw,(1,0,0,32,32)))==1
def test_empty_detection(): assert decode(np.zeros((1,84,10)),(1,0,0,32,32)).shape==(0,6)
@pytest.mark.parametrize('shape',[(1,85,10),(2,84,10),(1,84,0)])
def test_bad_output(shape):
    with pytest.raises(ValueError): decode(np.zeros(shape),(1,0,0,32,32))
def test_mapping(): assert COMPATIBLE=={1:3,2:4,5:9,7:6} and 0 not in COMPATIBLE
def test_metrics():
    m=metrics([10,20,30]); assert m['mean_ms']==20 and m['images_per_second']==50 and m['p95_ms']==29
@pytest.mark.parametrize('values',[[],[0],[float('nan')],[-1]])
def test_bad_metrics(values):
    with pytest.raises(ValueError): metrics(values)
def test_csv(tmp_path):
    p=tmp_path/'r.csv'; write_csv(p,[{'mean':1,'backend':'onnx'}]); assert list(csv.DictReader(p.open()))==[{'mean':'1','backend':'onnx'}]
def test_annotations(tmp_path):
    p=tmp_path/'a.txt'; p.write_text('1,2,3,4,0,0,0,0\n5,6,7,8,1,4,1,2\n'); a=annotations(p); assert a.shape==(2,8) and a[0,4]==0

def test_onnx_contract(tmp_path):
    import onnx
    from onnx import helper,TensorProto
    from src.backends import OnnxBackend
    x=helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,3,32,32]); y=helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,3,32,32])
    model=helper.make_model(helper.make_graph([helper.make_node('Identity',['x'],['y'])],'bad',[x],[y]),opset_imports=[helper.make_opsetid('',17)]); model.ir_version=9
    p=tmp_path/'bad.onnx'; onnx.save(model,p)
    with pytest.raises(ValueError): OnnxBackend(p,32)
