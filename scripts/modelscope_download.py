#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('facebook/sam3',cache_dir='/home/wdblink/Project/pyflyt-drone/models')