from utils_modified import *

obj = MultimodalDataset('MOSEI', visual='facet', audio='covarep', text='embeddings', pivot='words', sentiments=False, emotions=True, max_len=20)