# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

# 所有model在__init__.py中导入，是为了自动注册到Registers中

# 1. BackBone
from .backbone import TIMM

# 2. ImageClassification
from .ImageClassification import TIMMC

# 3. AnomalyDetection
from .anomaly import PaDiM, PaDiM2, PaDiM_demo, PaDiM2_demo, PaDiM2_export, PaDiM_export

# 4. SemanticSegmentation
from .SemanticSegmentation import Unet
from .SemanticSegmentation import UnetPlusPlus
from .SemanticSegmentation import PSPNet
from .SemanticSegmentation import PSPNet2
from .SemanticSegmentation import DeepLabV3
from .SemanticSegmentation import DeepLabV3Plus
from .SemanticSegmentation import DeepLabV3Plus2

# 5. Object Detection
from .det import YOLOX

