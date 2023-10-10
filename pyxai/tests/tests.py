# import os
import unittest

from pyxai.tests.functionality.GetInstances import *
from pyxai.tests.functionality.ToFeatures import *
from pyxai.tests.learning.ScikitLearn import *
from pyxai.tests.learning.LightGBM import *
from pyxai.tests.learning.XGBoost import *
from pyxai.tests.saveload.XGBoost import *
from pyxai.tests.saveload.ScikitLearn import *
from pyxai.tests.saveload.LightGBM import *
from pyxai.tests.importing.ScikitLearn import *
from pyxai.tests.importing.SimpleScikitLearn import *
from pyxai.tests.importing.LightGBM import *
from pyxai.tests.importing.XGBoost import *
from pyxai.tests.explaining.dt import *
from pyxai.tests.explaining.misc import *
from pyxai.tests.explaining.rf import *
from pyxai.tests.explaining.bt import *
from pyxai.tests.explaining.regressionbt import *

def suite():
    suite = unittest.TestSuite()
    
    # suite.addTest(unittest.makeSuite(TestToFeatures))
    # suite.addTest(unittest.makeSuite(TestGetInstances))
    #
    # suite.addTest(unittest.makeSuite(TestLearningScikitlearn))
    # suite.addTest(unittest.makeSuite(TestLearningXGBoost))
    # suite.addTest(unittest.makeSuite(TestLearningLightGBM))
    #
    # suite.addTest(unittest.makeSuite(TestSaveLoadScikitlearn))
    # suite.addTest(unittest.makeSuite(TestSaveLoadXgboost))
    # suite.addTest(unittest.makeSuite(TestSaveLoadLightGBM))
    #
    # suite.addTest(unittest.makeSuite(TestImportScikitlearn))
    # suite.addTest(unittest.makeSuite(TestImportSimpleScikitlearn))
    # suite.addTest(unittest.makeSuite(TestImportXGBoost))
    # suite.addTest(unittest.makeSuite(TestImportLightGBM))
    # suite.addTest(unittest.makeSuite(TestMisc))
    # suite.addTest(unittest.makeSuite(TestDT))
    # suite.addTest(unittest.makeSuite(TestRF))
    return suite


if __name__ == '__main__':
    
    unittest.main(verbosity=2)
