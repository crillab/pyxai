# import os
import unittest
import logging 
import sys 
import platform

from pyxai.tests.functionality.GetInstances import *
from pyxai.tests.functionality.ToFeatures import *
from pyxai.tests.functionality.Metrics import *
from pyxai.tests.functionality.Rectify import *
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

def linux_tests():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestToFeatures))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestGetInstances))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMetrics))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRectify))
    
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestLearningScikitlearn))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestLearningXGBoost))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestLearningLightGBM))
    
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSaveLoadScikitlearn))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSaveLoadXgboost))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSaveLoadLightGBM))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestImportScikitlearn))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestImportSimpleScikitlearn))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestImportXGBoost))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestImportLightGBM))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMisc))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDT))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRF))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestBT))
    return suite

def windows_tests():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestToFeatures))
    return suite


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    runner = unittest.TextTestRunner(verbosity=2)
    if platform.system() == 'Windows':
        runner.run(windows_tests())
    else:
        runner.run(linux_tests())
