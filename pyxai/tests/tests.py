import unittest
import traceback
import sys

class CustomTextTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_class = None

    def startTest(self, test):
        test_class = test.__class__.__name__
        if test_class != self.current_class:
            self.current_class = test_class
            self.stream.writeln(f"\n=== {test_class} ===")
        super().startTest(test)
            
    def _exc_info_to_string(self, err, test):
        """Converts a sys.exc_info()-style tuple of values into a string."""
        exctype, value, tb = err
        if exctype is NotImplementedError:
            msgLines = 'Not Implemented Error : ' + str(value)
        else:
            tb = self._clean_tracebacks(exctype, value, tb, test)
            tb_e = traceback.TracebackException(
                exctype, value, tb,
                capture_locals=self.tb_locals, compact=True)
            msgLines = list(tb_e.format())
    
            if self.buffer:
                output = sys.stdout.getvalue()
                error = sys.stderr.getvalue()
                if output:
                    if not output.endswith('\n'):
                        output += '\n'
                    msgLines.append(STDOUT_LINE % output)
                if error:
                    if not error.endswith('\n'):
                        error += '\n'
                    msgLines.append(STDERR_LINE % error)
        return ''.join(msgLines)


class CustomTextTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return CustomTextTestResult(self.stream, self.descriptions, self.verbosity)

if __name__ == "__main__":
    loader = unittest.TestLoader()
    tests = loader.discover('./tests', '*.py')
    testRunner = CustomTextTestRunner()
    testRunner.run(tests)


""" Old method by naming old file -> Useful for linux/windows separation ? TODO REMOVE (26/11/2025)
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
        """