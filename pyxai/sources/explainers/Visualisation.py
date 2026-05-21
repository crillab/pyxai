from collections import OrderedDict
from pyxai.sources.tools.utils import check_PyQt6


class Visualisation:
    def __init__(self, explainer, do_history=True):
        self._history = OrderedDict()
        self._do_history = do_history
        self._explainer = explainer

    @property
    def explainer(self):
        return self._explainer


    def get_PILImage(self, instance, reason, image=None, time_series=None, contrastive=False):
        r"""Return PIL images of the PIL library for an instance given in the instance parameter an explanation given in the reason parameter.

            Displays for an instance given in the instance parameter an explanation given in the reason parameter. 
            The given instance must be without the label. The given reason must be in Boolean variable form (without using the to_features() method). 
            The explanation can be shown differently for images and time series, using the parameters provided.

            Parameters
            ----------
            instance: ```list[float]```
                The instance to be explained.

            reason: ```list[int]```
                A set of (signed) binary variables. Reason (explanation) to display.
            
            image: ```dict``` (optional, default=```None```)
                Python dictionary containing some information specific to images. 
                See the page of the ```ImageDict``` class for more information.  

            time_series: ```dict``` (optional, default=```None```)
                To display time series. 
                Python dictionary where a key is the name of a time serie and each value of a key is a list containing time series feature names.

            contrastive: ```Boolean``` (optional, default=```False```)
                True if the explanation is a contrastive one. 
                When this parameter is set to True, the elimination of redundant features must be reversed. 

            width: ```int``` (optional, default=```250```)
                The width parameter specifies the width in pixels of the resulting image. 
                The width value causes the image height to be scaled proportionally to the requested width (the aspect ratio is preserved).
                    
            Examples
            --------
            >>> from pyxai import Learning, Explaining

            >>> # Machine learning part
            >>> learner = Learning.Scikitlearn("tests/datasets/australian_0.csv", problem_type=Learning.CLASSIFICATION) # Dataset deleted
            >>> model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF)
            >>> instance, prediction = learner.get_instances(model, n=1, seed=11200, is_correct=True)

            >>> australian_types = {
            >>>     "numerical": Learning.DEFAULT,
            >>>     "categorical": {"A4*": (1, 2, 3), 
            >>>                     "A5*": tuple(range(1, 15)),
            >>>                     "A6*": (1, 2, 3, 4, 5, 7, 8, 9), 
            >>>                     "A12*": tuple(range(1, 4))},
            >>>     "binary": ["A1", "A8", "A9", "A11"],
            >>> }

            >>> # Explainer part
            >>> explainer = Explaining.initialize(model, instance, features_type=australian_types)
            >>> majoritary_reason = explainer.majoritary_reason(time_limit=10)
            >>> explainer.visualisation.get_PILImage(instance, majoritary_reason)

            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/visualisation/visualisation/)
        """
        feature_names = self._explainer.get_feature_names()
        if time_series is not None:
            for key in time_series.keys():
                for feature in time_series[key]:
                    if feature not in feature_names:
                        raise ValueError("The feature " + str(
                            feature) + " in the `time_series` parameter is not an available feature name.")

        if image is not None:
            from pyxai.sources.tools.vizualisation import PyPlotImageGenerator
            pyplot_image_generator = PyPlotImageGenerator(image)
            instance_image = pyplot_image_generator.generate_instance(instance, pil_image=True)
            explanation_image = pyplot_image_generator.generate_explanation(instance, self._explainer.to_features(reason, details=True, contrastive=contrastive), pil_image=True)
            return [instance_image, explanation_image]
        else:
            from pyxai.sources.tools.vizualisation import PyPlotDiagramGenerator
            pyplot_image_generator = PyPlotDiagramGenerator(time_series=time_series)

            feature_values = dict()
            for i, value in enumerate(instance):
                feature_values[feature_names[i]] = value

            image = pyplot_image_generator.generate_explanation(feature_values, instance,
                                                                self._explainer.to_features(reason, details=True,
                                                                                 contrastive=contrastive),
                                                                pil_image=True)
            return [image]

    def save_png(self, file, instance, reason, image=None, time_series=None, contrastive=False, width=250):
        r"""Save PNG image files in the current directory (given via the ```file``` parameter) with the PIL library for an instance given in the instance parameter an explanation given in the reason parameter.
        
            Displays for an instance given in the instance parameter an explanation given in the reason parameter. 
            The given instance must be without the label. 
            The given reason must be in Boolean variable form (without using the to_features() method). 
            The explanation can be shown differently for images and time series, using the parameters provided.

            Parameters
            ----------
            file: ```str```
                The filename (possibly with a path) where the image is saved.  
                The file path must exist. 

            instance: ```list[float]```
                The instance to be explained.

            reason: ```list[int]```
                A set of (signed) binary variables. Reason (explanation) to display.
            
            image: ```dict``` (optional, default=```None```)
                Python dictionary containing some information specific to images. 
                See the page of the ```ImageDict``` class for more information.  

            time_series: ```dict``` (optional, default=```None```)
                To display time series. 
                Python dictionary where a key is the name of a time serie and each value of a key is a list containing time series feature names.

            contrastive: ```Boolean``` (optional, default=```False```)
                True if the explanation is a contrastive one. 
                When this parameter is set to True, the elimination of redundant features must be reversed. 

            width: ```int``` (optional, default=```250```)
                The width parameter specifies the width in pixels of the resulting image. 
                The width value causes the image height to be scaled proportionally to the requested width (the aspect ratio is preserved).
                    
            Examples
            --------
            >>> from pyxai import Learning, Explaining

            >>> # Machine learning part
            >>> learner = Learning.Scikitlearn("tests/datasets/australian_0.csv", problem_type=Learning.CLASSIFICATION) # Dataset deleted
            >>> model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF)
            >>> instance, prediction = learner.get_instances(model, n=1, seed=11200, is_correct=True)

            >>> australian_types = {
            >>>     "numerical": Learning.DEFAULT,
            >>>     "categorical": {"A4*": (1, 2, 3), 
            >>>                     "A5*": tuple(range(1, 15)),
            >>>                     "A6*": (1, 2, 3, 4, 5, 7, 8, 9), 
            >>>                     "A12*": tuple(range(1, 4))},
            >>>     "binary": ["A1", "A8", "A9", "A11"],
            >>> }

            >>> # Explainer part
            >>> explainer = Explaining.initialize(model, instance, features_type=australian_types)
            >>> majoritary_reason = explainer.majoritary_reason(time_limit=10)
            >>> explainer.visualisation.save_png("save/explanation1.png", instance, majoritary_reason)

            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/visualisation/visualisation/)
        """
        PILImage_list = self.get_PILImage(instance, reason, image, time_series, contrastive)
        for i, image in enumerate(PILImage_list):
            PILImage_list[i] = self.resize_PILimage(PILImage_list[i], width)

        if len(PILImage_list) == 1:
            PILImage_list[0].save(file)
        else:
            for i, image in enumerate(PILImage_list):
                image.save(str(i) + "_" + file)

    def resize_PILimage(self, image, width=250):
        r"""Resize a PIL image. 

            Parameters
            ----------
            image: ```PILimage```
                Image from the [PIL library](https://pillow.readthedocs.io/en/latest/reference/Image.html).

            width: ```int``` (optional, default=```250```)
                The width parameter specifies the width in pixels of the resulting image. 
                The width value causes the image height to be scaled proportionally to the requested width (the aspect ratio is preserved).
                    
            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/visualisation/visualisation/)
        """
        from PIL import Image
        wpercent = (width / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((width, hsize), Image.Resampling.LANCZOS)
        return image

    def notebook(self, instance, reason, image=None, time_series=None, contrastive=False, width=250):
        r"""Use the IPython.display() method to display images in a Jupyter notebook. 
        
            Displays for an instance given in the instance parameter an explanation given in the reason parameter. 
            The given instance must be without the label. 
            The given reason must be in Boolean variable form (without using the to_features() method). 
            The explanation can be shown differently for images and time series, using the parameters provided.


            Parameters
            ----------
            instance: ```list[float]```
                The instance to be explained.

            reason: ```list[int]```
                A set of (signed) binary variables. Reason (explanation) to display.
            
            image: ```dict``` (optional, default=```None```)
                Python dictionary containing some information specific to images. 
                See the page of the ```ImageDict``` class for more information.  

            time_series: ```dict``` (optional, default=```None```)
                To display time series. 
                Python dictionary where a key is the name of a time serie and each value of a key is a list containing time series feature names.

            contrastive: ```Boolean``` (optional, default=```False```)
                True if the explanation is a contrastive one. 
                When this parameter is set to True, the elimination of redundant features must be reversed. 

            width: ```int``` (optional, default=```250```)
                The width parameter specifies the width in pixels of the resulting image. 
                The width value causes the image height to be scaled proportionally to the requested width (the aspect ratio is preserved).
                    
            Examples
            --------
            >>> from pyxai import Learning, Explaining

            >>> # Machine learning part
            >>> learner = Learning.Scikitlearn("tests/datasets/australian_0.csv", problem_type=Learning.CLASSIFICATION) # Dataset deleted
            >>> model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF)
            >>> instance, prediction = learner.get_instances(model, n=1, seed=11200, is_correct=True)

            >>> australian_types = {
            >>>     "numerical": Learning.DEFAULT,
            >>>     "categorical": {"A4*": (1, 2, 3), 
            >>>                     "A5*": tuple(range(1, 15)),
            >>>                     "A6*": (1, 2, 3, 4, 5, 7, 8, 9), 
            >>>                     "A12*": tuple(range(1, 4))},
            >>>     "binary": ["A1", "A8", "A9", "A11"],
            >>> }

            >>> # Explainer part
            >>> explainer = Explaining.initialize(model, instance, features_type=australian_types)
            >>> majoritary_reason = explainer.majoritary_reason(time_limit=10)
            >>> explainer.visualisation.notebook(instance, majoritary_reason)

            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/visualisation/visualisation/)
        """
        PILImage_list = self.get_PILImage(instance, reason, image, time_series, contrastive)
        from IPython.display import display
        for i, image in enumerate(PILImage_list):
            PILImage_list[i] = self.resize_PILimage(PILImage_list[i], width)

        if len(PILImage_list) == 1:
            display(PILImage_list[0])
        else:
            display(*PILImage_list)

    def screen(self, instance, reason, image=None, time_series=None, contrastive=False, width=250):
        r"""Use the Image.show() method to display images on screen with the PIL library. 

            Displays for an instance given in the instance parameter an explanation given in the reason parameter. 
            The given instance must be without the label. The given reason must be in Boolean variable form (without using the to_features() method). 
            The explanation can be shown differently for images and time series, using the parameters provided.

            Parameters
            ----------
            instance: ```list[float]```
                The instance to be explained.

            reason: ```list[int]```
                A set of (signed) binary variables. Reason (explanation) to display.
            
            image: ```dict``` (optional, default=```None```)
                Python dictionary containing some information specific to images. 
                See the page of the ```ImageDict``` class for more information.  

            time_series: ```dict``` (optional, default=```None```)
                To display time series. 
                Python dictionary where a key is the name of a time serie and each value of a key is a list containing time series feature names.

            contrastive: ```Boolean``` (optional, default=```False```)
                True if the explanation is a contrastive one. 
                When this parameter is set to True, the elimination of redundant features must be reversed. 

            width: ```int``` (optional, default=```250```)
                The width parameter specifies the width in pixels of the resulting image. 
                The width value causes the image height to be scaled proportionally to the requested width (the aspect ratio is preserved).
                    
            Examples
            --------
            >>> from pyxai import Learning, Explaining

            >>> # Machine learning part
            >>> learner = Learning.Scikitlearn("tests/datasets/australian_0.csv", problem_type=Learning.CLASSIFICATION) # Dataset deleted
            >>> model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF)
            >>> instance, prediction = learner.get_instances(model, n=1, seed=11200, is_correct=True)

            >>> australian_types = {
            >>>     "numerical": Learning.DEFAULT,
            >>>     "categorical": {"A4*": (1, 2, 3), 
            >>>                     "A5*": tuple(range(1, 15)),
            >>>                     "A6*": (1, 2, 3, 4, 5, 7, 8, 9), 
            >>>                     "A12*": tuple(range(1, 4))},
            >>>     "binary": ["A1", "A8", "A9", "A11"],
            >>> }

            >>> # Explainer part
            >>> explainer = Explaining.initialize(model, instance, features_type=australian_types)
            >>> majoritary_reason = explainer.majoritary_reason(time_limit=10)
            >>> explainer.visualisation.screen(instance, majoritary_reason)

            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/visualisation/visualisation/)
        """
        PILImage_list = self.get_PILImage(instance, reason, image, time_series, contrastive)
        for i, image in enumerate(PILImage_list):
            PILImage_list[i] = self.resize_PILimage(PILImage_list[i], width)
        for image in PILImage_list:
            image.show()

        

    def gui(self, image=None, time_series=None):
        r"""Open the PyXAI’s Graphical User Interface.

            Parameters
            ----------    
            image: ```dict``` (optional, default=```None```)
                Python dictionary containing some information specific to images. 
                See the page of the ```ImageDict``` class for more information.  

            time_series: ```dict``` (optional, default=```None```)
                To display time series. 
                Python dictionary where a key is the name of a time serie and each value of a key is a list containing time series feature names.
        
            Examples
            --------
            >>> ...
            >>> import numpy

            >>> def get_pixel_value(instance, x, y, shape):
            >>>     index = x * shape[0] + y 
            >>>     return instance[index]

            >>> def instance_index_to_pixel_position(i, shape):
            >>>     return i // shape[0], i % shape[0]
            
            >>> explainer.visualisation.gui(image={"shape": (28,28),
            >>>           "dtype": numpy.uint8,
            >>>           "get_pixel_value": get_pixel_value,
            >>>           "instance_index_to_pixel_position": instance_index_to_pixel_position})

            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/visualisation/visualisation/)
        """
        feature_names = self._explainer.get_feature_names()
        if time_series is not None:
            for key in time_series.keys():
                for feature in time_series[key]:
                    if feature not in feature_names:
                        raise ValueError("The feature " + str(
                            feature) + " in the `time_series` parameter is not an available feature name.")
        check_PyQt6()

        from pyxai.sources.tools.GUIQT import GraphicalInterface
        graphical_interface = GraphicalInterface(self, image=image, time_series=time_series)
        graphical_interface.mainloop()
        
    def heat_map(self, name, reasons, contrastive=False):
        dict_heat_map = {}
        if isinstance(reasons, dict):
            # Case2: Dict with weights
            dict_heat_map = reasons
        elif isinstance(reasons, (tuple, list)):
            # Case1: List or Tuple of reasons
            for c in reasons:
                for lit in c:
                    if lit not in dict_heat_map:
                        dict_heat_map[lit] = 1
                    else:
                        dict_heat_map[lit] += 1
        else:
            raise ValueError(
                "The 'reasons' parameter must be either a list of reasons or a dict such as reasons[literal]->weight.")

        instance = self.explainer.instance
        if instance is not None and not isinstance(instance, tuple):
            instance = tuple(instance)

        reasons = [self._explainer.to_features(dict_heat_map, details=True, contrastive=contrastive)]
        if not self._do_history:
            return

        if reasons is None or len(reasons) == 0:
            return
        if (instance, self._explainer.target_prediction) in self._history.keys():
            self._history[(instance, self._explainer.target_prediction)].append(("HeatMap", name, reasons))
        else:
            self._history[(instance, self._explainer.target_prediction)] = [("HeatMap", name, reasons)]

    def add_history(self, instance, class_name, method_name, reasons):
        if not self._do_history:
            return

        if instance is not None and not isinstance(instance, tuple):
            instance = tuple(instance)

        if reasons is None or len(reasons) == 0:
            return
        if not isinstance(reasons[0], tuple): reasons = [reasons]
        reasons = [
            self._explainer.to_features(reason, details=True, contrastive=True if "contrastive" in method_name else False)
            for reason in reasons]
        if (instance, self._explainer.target_prediction) in self._history.keys():
            self._history[(instance, self._explainer.target_prediction)].append((class_name, method_name, reasons))
        else:
            self._history[(instance, self._explainer.target_prediction)] = [(class_name, method_name, reasons)]
