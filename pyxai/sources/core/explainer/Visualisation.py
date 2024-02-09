from collections import OrderedDict
from pyxai.sources.core.tools.utils import check_PyQt6


class Visualisation:
    def __init__(self, explainer, do_history=True):
        self._history = OrderedDict()
        self._do_history = do_history
        self._explainer = explainer

    @property
    def explainer(self):
        return self._explainer


    def get_PILImage(self, instance, reason, image=None, time_series=None, contrastive=False):
        feature_names = self._explainer.get_feature_names()
        if time_series is not None:
            for key in time_series.keys():
                for feature in time_series[key]:
                    if feature not in feature_names:
                        raise ValueError("The feature " + str(
                            feature) + " in the `time_series` parameter is not an available feature name.")

        if image is not None:
            from pyxai.sources.core.tools.vizualisation import PyPlotImageGenerator
            pyplot_image_generator = PyPlotImageGenerator(image)
            instance_image = pyplot_image_generator.generate_instance(instance, pil_image=True)
            image = pyplot_image_generator.generate_explanation(instance, self._explainer.to_features(reason, details=True,
                                                                                           contrastive=contrastive),
                                                                pil_image=True)
            return [instance_image, image]
        else:
            from pyxai.sources.core.tools.vizualisation import PyPlotDiagramGenerator
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
        PILImage_list = self.get_PILImage(instance, reason, image, time_series, contrastive)
        for i, image in enumerate(PILImage_list):
            PILImage_list[i] = self.resize_PILimage(PILImage_list[i], width)

        if len(PILImage_list) == 1:
            PILImage_list[0].save(file)
        else:
            for i, image in enumerate(PILImage_list):
                image.save(str(i) + "_" + file)

    def resize_PILimage(self, image, width=250):
        from PIL import Image
        wpercent = (width / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((width, hsize), Image.Resampling.LANCZOS)
        return image

    def notebook(self, instance, reason, image=None, time_series=None, contrastive=False, width=250):
        PILImage_list = self.get_PILImage(instance, reason, image, time_series, contrastive)
        from IPython.display import display
        for i, image in enumerate(PILImage_list):
            PILImage_list[i] = self.resize_PILimage(PILImage_list[i], width)

        if len(PILImage_list) == 1:
            display(PILImage_list[0])
        else:
            display(*PILImage_list)

    def screen(self, instance, reason, image=None, time_series=None, contrastive=False, width=250):
        PILImage_list = self.get_PILImage(instance, reason, image, time_series, contrastive)
        for i, image in enumerate(PILImage_list):
            PILImage_list[i] = self.resize_PILimage(PILImage_list[i], width)
        for image in PILImage_list:
            image.show()

    def gui(self, image=None, time_series=None):
        feature_names = self._explainer.get_feature_names()
        if time_series is not None:
            for key in time_series.keys():
                for feature in time_series[key]:
                    if feature not in feature_names:
                        raise ValueError("The feature " + str(
                            feature) + " in the `time_series` parameter is not an available feature name.")
        check_PyQt6()
        from pyxai.sources.core.tools.GUIQT import GraphicalInterface
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
